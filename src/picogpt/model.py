# ruff: noqa: N812, N806, PLR0913
import abc
import enum
import inspect
import logging
from collections.abc import Iterator
from typing import Literal, no_type_check

import torch
import torch.nn.functional as F
import transformers
from torch import nn, optim
from torch.nn import init

from picogpt import constants

logger = logging.getLogger(__name__)


class Type(enum.StrEnum):
    CHAR_BIGRAM = enum.auto()
    CHAR_TRANSFORMER = enum.auto()
    GPT2 = enum.auto()


class LanguageModel(nn.Module, abc.ABC):
    TYPE: Type

    def __init__(self, context_size: int) -> None:
        super().__init__()
        self.context_size = context_size

    @abc.abstractmethod
    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """idx and targets both have shape (B, T)"""

    def _sample_next(self, idx: torch.Tensor, temperature: float, top_k: int | None) -> torch.Tensor:
        # crop idx  to the last context_size tokens
        idx_cropped = idx[:, -self.context_size :]
        logits, _ = self(idx_cropped)  # (B, T, vocab_size)
        # focus on the last timestep
        logits = logits[:, -1, :]  # (B, vocab_size)
        # apply temperature
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        if top_k is None:
            # sample from the full distribution
            return torch.multinomial(probs, num_samples=1)  # (B, 1)

        # samples into the (B, K) distributions, NOT (B, vocab_size)!
        # meaning ix is sampled indices into topk_indices
        topk_probs, topk_indices = torch.topk(probs, k=top_k, dim=-1)  # (B, K)
        ix = torch.multinomial(topk_probs, num_samples=1)  # (B, 1)
        # essentially topk_indices[b, ix]
        return torch.gather(topk_indices, dim=-1, index=ix)  # (B, 1)

    def generate_stream(
        self,
        idx: torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = 50,
    ) -> Iterator[torch.Tensor]:
        """Yield next token tensor (B, 1) at each step."""
        for _ in range(max_new_tokens):
            idx_next = self._sample_next(idx, temperature, top_k)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=-1)

            yield idx_next

    def generate(
        self,
        idx: torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = 50,
    ) -> torch.Tensor:
        """Non-streaming generate. Returns full sequence."""
        for idx_next in self.generate_stream(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        ):
            idx = torch.cat((idx, idx_next), dim=-1)

        return idx


class CharBigram(LanguageModel):
    TYPE = Type.CHAR_BIGRAM

    def __init__(self, *, context_size: int, vocab_size: int) -> None:
        super().__init__(context_size)
        self.embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        logits = self.embedding(idx)  # (B, T, vocab_size)

        # batch size x time steps x channels/embeddings/features
        # instead of batch x time x channels, we group batch x time so the rows
        # become individual embeddings at each time step (essentially treating
        # them as individual examples)
        # corresponding with each embedding row is a single label index, so we
        # stretch out targets
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss


class AttentionHead(nn.Module):
    def __init__(self, *, context_size: int, embedding_size: int, head_size: int, dropout: float) -> None:
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(embedding_size, head_size, bias=False)
        self.query = nn.Linear(embedding_size, head_size, bias=False)
        self.value = nn.Linear(embedding_size, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(context_size, context_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, T, _ = x.shape
        x_q = self.query(x)  # (B, T, C) @ (C, head_size) -> (B, T, head_size)
        x_k = self.key(x)  # same
        x_v = self.value(x)  # same

        # attention scores/affinities
        attn = x_q @ x_k.transpose(-2, -1) * self.head_size**-0.5  # (B, T, T)
        attn = attn.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # ty:ignore[not-subscriptable]
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        # weighted aggregation of the values
        return attn @ x_v  # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        *,
        context_size: int,
        num_heads: int,
        head_size: int,
        embedding_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            AttentionHead(
                context_size=context_size, embedding_size=embedding_size, head_size=head_size, dropout=dropout
            )
            for _ in range(num_heads)
        )
        self.proj = nn.Linear(num_heads * head_size, embedding_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x has shape (B, T, embedding_size)"""
        # cat((B, T, head_size) x num_heads) -> (B, T, head_size * num_heads)
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return self.dropout(out)


class FeedForward(nn.Module):
    """a simple feed-forward network"""

    def __init__(self, embedding_size: int, projection_factor: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_size, projection_factor * embedding_size),
            nn.ReLU(),
            nn.Linear(projection_factor * embedding_size, embedding_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """a simple transfomer block: communication followed by computation, also a light touch of residual connections"""

    def __init__(
        self, *, context_size: int, num_heads: int, embedding_size: int, ffw_projection_factor: int, dropout: float
    ) -> None:
        super().__init__()
        head_size = embedding_size // num_heads
        self.sa = MultiHeadAttention(
            context_size=context_size,
            num_heads=num_heads,
            head_size=head_size,
            embedding_size=embedding_size,
            dropout=dropout,
        )
        self.ffwd = FeedForward(embedding_size=embedding_size, projection_factor=ffw_projection_factor, dropout=dropout)
        self.ln1 = nn.LayerNorm(embedding_size)
        self.ln2 = nn.LayerNorm(embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))  # pre-norm formulation
        return x + self.ffwd(self.ln2(x))


class CharTransformer(LanguageModel):
    TYPE = Type.CHAR_TRANSFORMER

    def __init__(
        self,
        *,
        context_size: int,
        vocab_size: int,
        embedding_size: int,
        num_blocks: int,
        num_heads: int,
        ffw_projection_factor: int,
        dropout: float,
    ) -> None:
        super().__init__(context_size)
        self.embed = nn.Embedding(vocab_size, embedding_size)
        self.pos_embed = nn.Embedding(context_size, embedding_size)
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    context_size=context_size,
                    num_heads=num_heads,
                    embedding_size=embedding_size,
                    ffw_projection_factor=ffw_projection_factor,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ],
        )
        self.ln_f = nn.LayerNorm(embedding_size)
        self.lm_head = nn.Linear(embedding_size, vocab_size)

        self.apply(self._init_weights)

    @torch.inference_mode()
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, T = idx.shape
        tok_embed = self.embed(idx)  # (B, T, embedding_size)
        pos_embed = self.pos_embed(torch.arange(T, dtype=torch.long, device=idx.device))  # (T, embedding_size)
        embeddings = tok_embed + pos_embed  # (B, T, embedding_size) broadcast over B
        embeddings = self.blocks(embeddings)  # (B, T, embedding_size)
        embeddings = self.ln_f(embeddings)  # (B, T, embedding_size)
        logits = self.lm_head(embeddings)  # (B, T, vocab_size)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss


class GPT2CausalSelfAttention(nn.Module):
    def __init__(self, embedding_size: int, num_heads: int) -> None:
        super().__init__()
        if embedding_size % num_heads != 0:
            msg = f"embedding_size {embedding_size} not divisible by num_heads {num_heads}"
            raise ValueError(msg)

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(embedding_size, 3 * embedding_size, bias=True)
        # regularization
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        # output project
        self.c_proj = nn.Linear(embedding_size, embedding_size)
        self.c_proj.SCALE_INIT = 1  # ty:ignore[unresolved-attribute]
        # ruff: disable[ERA001]
        # self.register_buffer(
        #     "bias", torch.tril(torch.ones(context_size, context_size).view(1, 1, context_size, context_size))
        # )
        # ruff: enable[ERA001]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x has shape (B, T, C)."""
        B, T, C = x.size()
        x_qkv = self.c_attn(x)  # (B, T, C) @ (C, 3 * C) = (B, T, 3 * C)
        x_q, x_k, x_v = x_qkv.split(self.embedding_size, dim=-1)  # (B, T, C) each
        x_q = x_q.view(B, T, self.num_heads, C // self.num_heads).transpose(-2, -3)  # (B, nh, T, hs)
        x_k = x_k.view(B, T, self.num_heads, C // self.num_heads).transpose(-2, -3)  # (B, nh, T, hs)
        x_v = x_v.view(B, T, self.num_heads, C // self.num_heads).transpose(-2, -3)  # (B, nh, T, hs)
        # attention  (materializes a large (T, T) matrix for all queries and keys)
        # ruff: disable[ERA001]
        # attn = x_q @ x_k.transpose(-2, -1) * x_k.size(-1) ** -0.5  # (B, nh, T, T)
        # attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        # attn = attn.softmax(dim=-1)
        # y = attn @ x_v  # (B, nh, T, hs)
        # ruff: enable[ERA001]
        y = F.scaled_dot_product_attention(
            x_q,
            x_k,
            x_v,
            is_causal=True,
        )  # (B, nh, T, hs)

        y = y.transpose(-2, -3).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        # output projection
        return self.c_proj(y)


class GPT2MLP(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.c_fc = nn.Linear(in_features, out_features, bias=True)  # up projection
        # GPT2 used tanh approx for being faster historically, not need in contemporary settings
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(out_features, in_features, bias=True)  # down projection
        self.c_proj.SCALE_INIT = 1  # ty:ignore[unresolved-attribute]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x has shape (B, T, C)."""
        x = self.c_fc(x)
        x = self.gelu(x)
        return self.c_proj(x)


class GPT2Block(nn.Module):
    def __init__(self, embedding_size: int, num_heads: int, ffw_projection_factor: int) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(embedding_size)
        self.attn = GPT2CausalSelfAttention(embedding_size, num_heads)
        self.ln_2 = nn.LayerNorm(embedding_size)
        self.mlp = GPT2MLP(embedding_size, ffw_projection_factor * embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x has shape (B, T, C)."""
        # pre-norm inputs
        # prefer clean (no-norm) residual pathway from learning signal directly to input
        x = x + self.attn(self.ln_1(x))  # (B, T, C)
        return x + self.mlp(self.ln_2(x))  # (B, T, C)


GPT2Type = Literal["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]


class GPT2(LanguageModel):
    TYPE = Type.GPT2

    def __init__(
        self,
        context_size: int,
        vocab_size: int,
        embedding_size: int,
        num_layers: int,
        num_heads: int,
        ffw_projection_factor: int,
    ) -> None:
        super().__init__(context_size)
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.num_layers = num_layers
        self.transformer = nn.ModuleDict(
            {
                # token embedding
                "wte": nn.Embedding(vocab_size, embedding_dim=embedding_size),
                # position embedding
                "wpe": nn.Embedding(context_size, embedding_dim=embedding_size),
                # hidden
                "h": nn.ModuleList(
                    GPT2Block(embedding_size, num_heads, ffw_projection_factor) for _ in range(num_layers)
                ),
                "ln_f": nn.LayerNorm(embedding_size),
            }
        )
        self.lm_head = nn.Linear(embedding_size, vocab_size, bias=False)

        # weight sharing between token embedding and output projection
        self.transformer.wte.weight = self.lm_head.weight  # ty:ignore[invalid-assignment]

        # init params
        self.apply(self._init_weights)

    @torch.inference_mode()
    def _init_weights(self, module: nn.Module) -> None:
        """Mirror GPT-2 parameter initialization."""
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "SCALE_INIT"):
                # the number of residual layers is num of gpt2 blocks times 2 because
                # each block has two residual paths (attention and feedforward)
                std *= (2 * self.num_layers) ** -0.5
            init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            init.normal_(module.weight, mean=0.0, std=0.02)

    @no_type_check
    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, T = idx.size()
        if self.context_size < T:
            msg = f"input sequence length {T} exceeds model context size {self.context_size}"
            raise ValueError(msg)

        tok_embed = self.transformer.wte(idx)  # (B, T, C)
        pos_embed = self.transformer.wpe(torch.arange(T, dtype=torch.long, device=idx.device))  # (T, C)
        embeddings = tok_embed + pos_embed  # (B, T, C) broadcast over B
        for block in self.transformer.h:
            embeddings = block(embeddings)  # (B, T, C)
        embeddings = self.transformer.ln_f(embeddings)  # (B, T, C)
        logits = self.lm_head(embeddings)  # (B, T, vocab_size)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type: GPT2Type) -> "GPT2":
        logger.info("loading weights from pretrained model %s", model_type)
        config_args = constants.GPT2_PRETRAINED_CONFIG[model_type]
        # share vocab size, context size, and feedforward projection factor
        config_args["vocab_size"] = constants.GPT2_PRETRAINED_VOCAB_SIZE
        config_args["context_size"] = constants.GPT2_PRETRAINED_CONTEXT_SIZE
        config_args["ffw_projection_factor"] = constants.GPT2_PRETRAINED_FFW_PROJECTION_FACTOR
        model = GPT2(
            context_size=config_args["context_size"],
            vocab_size=config_args["vocab_size"],
            embedding_size=config_args["embedding_size"],
            num_layers=config_args["num_layers"],
            num_heads=config_args["num_heads"],
            ffw_projection_factor=config_args["ffw_projection_factor"],
        )
        sd = model.state_dict()
        sd_keys = list(filter(lambda k: not k.endswith(".attn.bias"), sd.keys()))  # discard buffer

        # init hugging face model
        model_hf = transformers.GPT2LMHeadModel.from_pretrained(model_type)
        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_hf = model_hf.state_dict()
        # discard buffers
        sd_keys_hf = list(
            filter(lambda k: not (k.endswith(".attn.masked_bias") and k.endswith(".attn.bias")), sd_hf.keys())
        )
        transposed = ["attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        if len(sd_keys) != len(sd_keys_hf):
            msg = "mismatch in number of state dict keys between HF and local model"
            raise ValueError(msg)

        with torch.inference_mode():
            for k in sd_keys_hf:
                if any(k.endswith(x) for x in transposed):
                    if sd_hf[k].shape[::-1] != sd[k].shape:
                        msg = f"shape mismatch for transposed weight {k}: {sd_hf[k].shape} vs {sd[k].shape}"
                        raise ValueError(msg)
                    sd[k].copy_(sd_hf[k].t())
                else:
                    # vanilla copy
                    if sd_hf[k].shape != sd[k].shape:
                        msg = f"shape mismatch for weight {k}: {sd_hf[k].shape} vs {sd[k].shape}"
                        raise ValueError(msg)
                    sd[k].copy_(sd_hf[k])

        return model

    def create_optimizer(self, weight_decay: float, lr: float, device: torch.device) -> optim.AdamW:
        """Creates AdamW optimizer (fused if available) with parameter groups for weight decay."""
        # start with all parameters that require gradients
        param_dict: dict[str, nn.Parameter] = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        # create optim groups. any 2d params will be decayed (weights and embeddings),
        # all biases and layernorm won't
        decay_params: list[nn.Parameter] = [p for pn, p in param_dict.items() if p.ndim >= 2]  # noqa: PLR2004
        no_decay_params: list[nn.Parameter] = [p for pn, p in param_dict.items() if p.ndim < 2]  # noqa: PLR2004
        optim_groups: list[dict[str, list[nn.Parameter] | float]] = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_no_decay_params = sum(p.numel() for p in no_decay_params)
        logger.info("num decay param tensors: %d, with %d params", len(decay_params), num_decay_params)
        logger.info("num no decay param tensors: %d, with %d params", len(no_decay_params), num_no_decay_params)
        # create fused adamw optimizer if available
        fused_available = "fused" in inspect.signature(optim.AdamW).parameters
        use_fused = fused_available and device.type == "cuda"
        logger.info("fused AdamW available: %s, using fused: %s", fused_available, use_fused)
        return optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
