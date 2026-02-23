import logging

import torch
import torch.utils.data as data_utils
from torch import optim

from pealm import constants, dataset, model, settings, tokenizer, train, utils

logger = logging.getLogger(__name__)


def train_char_bigram(train_settings: settings.TrainCharBigram, model_settings: settings.CharBigram) -> None:
    # aliases
    batch_size = train_settings.batch_size
    context_size = model_settings.context_size
    rank, local_rank, world_size = (
        train_settings.ddp.rank,
        train_settings.ddp.local_rank,
        train_settings.ddp.world_size,
    )
    # each rank processes a subset of the effective batch
    rank_batch_size = batch_size // world_size

    device = utils.compute_init(
        use_accelerator=train_settings.use_accelerator,
        seed=train_settings.seed,
        torch_seed=train_settings.torch_seed,
        fp32_matmul_precision=train_settings.fp32_matmul_precision,
        ddp_enabled=train_settings.ddp.enabled,
        local_rank=local_rank,
        rank=rank,
        world_size=world_size,
    )

    # prepare dataset and model
    with train_settings.input.open("r", encoding="utf-8") as f:
        text = f.read()
    _tokenizer = tokenizer.CharTokenizer.train([text])
    n1 = int(train_settings.train_split * len(text))
    train_dataset = dataset.Text(
        _tokenizer,
        text[:n1],
        context_size,
    )
    val_dataset = dataset.Text(
        _tokenizer,
        text[n1:],
        context_size,
    )

    train_sampler = None
    val_sampler = None
    # DDP sampler only for use with map-style datasets, not iter-style
    if train_settings.ddp.enabled:
        # sampler is mutual exclusive with shuffle in dataloader, so we set shuffle to False
        # and use sampler for shuffling
        train_sampler = data_utils.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = data_utils.DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_dataloader = data_utils.DataLoader(
        train_dataset,
        rank_batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        pin_memory=True,
    )
    val_dataloader = data_utils.DataLoader(
        val_dataset,
        rank_batch_size,
        sampler=val_sampler,
        pin_memory=True,
    )

    _model = model.CharBigram(context_size=context_size, vocab_size=_tokenizer.vocab_size)
    optimizer = optim.AdamW(_model.parameters(), lr=train_settings.lr)

    # move and then compile so compiler don't have to reason about device-specific copies
    _model.to(device)
    # https://docs.pytorch.org/docs/main/notes/ddp#example
    # DDP works with TorchDynamo. When used with TorchDynamo, apply the DDP model wrapper before compiling the model,
    # such that torchdynamo can apply DDPOptimizer (graph-break optimizations) based on DDP bucket sizes
    if not train_settings.ddp.enabled:
        ddp_model = None
        _model.compile()
    else:
        # ddp_local_rank is the gpu id that the model lives on
        ddp_model = torch.nn.parallel.DistributedDataParallel(
            # gradient_as_bucket_view=True to reduce GPU memory fragmentation
            # and slightly improve performance
            _model,
            device_ids=[train_settings.ddp.local_rank],
            gradient_as_bucket_view=True,
        )
        ddp_model.compile()

    # create training context
    run_checkpoint_dir = train_settings.ckpt_dir / _model.TYPE / utils.current_dt()
    if train_settings.ddp.is_master_process:
        run_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        _tokenizer.save(run_checkpoint_dir / constants.TOKENIZER_DIR)

    ctx = train.Context(
        device=device,
        use_mixed_precision=train_settings.use_mixed_precision,
        is_ddp=train_settings.ddp.enabled,
        ddp_model=ddp_model,
        rank=rank,
        world_size=world_size,
        is_master_process=train_settings.ddp.is_master_process,
        model=_model,
        tokenizer=_tokenizer,
        optimizer=optimizer,
        global_batch_size=batch_size,
        train_loader=train_dataloader,
        train_iter=iter(train_dataloader),
        val_loader=val_dataloader,
        step=1,
        max_steps=train_settings.steps,
        train_loss=[],
        val_loss=[],
        save_every=train_settings.save_every,
        checkpoint_dir=run_checkpoint_dir,
        resume_from_checkpoint=train_settings.resume_from_ckpt,
        tokens_to_save=train_settings.tokens_to_save,
        val_every=train_settings.val_every,
    )

    # kick off train loop
    ctx.train_loop()
