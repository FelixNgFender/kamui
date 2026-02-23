"""
New and upgraded chat mode because a lot of the code has changed since the last one.

Intended to be run single GPU only atm
"""

import torch

from pealm import checkpoint, model, settings, utils
from pealm.chat import engine, state_machine
from pealm.tokenizer import PeashooterTokenizer


class ChatCliStateMachine(state_machine.Chat):
    def start(
        self,
        max_tokens: int,
        temperature: float,
        top_k: int | None,
        prompt: str | None,
    ) -> None:
        # ruff: disable[T201]
        print("\nPeashooter Interactive Mode")
        print("-" * 50)
        print("Type 'quit' or 'exit' to end the conversation")
        print("Type 'clear' to start a new conversation")
        print("-" * 50)

        while True:
            if prompt:
                # get the prompt from the launch command
                user_input = prompt
            else:
                # get the prompt interactively from the console
                try:
                    user_input = input("\nUser: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nGoodbye!")
                    break

            # handle special commands
            match user_input.lower():
                case "":
                    continue
                case "quit" | "exit":
                    print("Goodbye!")
                    break
                case "clear":
                    self.reset()
                    print("Conversation cleared.")
                    continue

            # add user message to the conversation
            self.add_user_message(user_input)

            # kick off the assistant
            self.conversation_tokens.append(self.assistant_start)
            print("\nAssistant: ", end="", flush=True)
            with self.autocast_ctx:
                for token_column, _ in self.engine.generate(
                    self.conversation_tokens,
                    num_samples=1,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k,
                ):
                    token = token_column[0]  # pop the batch dimension (num_samples=1)
                    self.conversation_tokens.append(token)
                    token_text = self.tokenizer.decode([token])
                    print(token_text, end="", flush=True)
            print()
            # we have to ensure that the assistant end token is the last token
            # so even if generation ends due to max tokens, we have to append it to the end
            if self.conversation_tokens[-1] != self.assistant_end:
                self.conversation_tokens.append(self.assistant_end)

            # in the prompt mode, we only want a single response and exit
            if prompt:
                break
        # ruff: enable[T201]


# TODO: replace with actual peashooter model
def chat_cli(chat_settings: settings.ChatCli, model_settings: settings.GPT2) -> None:
    # init the model and tokenizer
    device = utils.compute_init(
        use_accelerator=chat_settings.use_accelerator,
        seed=chat_settings.seed,
        torch_seed=chat_settings.torch_seed,
        fp32_matmul_precision=chat_settings.fp32_matmul_precision,
    )
    tokenizer = PeashooterTokenizer.load(chat_settings.tokenizer_dir)
    _model = model.GPT2(
        context_size=model_settings.context_size,
        # don't use tokenizer.vocab_size for GPT2 cuz we want 50304 for cuda niceness
        vocab_size=model_settings.vocab_size,
        embedding_size=model_settings.embedding_size,
        num_layers=model_settings.num_layers,
        num_heads=model_settings.num_heads,
    ).to(device)

    # load checkpoint
    model_state_dict = checkpoint.Checkpoint.load_weights(chat_settings.ckpt, map_location=device)
    _model.load_state_dict(model_state_dict)

    # only compile after graph is in place and on device
    _model.eval().compile()

    # create engine for efficient generation
    _engine = engine.Engine(_model, tokenizer)

    # create the chat state machine to manage the conversation state
    autocast_ctx = torch.amp.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=chat_settings.use_mixed_precision
    )
    chat_sm = ChatCliStateMachine(_engine, autocast_ctx)

    # start chat sm
    chat_sm.start(chat_settings.max_tokens, chat_settings.temperature, chat_settings.top_k, chat_settings.prompt)
