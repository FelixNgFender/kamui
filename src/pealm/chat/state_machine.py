import torch

from pealm import constants
from pealm.chat import engine


class Chat:
    """
    A state machine to manage the conversation flow between the user and the assistant.
    """

    def __init__(
        self,
        engine: engine.Engine,
        autocast_ctx: torch.amp.autocast,
    ) -> None:
        self.engine = engine
        self.autocast_ctx = autocast_ctx
        self.tokenizer = self.engine.tokenizer
        self.bos: int = self.tokenizer.bos_token_id
        self.user_start, self.user_end = (
            self.tokenizer.encode_special(constants.PeashooterSpecialTokens.USER_START),
            self.tokenizer.encode_special(constants.PeashooterSpecialTokens.USER_END),
        )
        self.assistant_start, self.assistant_end = (
            self.tokenizer.encode_special(constants.PeashooterSpecialTokens.ASSISTANT_START),
            self.tokenizer.encode_special(constants.PeashooterSpecialTokens.ASSISTANT_END),
        )
        self.conversation_tokens: list[int] = []
        self.reset()

    def reset(self) -> None:
        self.conversation_tokens = [self.bos]

    def add_user_message(self, message: str) -> None:
        self.conversation_tokens.append(self.user_start)
        self.conversation_tokens.extend(self.tokenizer.encode(message))
        self.conversation_tokens.append(self.user_end)

    def add_assistant_message(self, message: str) -> None:
        self.conversation_tokens.append(self.assistant_start)
        self.conversation_tokens.extend(self.tokenizer.encode(message))
        self.conversation_tokens.append(self.assistant_end)
