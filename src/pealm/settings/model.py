from typing import Annotated, Literal

import pydantic
import pydantic_settings as ps

from pealm import constants
from pealm.settings import base


class ModelBase(base.Base):
    """Settings common to all model type creation."""

    context_size: Annotated[
        pydantic.PositiveInt,
        pydantic.Field(description="Context size for the model"),
    ] = constants.CONTEXT_SIZE


class CharBigram(ModelBase):
    """Settings for creating a character-level bigram model."""


class CharTransformer(ModelBase):
    """Settings for creating a character-level transformer model."""

    embedding_size: Annotated[
        pydantic.PositiveInt,
        pydantic.Field(description="Embedding size for the model"),
    ] = constants.TRANSFORMER_EMBEDDING_SIZE
    num_blocks: Annotated[
        pydantic.PositiveInt,
        pydantic.Field(description="Number of transformer blocks"),
    ] = constants.TRANSFORMER_NUM_BLOCKS
    num_heads: Annotated[
        pydantic.PositiveInt,
        pydantic.Field(description="Number of transformer heads per layer"),
    ] = constants.TRANSFORMER_NUM_HEADS
    dropout: Annotated[
        pydantic.NonNegativeFloat,
        pydantic.Field(le=constants.MAX_DROPOUT, description="Transformer dropout rate"),
    ] = constants.TRANSFORMER_DROPOUT

    model_config = ps.SettingsConfigDict(env_file=".env", extra="ignore")


GPT2PretrainedVariant = Literal["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]


class GPT2Pretrained:
    """Settings for using a pretrained GPT-2 model from Hugging Face. No model creation settings since we just download
    the pretrained model."""

    variant: Annotated[GPT2PretrainedVariant, pydantic.Field(description="GPT2 model variant to use")] = "gpt2"

    model_config = ps.SettingsConfigDict(env_file=".env", extra="ignore")


class GPT2(ModelBase):
    """Settings for creating the GPT-2 (124M) model from OpenAI."""

    # override ModelBase
    context_size: Annotated[
        pydantic.PositiveInt,
        pydantic.Field(description="Context size for the model"),
    ] = constants.GPT2_CONTEXT_SIZE
    vocab_size: Annotated[
        pydantic.PositiveInt,
        pydantic.Field(description="Vocabulary size for the model"),
    ] = constants.GPT2_VOCAB_SIZE
    embedding_size: Annotated[
        pydantic.PositiveInt,
        pydantic.Field(description="Embedding size for the model"),
    ] = constants.GPT2_EMBEDDING_SIZE
    num_layers: Annotated[
        pydantic.PositiveInt,
        pydantic.Field(description="Number of transformer layers"),
    ] = constants.GPT2_NUM_LAYERS
    num_heads: Annotated[
        pydantic.PositiveInt,
        pydantic.Field(description="Number of transformer heads per layer"),
    ] = constants.GPT2_NUM_HEADS

    model_config = ps.SettingsConfigDict(env_file=".env", extra="ignore")


class PeashooterTokenizer(ps.BaseSettings):
    """Settings for creating the Peashooter tokenizer."""

    vocab_size: Annotated[
        pydantic.PositiveInt,
        pydantic.Field(description="Vocabulary size"),
    ] = constants.PS_VOCAB_SIZE

    model_config = ps.SettingsConfigDict(env_file=".env", extra="ignore")
