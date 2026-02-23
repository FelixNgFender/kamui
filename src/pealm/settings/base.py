"""Level 1 base settings all settings models inherit from, i.e., pydantic global settings"""

import pydantic_settings as ps


class Base(ps.BaseSettings):
    model_config = ps.SettingsConfigDict(env_file=".env", extra="ignore")
