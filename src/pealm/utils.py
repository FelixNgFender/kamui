import datetime

from pealm import constants


def current_dt() -> str:
    return datetime.datetime.now(tz=datetime.UTC).astimezone().strftime(constants.DATEFMT_STR)


def parse_dt(dt_str: str) -> datetime.datetime:
    return datetime.datetime.strptime(dt_str, constants.DATEFMT_STR).astimezone()


def current_dt_human() -> str:
    """Human-readable datetime string for logging and reporting."""
    return datetime.datetime.now(tz=datetime.UTC).astimezone().strftime(constants.DATEFMT_STR_HUMAN)


def parse_dt_human(dt_str: str) -> datetime.datetime:
    return datetime.datetime.strptime(dt_str, constants.DATEFMT_STR_HUMAN).astimezone()
