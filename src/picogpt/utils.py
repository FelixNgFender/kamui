import datetime

from picogpt import constants


def get_current_datetime() -> str:
    return datetime.datetime.now(tz=datetime.UTC).astimezone().strftime(constants.DATEFMT_STR)
