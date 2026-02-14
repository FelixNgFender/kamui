import datetime

from pealm import constants


def get_current_datetime() -> str:
    return datetime.datetime.now(tz=datetime.UTC).astimezone().strftime(constants.DATEFMT_STR)
