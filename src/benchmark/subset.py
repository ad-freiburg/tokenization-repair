from enum import Enum


class Subset(Enum):
    TUNING = 0
    DEVELOPMENT = 1
    TEST = 2

    def folder_name(self):
        if self == Subset.TUNING:
            return "tuning"
        elif self == Subset.DEVELOPMENT:
            return "development"
        else:
            return "test"
