from enum import Enum


class Subset(Enum):
    DEVELOPMENT = 0
    TEST = 1

    def folder_name(self):
        if self == Subset.DEVELOPMENT:
            return "development"
        else:
            return "test"
