from typing_extensions import Literal, TypeAlias

SplitStrategy: TypeAlias = Literal[
    "random", "stratified", "manual", "other", "none", "unknown"
]
