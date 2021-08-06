from numpy import ndarray
from numpy.random import RandomState
from numpy.typing import ArrayLike
from numba.typed import List
from typing import Union

FormattedSequences = Union[ndarray, List[ndarray]]

Sequences = Union[ArrayLike, ndarray, List[ndarray]]

Seed = Union[None, int, RandomState]
