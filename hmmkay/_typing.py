from typing import Union

from numba.typed import List
from numpy import ndarray
from numpy.random import RandomState
from numpy.typing import ArrayLike

FormattedSequences = Union[ndarray, List[ndarray]]

Sequences = Union[ArrayLike, ndarray, List[ndarray]]

Seed = Union[None, int, RandomState]
