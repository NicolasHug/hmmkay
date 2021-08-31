from typing import Sequence, Union

from numba.typed import List
from numpy import ndarray
from numpy.random import RandomState

FormattedSequences = Union[ndarray, List[ndarray]]

Sequences = Union[Sequence[Sequence], ndarray, List[ndarray]]

Seed = Union[None, int, RandomState]
