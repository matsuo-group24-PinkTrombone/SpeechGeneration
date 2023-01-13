from typing import Sequence, Tuple, Union

import torch
from packaging.version import parse as V
from torch_complex import functional as FC
from torch_complex.tensor import ComplexTensor

is_torch_1_9_plus = V(torch.__version__) >= V("1.9.0")


def is_torch_complex_tensor(c):
    return not isinstance(c, ComplexTensor) and is_torch_1_9_plus and torch.is_complex(c)


def is_complex(c):
    return isinstance(c, ComplexTensor) or is_torch_complex_tensor(c)
