from typing import Any, TypeVar, Union

import torch

T = TypeVar("T")
_t_or_any = Union[T, Any]
_tensor_or_any = _t_or_any[torch.Tensor]
