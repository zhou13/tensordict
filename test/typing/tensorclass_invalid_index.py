import torch
from tensordict import TensorClass


class Foo(TensorClass):
    value: torch.Tensor


foo = Foo(value=torch.ones(2), batch_size=[2])
foo["value"]
foo[("value",)]
