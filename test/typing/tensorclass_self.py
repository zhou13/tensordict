from typing import assert_type

import torch
from tensordict import TensorClass


class Foo(TensorClass):
    config: torch.Tensor
    value: torch.Tensor


foo = Foo(
    config=torch.zeros(8, 1),
    value=torch.ones(8, 2),
    batch_size=[8],
)

# Batch indexing
assert_type(foo[2], Foo)
assert_type(foo[2:], Foo)
assert_type(foo[...], Foo)
assert_type(foo[torch.tensor([1, 3])], Foo)
assert_type(foo[range(2, 6)], Foo)
assert_type(foo[[1, 3]], Foo)
assert_type(foo[(slice(1, 7),)], Foo)
assert_type(foo[torch.tensor([True, False, True, False, True, False, True, False])], Foo)

# Device and dtype conversion
assert_type(foo.to("cpu"), Foo)
assert_type(foo.cpu(), Foo)
assert_type(foo.cuda(), Foo)

# Copy-like transforms
assert_type(foo.clone(), Foo)
assert_type(foo.detach(), Foo)
assert_type(foo.contiguous(), Foo)

# Shape transforms
assert_type(foo.reshape(2, 4), Foo)
assert_type(foo.view(2, 4), Foo)
assert_type(foo.flatten(), Foo)
assert_type(foo.reshape(2, 4).unflatten(0, torch.Size([1, 2])), Foo)
assert_type(foo.unsqueeze(0), Foo)
assert_type(foo.unsqueeze(0).squeeze(0), Foo)
matrix = foo.reshape(2, 4)
assert_type(matrix.permute(1, 0), Foo)
assert_type(matrix.transpose(0, 1), Foo)
assert_type(matrix.movedim(0, 1), Foo)
assert_type(matrix.moveaxis(0, 1), Foo)

# Named fields remain available through get().
assert_type(foo.get("value"), torch.Tensor)

# ctp-trading regression
accumulated_batch: Foo | None = foo[4:]
batch: Foo = foo.to("cuda")
