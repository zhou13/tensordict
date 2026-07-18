from __future__ import annotations

import pytest
import torch

from tensordict import TensorClass


class Foo(TensorClass):
    config: torch.Tensor
    value: torch.Tensor


@pytest.fixture
def foo() -> Foo:
    return Foo(
        config=torch.arange(8).reshape(8, 1),
        value=torch.arange(16).reshape(8, 2),
        batch_size=[8],
    )


def _assert_foo(result: Foo, batch_size: list[int]) -> None:
    assert type(result) is Foo
    assert result.batch_size == torch.Size(batch_size)


def test_batch_indexing_preserves_tensorclass(foo: Foo) -> None:
    cases = (
        (2, [], foo.value[2]),
        (slice(2, None), [6], foo.value[2:]),
        (Ellipsis, [8], foo.value),
        (torch.tensor([1, 3]), [2], foo.value[[1, 3]]),
        (range(2, 6), [4], foo.value[2:6]),
        ([1, 3], [2], foo.value[[1, 3]]),
        ((slice(1, 7),), [6], foo.value[1:7]),
        (
            torch.tensor([True, False, True, False, True, False, True, False]),
            [4],
            foo.value[[0, 2, 4, 6]],
        ),
    )
    for index, batch_size, expected in cases:
        result = foo[index]
        _assert_foo(result, batch_size)
        torch.testing.assert_close(result.value, expected)


def test_conversions_and_copies_preserve_tensorclass(foo: Foo) -> None:
    results = (foo.to("cpu"), foo.cpu(), foo.clone(), foo.detach(), foo.contiguous())
    for result in results:
        _assert_foo(result, [8])
        torch.testing.assert_close(result.value, foo.value)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_preserves_tensorclass(foo: Foo) -> None:
    result = foo.cuda()
    _assert_foo(result, [8])
    assert result.value.is_cuda
    torch.testing.assert_close(result.value.cpu(), foo.value)


def test_shape_transforms_preserve_tensorclass(foo: Foo) -> None:
    reshaped = foo.reshape(2, 4)
    cases = (
        (reshaped, [2, 4], foo.value.reshape(2, 4, 2)),
        (foo.view(2, 4), [2, 4], foo.value.reshape(2, 4, 2)),
        (reshaped.flatten(), [8], foo.value),
        (reshaped.unflatten(1, torch.Size([2, 2])), [2, 2, 2], foo.value.reshape(2, 2, 2, 2)),
        (foo.unsqueeze(0), [1, 8], foo.value.unsqueeze(0)),
        (foo.unsqueeze(0).squeeze(0), [8], foo.value),
        (reshaped.permute(1, 0), [4, 2], foo.value.reshape(2, 4, 2).permute(1, 0, 2)),
        (reshaped.transpose(0, 1), [4, 2], foo.value.reshape(2, 4, 2).transpose(0, 1)),
        (reshaped.movedim(0, 1), [4, 2], foo.value.reshape(2, 4, 2).movedim(0, 1)),
        (reshaped.moveaxis(0, 1), [4, 2], foo.value.reshape(2, 4, 2).moveaxis(0, 1)),
    )
    for result, batch_size, expected in cases:
        _assert_foo(result, batch_size)
        torch.testing.assert_close(result.value, expected)


def test_named_field_access_and_string_index_rejection(foo: Foo) -> None:
    assert foo.get("value") is foo.value
    with pytest.raises(ValueError, match="Invalid indexing arguments"):
        foo["value"]
    with pytest.raises(ValueError, match="Invalid indexing arguments"):
        foo[("value",)]
