# TensorClass subtype-preserving annotations

Runtime TensorClass operations already reconstruct the user's concrete subclass. The corresponding
annotations now preserve that subtype for batch indexing as they already do for device, copy, and
shape transforms.

Before, both `foo[start:]` and `foo.to("cuda")` could be widened by static analysis, preventing the
following `ctp-trading` assignments. They now infer `Foo` directly:

```python
accumulated_batch: Foo | None = foo[4:]
batch: Foo = foo.to("cuda")
```

`TensorClass.__getitem__` describes batch indexing only. Named fields remain available as attributes
and through `foo.get("value")`; string and all-string-tuple indexing retain their existing runtime
rejection behavior.
