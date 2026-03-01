# ptloader

Load zip-based PyTorch checkpoints as NumPy arrays, without installing `torch` at runtime.

`ptloader` is designed to feel familiar if you already use `torch.load`, while staying minimal and safe.

## Quickstart

Install:

```bash
uv add ptloader
```

Use:

```python
from ptloader import load

state = load("model.pt")
print(type(state["layer.weight"]))  # numpy.ndarray
```

`load(...)` preserves nested Python structures (`dict`, `list`, `tuple`, `OrderedDict`) and converts tensors to `numpy.ndarray`.

## API (torch-like usage)

```python
from ptloader import load

obj = load(
    f,                    # path or binary file object
    weights_only=True,    # default behavior
)
```

Supported parameters:

- `f`: `str | pathlib.Path | BinaryIO`
- `weights_only`: `None` or `True` supported (`False` raises `CheckpointError`)

Unsupported `torch.load` parameters currently raise `CheckpointError`:

- `map_location`
- `pickle_module`
- `mmap`
- custom pickle load args

## Notes

- Runtime dependency: `numpy`
- Tests use `torch` to generate realistic checkpoint fixtures
- Only zip-based checkpoints are supported (PyTorch default format)

## Development

```bash
uv sync
uv run pytest -q
```
