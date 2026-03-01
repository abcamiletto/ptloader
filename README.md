# ptloader

Load zip-based PyTorch checkpoints as NumPy arrays, without installing `torch` at runtime.

## Quickstart

```bash
uv add ptloader
```

```python
import ptloader

state = ptloader.load("model.pt")
print(type(state["layer.weight"]))  # numpy.ndarray
```

`ptloader.load(...)` supports `str | pathlib.Path | BinaryIO` and converts tensors to `numpy.ndarray` while preserving nested container structure.
