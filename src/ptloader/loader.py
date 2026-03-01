from __future__ import annotations

from collections import OrderedDict
from io import BytesIO
from pathlib import Path
import pickle
import zipfile
from typing import Any

import numpy as np


class CheckpointError(ValueError):
    """Raised when a checkpoint cannot be parsed without torch."""


_STORAGE_DTYPES: dict[str, np.dtype[Any]] = {
    "DoubleStorage": np.dtype(np.float64),
    "FloatStorage": np.dtype(np.float32),
    "HalfStorage": np.dtype(np.float16),
    "LongStorage": np.dtype(np.int64),
    "IntStorage": np.dtype(np.int32),
    "ShortStorage": np.dtype(np.int16),
    "CharStorage": np.dtype(np.int8),
    "ByteStorage": np.dtype(np.uint8),
    "BoolStorage": np.dtype(np.bool_),
    "ComplexFloatStorage": np.dtype(np.complex64),
    "ComplexDoubleStorage": np.dtype(np.complex128),
}

try:
    _STORAGE_DTYPES["BFloat16Storage"] = np.dtype("bfloat16")
except TypeError:
    # Some NumPy builds do not expose bfloat16.
    pass


class _TensorRef:
    __slots__ = ("storage", "offset", "shape", "stride")

    def __init__(
        self,
        storage: np.ndarray[Any, Any],
        offset: int,
        shape: tuple[int, ...],
        stride: tuple[int, ...],
    ) -> None:
        self.storage = storage
        self.offset = offset
        self.shape = shape
        self.stride = stride

    def to_numpy(self) -> np.ndarray[Any, Any]:
        itemsize = self.storage.dtype.itemsize
        return np.ndarray(
            shape=self.shape,
            dtype=self.storage.dtype,
            buffer=self.storage,
            offset=self.offset * itemsize,
            strides=tuple(step * itemsize for step in self.stride),
        )


class _TorchCheckpointUnpickler(pickle.Unpickler):
    def __init__(self, file: bytes, archive: zipfile.ZipFile, root: str, byteorder: str) -> None:
        super().__init__(BytesIO(file))
        self._archive = archive
        self._root = root
        self._byteorder = byteorder
        self._storages: dict[str, np.ndarray[Any, Any]] = {}

    def find_class(self, module: str, name: str) -> Any:
        if module == "torch" and name.endswith("Storage"):
            return name

        allowed = {
            ("collections", "OrderedDict"): OrderedDict,
            ("torch._utils", "_rebuild_tensor"): _rebuild_tensor,
            ("torch._utils", "_rebuild_tensor_v2"): _rebuild_tensor_v2,
            ("torch._utils", "_rebuild_parameter"): _rebuild_parameter,
        }
        try:
            return allowed[(module, name)]
        except KeyError as exc:
            raise CheckpointError(f"Unsupported pickle global: {module}.{name}") from exc

    def persistent_load(self, pid: Any) -> np.ndarray[Any, Any]:
        if not isinstance(pid, tuple) or len(pid) < 5 or pid[0] != "storage":
            raise CheckpointError(f"Unsupported persistent id: {pid!r}")

        _, storage_type, key, _location, size = pid[:5]
        if not isinstance(storage_type, str):
            raise CheckpointError(f"Unsupported storage type: {storage_type!r}")

        dtype = _STORAGE_DTYPES.get(storage_type)
        if dtype is None:
            raise CheckpointError(f"Unsupported storage dtype: {storage_type}")

        key_str = str(key)
        cached = self._storages.get(key_str)
        if cached is not None:
            return cached

        try:
            raw = self._archive.read(f"{self._root}/data/{key_str}")
        except KeyError as exc:
            raise CheckpointError(f"Missing storage blob: {self._root}/data/{key_str}") from exc

        array = np.frombuffer(raw, dtype=dtype, count=int(size))
        if self._byteorder == "big":
            array = array.byteswap().newbyteorder()

        self._storages[key_str] = array
        return array


def _rebuild_tensor(storage: np.ndarray[Any, Any], storage_offset: int, size: Any, stride: Any) -> _TensorRef:
    return _TensorRef(
        storage=storage,
        offset=int(storage_offset),
        shape=tuple(int(v) for v in size),
        stride=tuple(int(v) for v in stride),
    )


def _rebuild_tensor_v2(
    storage: np.ndarray[Any, Any],
    storage_offset: int,
    size: Any,
    stride: Any,
    _requires_grad: Any,
    _backward_hooks: Any,
) -> _TensorRef:
    return _rebuild_tensor(storage, storage_offset, size, stride)


def _rebuild_parameter(data: Any, _requires_grad: Any, _backward_hooks: Any) -> Any:
    return data


def _convert_tensors(obj: Any) -> Any:
    if isinstance(obj, _TensorRef):
        return obj.to_numpy()
    if isinstance(obj, list):
        return [_convert_tensors(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(_convert_tensors(item) for item in obj)
    if isinstance(obj, dict):
        return obj.__class__((k, _convert_tensors(v)) for k, v in obj.items())
    return obj


def _archive_root(archive: zipfile.ZipFile) -> str:
    for name in archive.namelist():
        if name.endswith("/data.pkl"):
            return name[: -len("/data.pkl")]
    raise CheckpointError("Could not locate data.pkl in checkpoint archive")


def load(path: str | Path) -> Any:
    """Load a torch zip checkpoint into Python objects with NumPy tensors."""

    checkpoint_path = Path(path)
    if not zipfile.is_zipfile(checkpoint_path):
        raise ValueError("Only zip-based torch checkpoints are supported")

    with zipfile.ZipFile(checkpoint_path, mode="r") as archive:
        root = _archive_root(archive)
        names = set(archive.namelist())

        byteorder = "little"
        byteorder_name = f"{root}/byteorder"
        if byteorder_name in names:
            byteorder = archive.read(byteorder_name).decode("utf-8").strip()

        data_name = f"{root}/data.pkl"
        if data_name not in names:
            raise CheckpointError("Could not locate data.pkl in checkpoint archive")

        payload = archive.read(data_name)
        obj = _TorchCheckpointUnpickler(payload, archive, root, byteorder).load()
        return _convert_tensors(obj)
