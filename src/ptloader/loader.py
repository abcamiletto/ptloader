from __future__ import annotations

from collections import OrderedDict
from io import BytesIO
from pathlib import Path
import pickle
import zipfile
from typing import Any, BinaryIO, Callable, Mapping

import numpy as np


class CheckpointError(ValueError):
    """Raised when a checkpoint cannot be parsed without torch."""


class _TorchScriptUnknown:
    pass


StorageType = str
StorageResolver = Callable[[StorageType, int], np.dtype[Any] | None]
GlobalResolver = Callable[[str, str], Any | None]


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
    "UInt32Storage": np.dtype(np.uint32),
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


_torchscript_object_types: dict[str, type[Any]] = {}


_UNKNOWN_STORAGE_HANDLER: StorageResolver | None = None
_UNKNOWN_PICKLE_GLOBAL_HANDLER: GlobalResolver | None = None


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



def _restore_type_tag(*args: Any) -> Any:
    return args[0] if args else None



def _build_intlist(*values: Any) -> list[int]:
    if not values:
        return []
    if len(values) == 1:
        return list(values[0])
    return list(values)



def _get_torchscript_object_type(name: str) -> type[Any]:
    cached = _torchscript_object_types.get(name)
    if cached is not None:
        return cached

    class _TorchScriptObject(_TorchScriptUnknown):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.__torch_type__ = name
            self.args = list(args)
            self.__dict__.update(kwargs)

        def __repr__(self) -> str:
            return f"<{name} {self.__dict__}>"

    _TorchScriptObject.__name__ = name
    cls = _TorchScriptObject
    _torchscript_object_types[name] = cls
    return cls


_PICKLE_GLOBALS: dict[tuple[str, str], Any] = {
    ("collections", "OrderedDict"): OrderedDict,
    ("torch._utils", "_rebuild_tensor"): _rebuild_tensor,
    ("torch._utils", "_rebuild_tensor_v2"): _rebuild_tensor_v2,
    ("torch._utils", "_rebuild_parameter"): _rebuild_parameter,
    ("torch.jit._pickle", "restore_type_tag"): _restore_type_tag,
    ("torch.jit._pickle", "build_intlist"): _build_intlist,
}


def register_storage_dtype(storage_type: str, dtype: Any) -> None:
    """Register additional storage->dtype mappings for unknown torch storage formats."""

    _STORAGE_DTYPES[storage_type] = np.dtype(dtype)


def register_pickle_global(module: str, name: str, target: Any) -> None:
    """Register additional pickle globals as known safe load targets."""

    _PICKLE_GLOBALS[(module, name)] = target


def set_unknown_storage_handler(handler: StorageResolver | None) -> None:
    """Set a fallback for unknown torch storage types."""

    global _UNKNOWN_STORAGE_HANDLER
    _UNKNOWN_STORAGE_HANDLER = handler


def set_unknown_pickle_global_handler(handler: GlobalResolver | None) -> None:
    """Set a fallback for unknown pickle globals."""

    global _UNKNOWN_PICKLE_GLOBAL_HANDLER
    _UNKNOWN_PICKLE_GLOBAL_HANDLER = handler


class _TorchCheckpointUnpickler(pickle.Unpickler):
    def __init__(
        self,
        file: bytes,
        archive: zipfile.ZipFile,
        root: str,
        byteorder: str,
        *,
        storage_types: Mapping[str, np.dtype[Any]],
        pickle_globals: Mapping[tuple[str, str], Any],
        storage_type_resolver: StorageResolver | None = None,
        pickle_global_resolver: GlobalResolver | None = None,
        permissive_torchscript: bool = False,
    ) -> None:
        super().__init__(BytesIO(file))
        self._archive = archive
        self._root = root
        self._byteorder = byteorder
        self._storages: dict[str, np.ndarray[Any, Any]] = {}
        self._storage_type_resolver = storage_type_resolver
        self._pickle_global_resolver = pickle_global_resolver
        self._storage_types = dict(storage_types)
        self._pickle_globals = dict(pickle_globals)
        self._permissive_torchscript = permissive_torchscript

    def find_class(self, module: str, name: str) -> Any:
        if module == "torch" and name.endswith("Storage"):
            return name

        if module == "__torch__":
            if not self._permissive_torchscript:
                raise CheckpointError(f"Unsupported pickle global: {module}.{name}")
            return _get_torchscript_object_type(name)

        if module.startswith("torch.jit._pickle"):
            if not self._permissive_torchscript:
                raise CheckpointError(f"Unsupported pickle global: {module}.{name}")

        known = self._pickle_globals.get((module, name))
        if known is not None:
            return known

        if self._pickle_global_resolver is not None:
            resolved = self._pickle_global_resolver(module, name)
            if resolved is not None:
                return resolved

        raise CheckpointError(f"Unsupported pickle global: {module}.{name}")

    def persistent_load(self, pid: Any) -> np.ndarray[Any, Any]:
        if not isinstance(pid, tuple) or len(pid) < 5 or pid[0] != "storage":
            raise CheckpointError(f"Unsupported persistent id: {pid!r}")

        _, storage_type, key, _location, size = pid[:5]
        if not isinstance(storage_type, str):
            raise CheckpointError(f"Unsupported storage type: {storage_type!r}")

        dtype = self._storage_types.get(storage_type)
        if dtype is None and self._storage_type_resolver is not None:
            dtype = self._storage_type_resolver(storage_type, int(size))

        if dtype is None:
            raise CheckpointError(f"Unsupported storage dtype: {storage_type}")

        key_str = str(key)
        cached = self._storages.get(key_str)
        if cached is not None:
            return cached

        data_name = f"{self._root}/data/{key_str}" if self._root else f"data/{key_str}"
        try:
            raw = self._archive.read(data_name)
        except KeyError as exc:
            raise CheckpointError(f"Missing storage blob: {data_name}") from exc

        array = np.frombuffer(raw, dtype=dtype, count=int(size))
        if self._byteorder == "big":
            array = array.byteswap().newbyteorder()

        self._storages[key_str] = array
        return array


def _convert_tensors(obj: Any, convert_attributes: bool = False, *, _seen: set[int] | None = None) -> Any:
    if isinstance(obj, _TensorRef):
        return obj.to_numpy()

    if isinstance(obj, list):
        return [_convert_tensors(item, convert_attributes, _seen=_seen) for item in obj]

    if isinstance(obj, tuple):
        return tuple(_convert_tensors(item, convert_attributes, _seen=_seen) for item in obj)

    if isinstance(obj, dict):
        return obj.__class__((key, _convert_tensors(value, convert_attributes, _seen=_seen)) for key, value in obj.items())

    if convert_attributes and hasattr(obj, "__dict__"):
        seen = _seen if _seen is not None else set()
        if id(obj) in seen:
            return obj

        seen.add(id(obj))
        for key, value in list(vars(obj).items()):
            setattr(obj, key, _convert_tensors(value, convert_attributes, _seen=seen))
        return obj

    return obj


def _find_archive_payload(archive: zipfile.ZipFile, *, payload: str, prefer_data: bool) -> tuple[str, str]:
    if payload not in ("auto", "data", "constants"):
        raise CheckpointError("payload must be one of: 'auto', 'data', 'constants'")

    names = set(archive.namelist())
    if payload == "auto":
        payload_order = ("data.pkl", "constants.pkl") if prefer_data else ("constants.pkl", "data.pkl")
    else:
        payload_order = (f"{payload}.pkl",)

    for payload_name in payload_order:
        if payload_name in names:
            return "", payload_name

    for payload_name in payload_order:
        suffix = f"/{payload_name}"
        for name in names:
            if name.endswith(suffix):
                return name[: -len(suffix)], payload_name

    raise CheckpointError(
        f"Could not locate checkpoint payload ({payload}.pkl)"
        if payload in ("data", "constants")
        else "Could not locate checkpoint payload (data.pkl or constants.pkl)"
    )


def _archive_byteorder(archive: zipfile.ZipFile, root: str) -> str:
    byteorder_name = "byteorder" if root == "" else f"{root}/byteorder"
    if byteorder_name not in archive.namelist():
        return "little"

    return archive.read(byteorder_name).decode("utf-8").strip()


def _load_from_archive(
    archive: zipfile.ZipFile,
    *,
    torchscript_mode: str | None,
    payload: str = "auto",
    storage_registry: Mapping[str, np.dtype[Any]] | None = None,
    pickle_global_registry: Mapping[tuple[str, str], Any] | None = None,
    storage_type_resolver: StorageResolver | None = None,
    pickle_global_resolver: GlobalResolver | None = None,
) -> Any:
    root, payload_name = _find_archive_payload(
        archive,
        payload=payload,
        prefer_data=torchscript_mode == "permissive",
    )
    payload_path = payload_name if root == "" else f"{root}/{payload_name}"
    byteorder = _archive_byteorder(archive, root)

    pickle_globals = dict(_PICKLE_GLOBALS)
    if pickle_global_registry:
        pickle_globals.update(pickle_global_registry)

    storage_types = dict(_STORAGE_DTYPES)
    if storage_registry:
        storage_types.update(storage_registry)

    if storage_type_resolver is None:
        storage_type_resolver = _UNKNOWN_STORAGE_HANDLER
    if pickle_global_resolver is None:
        pickle_global_resolver = _UNKNOWN_PICKLE_GLOBAL_HANDLER

    permissive_torchscript = torchscript_mode == "permissive"
    payload = archive.read(payload_path)
    obj = _TorchCheckpointUnpickler(
        payload,
        archive,
        root,
        byteorder,
        storage_types=storage_types,
        pickle_globals=pickle_globals,
        storage_type_resolver=storage_type_resolver,
        pickle_global_resolver=pickle_global_resolver,
        permissive_torchscript=permissive_torchscript,
    ).load()

    return _convert_tensors(obj, convert_attributes=permissive_torchscript)


def load(
    f: str | Path | BinaryIO,
    map_location: Any | None = None,
    pickle_module: Any | None = None,
    *,
    weights_only: bool | None = None,
    mmap: Any | None = None,
    torchscript_mode: str | None = None,
    payload: str = "auto",
    return_script_object: bool = False,
    storage_registry: Mapping[str, np.dtype[Any]] | None = None,
    pickle_global_registry: Mapping[tuple[str, str], Any] | None = None,
    storage_type_resolver: StorageResolver | None = None,
    pickle_global_resolver: GlobalResolver | None = None,
    **pickle_load_args: Any,
) -> Any:
    """Load a torch checkpoint and return objects where tensors are NumPy arrays."""

    if map_location is not None:
        raise CheckpointError("map_location is not supported by ptloader.load")
    if pickle_module is not None or mmap is not None or pickle_load_args:
        raise CheckpointError(
            "pickle_module, mmap, and custom pickle args are not supported by ptloader.load"
        )
    if weights_only is False:
        raise CheckpointError("ptloader.load only supports weights_only=True behavior")
    if torchscript_mode not in (None, "permissive"):
        raise CheckpointError("torchscript_mode must be one of: None, 'permissive'")
    if payload not in ("auto", "data", "constants"):
        raise CheckpointError("payload must be one of: 'auto', 'data', 'constants'")

    if return_script_object and torchscript_mode is None:
        torchscript_mode = "permissive"

    source: BinaryIO | Path
    source = f if hasattr(f, "read") and hasattr(f, "seek") else Path(f)
    if not zipfile.is_zipfile(source):
        raise ValueError("Only zip-based torch checkpoints are supported")
    with zipfile.ZipFile(source, mode="r") as archive:
        return _load_from_archive(
            archive,
            torchscript_mode=torchscript_mode,
            payload=payload,
            storage_registry=storage_registry,
            pickle_global_registry=pickle_global_registry,
            storage_type_resolver=storage_type_resolver,
            pickle_global_resolver=pickle_global_resolver,
        )


def load_torchscript(
    f: str | Path | BinaryIO,
    *,
    return_script_object: bool = True,
    **kwargs: Any,
) -> Any:
    """Load a TorchScript archive and return reconstructed script objects."""

    kwargs["torchscript_mode"] = "permissive"
    kwargs.setdefault("payload", "auto")
    kwargs["return_script_object"] = return_script_object
    return load(f, **kwargs)
