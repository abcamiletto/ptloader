from __future__ import annotations

from collections import OrderedDict
from fractions import Fraction
from io import BytesIO
from pathlib import Path
import pickle
import sys
import types
from typing import Any
import zipfile

import numpy as np
import pytest
import torch

from ptloader import load, load_torchscript
from ptloader.loader import CheckpointError


def _save_checkpoint(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


class _StorageRef:
    def __init__(self, storage_type: str, key: str, size: int) -> None:
        self.storage_type = storage_type
        self.key = key
        self.size = size


def _contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    if not shape:
        return ()
    stride = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        stride[i] = stride[i + 1] * shape[i + 1]
    return tuple(stride)


class _TensorPayload:
    def __init__(
        self,
        storage: _StorageRef,
        shape: tuple[int, ...],
        stride: tuple[int, ...] | None = None,
        storage_offset: int = 0,
    ) -> None:
        self.storage = storage
        self.shape = shape
        self.stride = stride if stride is not None else _contiguous_stride(shape)
        self.storage_offset = storage_offset

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        return (
            torch._utils._rebuild_tensor,
            (self.storage, self.storage_offset, self.shape, self.stride),
        )


class _PersistentPickler(pickle.Pickler):
    def persistent_id(self, obj: Any) -> Any:
        if isinstance(obj, _StorageRef):
            return ("storage", obj.storage_type, obj.key, "cpu", obj.size)
        return None


def _dumps_payload(payload: Any) -> bytes:
    buffer = BytesIO()
    _PersistentPickler(buffer, protocol=4).dump(payload)
    return buffer.getvalue()


def _write_synthetic_archive(
    path: Path,
    payload: Any,
    storage_blobs: dict[str, bytes],
    *,
    payload_name: str = "data.pkl",
    root: str = "archive",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload_entry = f"{root}/{payload_name}" if root else payload_name
    byteorder_entry = f"{root}/byteorder" if root else "byteorder"
    data_prefix = f"{root}/data" if root else "data"
    payload_blob = _dumps_payload(payload)

    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(payload_entry, payload_blob)
        archive.writestr(byteorder_entry, "little")
        for key, blob in storage_blobs.items():
            archive.writestr(f"{data_prefix}/{key}", blob)


def _make_torchscript_module_like(payload: Any) -> Any:
    module = sys.modules.get("__torch__")
    if module is None:
        module = types.ModuleType("__torch__")
        sys.modules["__torch__"] = module

    cls = getattr(module, "ModuleLike", None)
    if cls is None:
        cls = type("ModuleLike", (), {})
        cls.__module__ = "__torch__"
        module.ModuleLike = cls

    obj = cls()
    obj.tensor = payload
    obj.inner = {"tensor": payload}
    return obj


def _custom_build_box(value: Any) -> dict[str, Any]:
    return {"wrapped": value}


def _float_storage_payload(values: np.ndarray[Any, Any]) -> _TensorPayload:
    return _TensorPayload(
        _StorageRef("FloatStorage", "0", values.size),
        shape=(values.size,),
    )


def test_load_tensor_as_numpy_array(tmp_path: Path) -> None:
    checkpoint = tmp_path / "single_tensor.pt"
    _save_checkpoint(checkpoint, {"weights": torch.arange(12, dtype=torch.float32).reshape(3, 4)})

    loaded = load(checkpoint)

    assert isinstance(loaded, dict)
    assert isinstance(loaded["weights"], np.ndarray)
    np.testing.assert_array_equal(
        loaded["weights"],
        np.arange(12, dtype=np.float32).reshape(3, 4),
    )


def test_load_nested_structure(tmp_path: Path) -> None:
    checkpoint = tmp_path / "nested.pt"
    payload = {
        "layer": {
            "bias": torch.tensor([1, 2, 3], dtype=torch.int64),
            "weights": torch.tensor([[0.25, 0.5], [0.75, 1.0]], dtype=torch.float64),
        },
        "epoch": 3,
        "names": ["a", "b"],
    }
    _save_checkpoint(checkpoint, payload)

    loaded = load(checkpoint)

    assert isinstance(loaded["layer"]["bias"], np.ndarray)
    np.testing.assert_array_equal(loaded["layer"]["bias"], np.array([1, 2, 3], dtype=np.int64))
    np.testing.assert_array_equal(
        loaded["layer"]["weights"],
        np.array([[0.25, 0.5], [0.75, 1.0]], dtype=np.float64),
    )
    assert loaded["epoch"] == 3
    assert loaded["names"] == ["a", "b"]


def test_load_reuses_storage_views(tmp_path: Path) -> None:
    checkpoint = tmp_path / "views.pt"

    base = torch.arange(8, dtype=torch.float32)
    payload = {
        "base": base,
        "view": base[2:6],
    }
    _save_checkpoint(checkpoint, payload)

    loaded = load(checkpoint)

    np.testing.assert_array_equal(loaded["base"], np.arange(8, dtype=np.float32))
    np.testing.assert_array_equal(loaded["view"], np.arange(2, 6, dtype=np.float32))


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (torch.float16, np.float16),
        (torch.float32, np.float32),
        (torch.float64, np.float64),
        (torch.int32, np.int32),
        (torch.int64, np.int64),
        (torch.uint8, np.uint8),
        (torch.bool, np.bool_),
    ],
)
def test_supported_dtypes(tmp_path: Path, dtype: torch.dtype, expected: np.dtype) -> None:
    checkpoint = tmp_path / f"dtype_{dtype}.pt"
    tensor = torch.tensor([0, 1, 2, 3], dtype=dtype)
    _save_checkpoint(checkpoint, {"x": tensor})

    loaded = load(checkpoint)

    assert loaded["x"].dtype == expected


def test_rejects_legacy_non_zip_checkpoints(tmp_path: Path) -> None:
    checkpoint = tmp_path / "legacy.pt"
    torch.save({"x": torch.tensor([1, 2, 3])}, checkpoint, _use_new_zipfile_serialization=False)

    with pytest.raises(ValueError, match="zip"):
        load(checkpoint)


def test_accepts_file_object(tmp_path: Path) -> None:
    checkpoint = tmp_path / "fileobj.pt"
    _save_checkpoint(checkpoint, {"x": torch.tensor([1, 2, 3], dtype=torch.int64)})

    with checkpoint.open("rb") as f:
        loaded = load(f)

    np.testing.assert_array_equal(loaded["x"], np.array([1, 2, 3], dtype=np.int64))


def test_rejects_map_location(tmp_path: Path) -> None:
    checkpoint = tmp_path / "map_location.pt"
    _save_checkpoint(checkpoint, {"x": torch.tensor([4, 5], dtype=torch.int64)})

    with pytest.raises(CheckpointError, match="map_location"):
        load(checkpoint, map_location="cpu")


def test_rejects_weights_only_false(tmp_path: Path) -> None:
    checkpoint = tmp_path / "weights_only.pt"
    _save_checkpoint(checkpoint, {"x": torch.tensor([1], dtype=torch.int64)})

    with pytest.raises(CheckpointError, match="weights_only"):
        load(checkpoint, weights_only=False)


def test_load_scalar_tensor(tmp_path: Path) -> None:
    checkpoint = tmp_path / "scalar.pt"
    _save_checkpoint(checkpoint, {"value": torch.tensor(7.5, dtype=torch.float32)})

    loaded = load(checkpoint)

    assert isinstance(loaded["value"], np.ndarray)
    assert loaded["value"].shape == ()
    assert loaded["value"].item() == pytest.approx(7.5)


def test_load_non_contiguous_transposed_tensor(tmp_path: Path) -> None:
    checkpoint = tmp_path / "transposed.pt"
    tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
    _save_checkpoint(checkpoint, {"x": tensor})

    loaded = load(checkpoint)

    assert loaded["x"].shape == (4, 3)
    np.testing.assert_array_equal(loaded["x"], np.arange(12, dtype=np.float32).reshape(3, 4).T)


def test_load_parameter_object(tmp_path: Path) -> None:
    checkpoint = tmp_path / "parameter.pt"
    param = torch.nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32))
    _save_checkpoint(checkpoint, {"p": param})

    loaded = load(checkpoint)

    assert isinstance(loaded["p"], np.ndarray)
    np.testing.assert_array_equal(loaded["p"], np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))


def test_preserves_ordered_dict_type(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ordered.pt"
    payload = OrderedDict(
        [
            ("a", torch.tensor([1], dtype=torch.int64)),
            ("b", torch.tensor([2], dtype=torch.int64)),
        ]
    )
    _save_checkpoint(checkpoint, payload)

    loaded = load(checkpoint)

    assert isinstance(loaded, OrderedDict)
    assert list(loaded.keys()) == ["a", "b"]
    np.testing.assert_array_equal(loaded["a"], np.array([1], dtype=np.int64))
    np.testing.assert_array_equal(loaded["b"], np.array([2], dtype=np.int64))


def test_rejects_unsupported_pickled_globals(tmp_path: Path) -> None:
    checkpoint = tmp_path / "unsupported_global.pt"
    _save_checkpoint(checkpoint, {"obj": Fraction(2, 3)})

    with pytest.raises(CheckpointError, match="Unsupported pickle global"):
        load(checkpoint)


def test_rejects_missing_data_pickle_entry(tmp_path: Path) -> None:
    checkpoint = tmp_path / "missing_data.pkl.pt"
    _save_checkpoint(checkpoint, {"x": torch.tensor([1, 2, 3], dtype=torch.float32)})

    with zipfile.ZipFile(checkpoint, "r") as src:
        names = src.namelist()
        entries = {name: src.read(name) for name in names if not name.endswith("/data.pkl")}

    with zipfile.ZipFile(checkpoint, "w") as dst:
        for name, data in entries.items():
            dst.writestr(name, data)

    with pytest.raises(CheckpointError, match="data.pkl"):
        load(checkpoint)


def test_supports_uint32_storage_in_synthetic_archive(tmp_path: Path) -> None:
    checkpoint = tmp_path / "uint32_storage.pt"
    values = np.array([1, 2, 3, 2**31], dtype=np.uint32)
    payload = {
        "x": _TensorPayload(
            _StorageRef("UInt32Storage", "0", values.size),
            shape=(values.size,),
        )
    }
    _write_synthetic_archive(checkpoint, payload, {"0": values.tobytes()})

    loaded = load(checkpoint)

    assert loaded["x"].dtype == np.uint32
    np.testing.assert_array_equal(loaded["x"], values)


def test_loads_torchscript_constants_archive(tmp_path: Path) -> None:
    checkpoint = tmp_path / "torchscript_constants.pt"
    values = np.array([0.25, 0.5, 0.75], dtype=np.float32)
    payload = {"const": _float_storage_payload(values)}
    _write_synthetic_archive(
        checkpoint,
        payload,
        {"0": values.tobytes()},
        payload_name="constants.pkl",
        root="",
    )

    loaded = load(checkpoint)

    np.testing.assert_array_equal(loaded["const"], values)


def test_torchscript_permissive_converts_object_attributes(tmp_path: Path) -> None:
    checkpoint = tmp_path / "torchscript_permissive.pt"
    values = np.array([7.0, 8.0], dtype=np.float32)
    tensor_payload = _float_storage_payload(values)
    payload = _make_torchscript_module_like(tensor_payload)
    _write_synthetic_archive(
        checkpoint,
        payload,
        {"0": values.tobytes()},
        payload_name="constants.pkl",
        root="",
    )

    with pytest.raises(CheckpointError, match="Unsupported pickle global: __torch__"):
        load(checkpoint)

    loaded = load(checkpoint, torchscript_mode="permissive", return_script_object=True)

    assert isinstance(loaded.tensor, np.ndarray)
    np.testing.assert_array_equal(loaded.tensor, values)
    assert isinstance(loaded.inner["tensor"], np.ndarray)
    np.testing.assert_array_equal(loaded.inner["tensor"], values)


def test_load_torchscript_public_api(tmp_path: Path) -> None:
    checkpoint = tmp_path / "load_torchscript_api.pt"
    values = np.array([11.0, 12.0], dtype=np.float32)
    payload = {"x": _float_storage_payload(values)}
    _write_synthetic_archive(
        checkpoint,
        payload,
        {"0": values.tobytes()},
        payload_name="constants.pkl",
        root="",
    )

    loaded = load_torchscript(checkpoint)

    np.testing.assert_array_equal(loaded["x"], values)


def test_unknown_storage_callback_can_decode_custom_storage(tmp_path: Path) -> None:
    checkpoint = tmp_path / "unknown_storage_callback.pt"
    values = np.array([1.5, 2.5], dtype=np.float32)
    payload = {
        "x": _TensorPayload(
            _StorageRef("MysteryStorage", "0", values.size),
            shape=(values.size,),
        )
    }
    _write_synthetic_archive(checkpoint, payload, {"0": values.tobytes()})

    with pytest.raises(CheckpointError, match="Unsupported storage dtype"):
        load(checkpoint)

    loaded = load(
        checkpoint,
        storage_type_resolver=lambda storage_type, _size: (
            np.dtype(np.float32) if storage_type == "MysteryStorage" else None
        ),
    )

    np.testing.assert_array_equal(loaded["x"], values)


def test_unknown_pickle_global_callback_can_decode_custom_global(tmp_path: Path) -> None:
    checkpoint = tmp_path / "unknown_global_callback.pt"
    values = np.array([3.0, 4.0], dtype=np.float32)
    tensor_payload = _float_storage_payload(values)

    class _CustomGlobalPayload:
        def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
            return (_custom_build_box, (tensor_payload,))

    _write_synthetic_archive(checkpoint, _CustomGlobalPayload(), {"0": values.tobytes()})

    with pytest.raises(CheckpointError, match="Unsupported pickle global"):
        load(checkpoint)

    def _resolve_custom_global(module: str, name: str) -> Any | None:
        if module == _custom_build_box.__module__ and name == _custom_build_box.__name__:
            return _custom_build_box
        return None

    loaded = load(checkpoint, pickle_global_resolver=_resolve_custom_global)

    assert isinstance(loaded["wrapped"], np.ndarray)
    np.testing.assert_array_equal(loaded["wrapped"], values)
