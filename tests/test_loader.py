from __future__ import annotations

from collections import OrderedDict
from fractions import Fraction
from pathlib import Path
import zipfile

import numpy as np
import pytest
import torch

from ptloader import load
from ptloader.loader import CheckpointError


def _save_checkpoint(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


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
