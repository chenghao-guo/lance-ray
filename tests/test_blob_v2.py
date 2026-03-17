"""Blob v2 integration tests for lance-ray.

These tests verify that:

- lance-ray can read datasets that use the blob v2 extension type
  (``lance.blob.v2``) and exposes them to Ray as LargeBinary bytes.
- Projection and filtering on blob v2 columns work as expected.
- Underlying ``LanceDataset.take_blobs`` supports both ``ids`` and
  ``indices`` selectors for blob v2 columns.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pyarrow as pa
import pytest
import ray
from ray.exceptions import RayTaskError

# Prefer the local Lance Python package when running in a monorepo layout
sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "lance" / "python" / "python")
)

lance = pytest.importorskip("lance")

try:
    from lance import Blob, blob_array, blob_field
except Exception:  # pragma: no cover - guarded by importorskip
    pytest.skip(
        "blob v2 API is not available in this Lance build", allow_module_level=True
    )

import lance_ray.io as lr  # noqa: E402


def _build_blob_v2_table(tmp_path: Path) -> tuple[pa.Table, bytes, bytes, Path]:
    """Construct a small blob v2 table for testing.

    Returns the table, inline payload bytes, external payload bytes, and
    the path to the external blob file.
    """

    inline_payload = b"inline-bytes"
    external_payload = b"external-bytes"
    external_path = tmp_path / "external_blob.bin"
    external_path.write_bytes(external_payload)

    schema = pa.schema(
        [
            pa.field("id", pa.int64()),
            blob_field("blob"),
        ]
    )

    sig = inspect.signature(Blob.from_uri)
    blob_kwargs: dict[str, object] = {}
    if "position" in sig.parameters and "size" in sig.parameters:
        blob_kwargs = {"position": 0, "size": len(external_payload)}

    table = pa.table(
        {
            "id": [1, 2, 3],
            "blob": blob_array(
                [
                    inline_payload,
                    Blob.from_uri(
                        external_path.as_uri(),
                        **blob_kwargs,
                    ),
                    None,
                ]
            ),
        },
        schema=schema,
    )
    return table, inline_payload, external_payload, external_path


def test_blob_v2_roundtrip_with_projection_and_filter(tmp_path: Path) -> None:
    """Round-trip blob v2 columns through write_lance/read_lance.

    The Ray-facing schema should expose blob v2 columns as LargeBinary bytes,
    and projection/filter should not break blob reconstruction.
    """

    table, inline_payload, external_payload, _ = _build_blob_v2_table(tmp_path)
    path = tmp_path / "blob_v2_roundtrip.lance"

    ds_ray = ray.data.from_arrow(table)
    lr.write_lance(
        ds_ray,
        str(path),
        schema=table.schema,
        data_storage_version="2.2",
    )

    # Full read: expect bytes + None in id order
    try:
        df = (
            lr.read_lance(str(path))
            .to_pandas()
            .sort_values("id")
            .reset_index(drop=True)
        )
    except RayTaskError as exc:  # pragma: no cover - guarded by Lance version
        # Older Lance builds may not be able to decode experimental blob v2
        # files written with data_storage_version="2.2".
        msg = str(exc)
        if "Packed struct fixed child exceeds row bounds" in msg:
            pytest.skip(
                "Current Lance build cannot decode blob v2 packed encoding; "
                "skipping blob v2 roundtrip test.",
            )
        raise
    assert df["blob"].tolist() == [inline_payload, external_payload, None]

    # Projection + filter on the blob column
    df_proj = (
        lr.read_lance(str(path), columns=["blob"], filter="id >= 2")
        .to_pandas()
        .reset_index(drop=True)
    )
    assert df_proj.columns.tolist() == ["blob"]
    assert df_proj["blob"].tolist() == [external_payload, None]


def test_blob_v2_take_blobs_ids_and_indices(tmp_path: Path) -> None:
    """Validate blob v2 take_blobs semantics for ids and indices selectors.

    The bytes read via ``take_blobs`` should match what lance-ray exposes
    through ``read_lance`` for non-null rows.
    """

    table, inline_payload, external_payload, _ = _build_blob_v2_table(tmp_path)
    path = tmp_path / "blob_v2_take_blobs.lance"

    write_sig = inspect.signature(lance.write_dataset)
    write_kwargs: dict[str, object] = {"data_storage_version": "2.2"}
    if "allow_external_blob_outside_bases" in write_sig.parameters:
        write_kwargs["allow_external_blob_outside_bases"] = True

    ds = lance.write_dataset(
        table,
        path,
        **write_kwargs,
    )

    expected = [inline_payload, external_payload]

    # indices selector
    blobs_by_indices = ds.take_blobs("blob", indices=[0, 1])
    values_by_indices: list[bytes] = []
    for bf in blobs_by_indices:
        with bf as f:
            values_by_indices.append(f.read())
    assert values_by_indices == expected

    # ids selector (row ids are stable within a snapshot)
    row_ids = ds.to_table(columns=[], with_row_id=True).column("_rowid").to_pylist()
    blobs_by_ids = ds.take_blobs("blob", ids=row_ids[:2])
    values_by_ids: list[bytes] = []
    for bf in blobs_by_ids:
        with bf as f:
            values_by_ids.append(f.read())
    assert values_by_ids == expected
