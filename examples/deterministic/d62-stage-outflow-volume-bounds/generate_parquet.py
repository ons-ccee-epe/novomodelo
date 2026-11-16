"""
Generate Parquet fixtures for the d62-stage-outflow-volume-bounds test case.

Run with: python3 generate_parquet.py

Produces:
  - constraints/hydro_bounds.parquet

Case: the d60-stage-outflow-volume-cap study with its generic-constraint
bounds on H0's stage outflow volume declared natively through the
`min_outflow_hm3` / `max_outflow_hm3` columns: an equality of 36.0 hm3 at
stage 0, a 10.8 hm3 ceiling at stage 1, a 64.8 hm3 floor at stage 2 and no
bound at stage 3. The absent side of each bound is null, not 0.0.
"""

import os

import pyarrow as pa
import pyarrow.parquet as pq

script_dir = os.path.dirname(os.path.abspath(__file__))


def write_table(table: pa.Table, relative_path: str) -> None:
    path = os.path.join(script_dir, relative_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pq.write_table(table, path, compression="zstd")
    print(f"wrote {len(table)} rows -> {path}")


hydro_bounds_schema = pa.schema(
    [
        pa.field("hydro_id", pa.int32(), nullable=False),
        pa.field("stage_id", pa.int32(), nullable=False),
        pa.field("min_outflow_hm3", pa.float64(), nullable=True),
        pa.field("max_outflow_hm3", pa.float64(), nullable=True),
    ]
)

hydro_bounds_table = pa.table(
    {
        "hydro_id": pa.array([0, 0, 0], type=pa.int32()),
        "stage_id": pa.array([0, 1, 2], type=pa.int32()),
        "min_outflow_hm3": pa.array([36.0, None, 64.8], type=pa.float64()),
        "max_outflow_hm3": pa.array([36.0, 10.8, None], type=pa.float64()),
    },
    schema=hydro_bounds_schema,
)
write_table(hydro_bounds_table, "constraints/hydro_bounds.parquet")

print("done.")
