"""
Generate Parquet scenario files for the d40-filling-cascade test case.

Run with: uv tool run --with pyarrow -- python3 generate_parquet.py

Produces:
  - inflow_seasonal_stats.parquet
  - load_seasonal_stats.parquet

Determinism: std_m3s = 0.0 and std_mw = 0.0 on every row, so each realized
inflow/load equals its mean exactly (no per-opening noise) and the simulated
trajectory is bit-reproducible.

Topology / binding regime
=========================
Two reservoirs in the Filling phase at the SAME stages, chained in a cascade:
H_up (id 0, filling) -> H_down (id 1, filling) -> H_sink (id 2, real outlet),
plus an off-cascade control H_ctrl (id 3). Both filling hydros share the window
start_stage_id=1, entry_stage_id=4, so both are PreFilling at stage 0, Filling at
stages 1-3, and Operating at stages 4-5. This is the topology the volume-target
filling model (design spec section 9) says needs NO special handling: each carries
its OWN per-stage soft floor v_out[t] + sigma_fill[t] >= V_target[t] and the two
couple ONLY through normal cascade releases (H_up's release is downstream inflow
on H_down's water-balance row; the floor rows carry no inflow term).

V_target trajectory (both, rate 12.0, min_storage 60.0, zeta = 720 h * 0.0036 =
2.592 hm3 per m3/s), anchored backward from the dead volume:
  V_target[3] = min_storage           = 60.0
  V_target[2] = 60.0 - zeta * 12      = 28.896
  V_target[1] = 28.896 - zeta * 12    = -2.208  (clipped negative => trivially
                                                  satisfied, sigma_fill[1] = 0)
so each floor BINDS at the Filling stages 2 and 3 wherever inflow-only storage
falls short.

Inflow schedule (the crux). During Filling each hydro impounds (turbine/diversion
are pinned [0,0]; spilling water would only raise its own costly sigma_fill), so
storage accumulates inflow only and the upstream release onto H_down is ~0 at the
Filling stages. The own incremental inflows are deliberately DISTINCT so the two
sigma_fill shortfalls differ, making the per-floor INDEPENDENCE observable:

         stage:     0     1     2     3     4     5
  H_up   (0):      18     5     5     5    60    60
  H_down (1):      12     3     3     3    60    60
  H_sink (2):      20    20    20    20    20    20
  H_ctrl (3):      25    25    25    25    25    25

  - H_up storage (seed 0, impound 5 m3/s): 12.96 at stage 1, 25.92 at stage 2
    (< V_target[2] = 28.896 => sigma_fill[2] = 2.976), 38.88 at stage 3
    (< V_target[3] = 60.0 => sigma_fill[3] = 21.12). Both > 1e-6.
  - H_down storage (seed 0, impound 3 m3/s, ~0 upstream release): 7.776 at stage 1,
    15.552 at stage 2 (< 28.896 => sigma_fill[2] = 13.344), 23.328 at stage 3
    (< 60.0 => sigma_fill[3] = 36.672). Both > 1e-6, and DISTINCT from H_up's
    shortfalls => the floors are computed from each hydro's OWN V_target trajectory
    against its OWN realized storage, not a shared floor.
  - Operating (stages 4-5): both inflows recover to 60 m3/s so storage climbs above
    the dead volume and sigma^{v-} -> 0 by stage 5.
  - H_sink (20 m3/s, real fed reservoir) holds the cascade's released water; its
    turbine is small (max_turbined 20 m3/s in system/hydros.json) so the system
    cannot monetize a last-Filling-stage water dump from the filling reservoirs.
    Without that small cap the terminal horizon (finite, zero discount) would value
    the dead-volume water as cheap hydro generation routed through H_sink, and the
    LP would optimally spill the filling reservoirs at the last Filling stage
    (driving sigma_fill[3] to the full dead volume for BOTH and collapsing the
    independence) rather than hold it. The cap keeps both reservoirs impounding, so
    sigma_fill stays at the per-hydro shortfall. H_ctrl (25 m3/s, off-cascade,
    seeded mid) dispatches as an ordinary Operating plant at every stage.

PreFilling stage 0: both filling dams are absent, so H_up's and H_down's stage-0
incrementals short-circuit down the cascade (through the PreFilling H_down) onto
H_sink's real water-balance row.
"""

import os
import pyarrow as pa
import pyarrow.parquet as pq

script_dir = os.path.dirname(os.path.abspath(__file__))

N_STAGES = 6

# ── inflow_seasonal_stats.parquet ─────────────────────────────────────────────
#
# Schema: hydro_id INT32, stage_id INT32, mean_m3s FLOAT64, std_m3s FLOAT64
#
# Per-(hydro, stage) mean schedule; std is 0.0 everywhere (deterministic).
#
#             stage:    0     1     2     3     4     5
# H_up   (0, filling): 18     5     5     5    60    60   (short in Filling, recovers)
# H_down (1, filling): 12     3     3     3    60    60   (shorter own incremental)
# H_sink (2, real):    20    20    20    20    20    20   (own incremental)
# H_ctrl (3, control): 25    25    25    25    25    25   (off-cascade)
inflow_means = {
    0: [18.0, 5.0, 5.0, 5.0, 60.0, 60.0],
    1: [12.0, 3.0, 3.0, 3.0, 60.0, 60.0],
    2: [20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
    3: [25.0, 25.0, 25.0, 25.0, 25.0, 25.0],
}

inflow_hydro_ids = []
inflow_stage_ids = []
inflow_mean_m3s = []
inflow_std_m3s = []
for hydro_id in sorted(inflow_means):
    for stage_id in range(N_STAGES):
        inflow_hydro_ids.append(hydro_id)
        inflow_stage_ids.append(stage_id)
        inflow_mean_m3s.append(inflow_means[hydro_id][stage_id])
        inflow_std_m3s.append(0.0)

inflow_schema = pa.schema(
    [
        pa.field("hydro_id", pa.int32(), nullable=False),
        pa.field("stage_id", pa.int32(), nullable=False),
        pa.field("mean_m3s", pa.float64(), nullable=False),
        pa.field("std_m3s", pa.float64(), nullable=False),
    ]
)

inflow_table = pa.table(
    {
        "hydro_id": pa.array(inflow_hydro_ids, type=pa.int32()),
        "stage_id": pa.array(inflow_stage_ids, type=pa.int32()),
        "mean_m3s": pa.array(inflow_mean_m3s, type=pa.float64()),
        "std_m3s": pa.array(inflow_std_m3s, type=pa.float64()),
    },
    schema=inflow_schema,
)

inflow_path = os.path.join(script_dir, "inflow_seasonal_stats.parquet")
pq.write_table(inflow_table, inflow_path, compression="zstd")
print(f"wrote {len(inflow_table)} rows -> {inflow_path}")

# ── load_seasonal_stats.parquet ───────────────────────────────────────────────
#
# Schema: bus_id INT32, stage_id INT32, mean_mw FLOAT64, std_mw FLOAT64
#
# One bus (id 0); constant 150 MW mean; std 0.0 (deterministic). Per-block shape
# comes from load_factors.json (sized to each stage's block count).
load_bus_ids = [0] * N_STAGES
load_stage_ids = list(range(N_STAGES))
load_mean_mw = [150.0] * N_STAGES
load_std_mw = [0.0] * N_STAGES

load_schema = pa.schema(
    [
        pa.field("bus_id", pa.int32(), nullable=False),
        pa.field("stage_id", pa.int32(), nullable=False),
        pa.field("mean_mw", pa.float64(), nullable=False),
        pa.field("std_mw", pa.float64(), nullable=False),
    ]
)

load_table = pa.table(
    {
        "bus_id": pa.array(load_bus_ids, type=pa.int32()),
        "stage_id": pa.array(load_stage_ids, type=pa.int32()),
        "mean_mw": pa.array(load_mean_mw, type=pa.float64()),
        "std_mw": pa.array(load_std_mw, type=pa.float64()),
    },
    schema=load_schema,
)

load_path = os.path.join(script_dir, "load_seasonal_stats.parquet")
pq.write_table(load_table, load_path, compression="zstd")
print(f"wrote {len(load_table)} rows -> {load_path}")

print("done.")
