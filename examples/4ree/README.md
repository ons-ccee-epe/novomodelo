# 4ree — Four-Region Brazilian Interconnected System

This example models the four-region Brazilian interconnected power system
(SUDESTE, SUL, NORDESTE, NORTE) with hydro and thermal generation over a
12-month planning horizon (January–December 2015).

The source data is the `4ree` example from the
[sddp-lab](https://github.com/rjmalves/sddp-lab) reference implementation,
located at `example/4ree/` in that repository.

## System Summary

| Entity   | Count | Notes                                                                           |
| -------- | ----- | ------------------------------------------------------------------------------- |
| Buses    | 5     | SUDESTE (0), SUL (1), NORDESTE (2), NORTE (3), NOFICT1 (4)                      |
| Hydros   | 4     | One per real region, independent cascades                                       |
| Thermals | 126   | All original thermals, remapped to 4 real buses                                 |
| Lines    | 5     | SUDESTE-SUL, SUDESTE-NORDESTE, SUDESTE-NOFICT1, NORDESTE-NOFICT1, NORTE-NOFICT1 |
| Stages   | 12    | Monthly, Jan 2015 – Dec 2015                                                    |

## Usage

Validate the case (checks all 5 validation layers):

```sh
novomodelo validate examples/4ree
```

Run the optimization:

```sh
novomodelo run examples/4ree
```

## Conversion Decisions

### Bus ID remapping

sddp-lab uses 1-indexed bus IDs; Novomodelo uses 0-indexed IDs. The mapping is:

| sddp-lab ID | sddp-lab name | Novomodelo ID | Novomodelo name |
| ----------- | ------------- | -------- | ---------- |
| 1           | SUDESTE       | 0        | SUDESTE    |
| 2           | SUL           | 1        | SUL        |
| 3           | NORDESTE      | 2        | NORDESTE   |
| 4           | NORTE         | 3        | NORTE      |
| 5           | NOFICT1       | 4        | NOFICT1    |

All `bus_id` references in hydros, thermals, and lines are remapped accordingly.
Thermal IDs are also remapped from 1-indexed (sddp-lab) to 0-indexed (Novomodelo).

### NOFICT1 as a transit hub

sddp-lab includes a fictitious aggregation node NOFICT1 (sddp-lab id=5) with zero
load that acts as an intermediate hub connecting northern generation to southern
load centers. In this conversion NOFICT1 is retained as bus id=4 because three of
the five modeled transmission lines use it as an endpoint.

All 126 thermals in sddp-lab are connected to real buses 1–4; none were connected
to bus 5, so no thermal reassignment was needed. No hydro plant is assigned to
NOFICT1.

- **Lines retained**: all ten sddp-lab lines collapse to five Novomodelo bidirectional
  entries using `capacity.direct_mw` / `capacity.reverse_mw`:
  - `SUDESTE_SUL` (direct: 7500 MW, reverse: 5470 MW)
  - `SUDESTE_NORDESTE` (direct: 1000 MW, reverse: 600 MW)
  - `SUDESTE_NOFICT1` (direct: 4000 MW, reverse: 2940 MW)
  - `NORDESTE_NOFICT1` (direct: 3500 MW, reverse: 3300 MW)
  - `NORTE_NOFICT1` (direct: 10000 MW, reverse: 4407 MW)

  The sddp-lab model used paired unidirectional lines for asymmetric capacity.
  Novomodelo's single bidirectional line entry encodes both directions.

### Inflow model (NOT converted)

sddp-lab uses "Naive" inflow scenarios with per-season LogNormal marginal
distributions and identity Gaussian copulas (independent hydros). Novomodelo uses
PAR(p) with additive normal noise.

Converting LogNormal(mu, sigma) parameters to PAR(0) normal parameters requires
moment-matching (`mean = exp(mu + sigma^2/2)`), but the resulting distributions
have fundamentally different tail shapes, making convergence bound comparisons
unreliable.

Decision: provide seasonal statistics via the `scenarios/` directory and run
with stochastic inflows using PAR(p). The `scenarios/inflow_seasonal_stats.parquet`
file supplies per-season means and standard deviations derived from the sddp-lab
LogNormal parameters via moment-matching. The resulting inflow distributions differ
from the original LogNormal tails, so convergence bounds remain incomparable with
sddp-lab, but the model produces physically plausible hydro dispatch rather than
zero-inflow drawdown.

### Risk measure

sddp-lab's 4ree uses CVaR (alpha=0.5, lambda=0.5). This example uses the default
Expectation (risk-neutral) risk measure. CVaR is also available via `stages.json`.
The two objective functions are not directly comparable even with matching risk
measures due to the differences in inflow distributions noted above.

### Discount rate

sddp-lab's graph edges all have `discount_rate: 0.0`. Novomodelo's `stages.json` sets
`annual_discount_rate: 0.0` to match.

### Spillage penalty

The sddp-lab `hydros.csv` lists `spillage_penalty = 1` ($/hm³) for all hydros.
The global spillage penalty in `penalties.json` is set to 1.0 $/hm³.

### Initial storage

Initial reservoir storage values are taken directly from `hydros.csv`:

| Hydro (Novomodelo ID) | Initial storage (hm³) |
| ---------------- | --------------------- |
| 0 (SUDESTE)      | 38343.9               |
| 1 (SUL)          | 10068.8               |
| 2 (NORDESTE)     | 9030.2                |
| 3 (NORTE)        | 5161.9                |

## Known Limitations

- **Results are NOT comparable to sddp-lab**: different stochastic model
  (PAR(p) normal vs. lognormal), different risk measure (Expectation vs. CVaR),
  and differences in how the NOFICT1 hub lines are modeled all mean the objective
  values and dispatch patterns will differ.
- **NOFICT1 carries no load and no generation**: as a fictitious hub node it has a
  zero-load balance constraint. Energy may flow through it in transit between NORTE,
  NORDESTE, and SUDESTE, but there is no generator or consumer attached directly
  to it.
