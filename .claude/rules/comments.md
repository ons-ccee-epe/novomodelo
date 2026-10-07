---
paths:
  - "**/*.rs"
---

# Novomodelo Comment & Documentation Rules

Governs every comment in any `.rs` file in this workspace. Auto-loads on the
`**/*.rs` glob — including infra crates (`novomodelo-core`, `novomodelo-io`, `novomodelo-solver`,
`novomodelo-stochastic`, `novomodelo-comm`) bound by the genericity hard rule. Keep all
directive statements free of algorithm names.

**Default is silence.** A comment is a liability — it costs reader attention and
it can rot away from the code it describes. It is not a free good and it is not a
sign of care. The job of this rule is to keep the **small set** of comments that
prevent real bugs and **delete everything else**. The load-bearing few are named
in §7 (Protected Contracts) and exemplified in `.claude/rules/sddp.md`; protect
those, and be ruthless with the rest.

---

## 0. The Default-Off Discipline (read first)

You are writing code with **no comments** until a specific comment earns its
place against §1 (the Deletion Test). In order, before any comment exists:

1. **Refactor first (§2).** Rename → extract a well-named function/const →
   introduce a type. A comment that a better name would carry is a refactor you
   skipped, not a comment you earned.
2. **Relocate (§4).** Most "why" is not line-local. The why of a _change_ goes in
   the commit message; a durable cross-cutting contract goes in a `rules/*.md`
   file or a module doc; a behavioural fact goes in a **test name**. Inline
   comments are the _residual_: the line-local trap that none of those can carry.
3. **Hoist (§3).** A fact that repeats across siblings (struct fields, match arms,
   call sites) is stated **once** at the enclosing scope, never per item.
4. **Only then comment** — and write the shortest form that survives the Deletion
   Test, referencing owning symbols instead of copying their formulas or numbers.

This default-off stance is injected into specialist dispatch prompts, not only
read here: any agent writing Rust is told "default to no comment; apply the
Deletion Test; keep it to one clause." The rule and the prompt say the same thing.

### The `missing_docs` floor (terse, not absent)

`missing_docs = "warn"` mandates a doc on every _publicly reachable_ item, so a
`pub` field whose name already says everything still needs a `///`. Pay that floor
with **one terse line** — never a multi-line essay, and never the same clause
restated across siblings: hoist the shared context to the struct/module doc (§3)
so each field's line carries only what is unique to it. `#[allow(missing_docs)]` is
**not** the escape hatch — do not suppress the lint to delete an obvious field's
doc; write the one terse line.

`pub(crate)` is the honest fix when an item is **genuinely internal** — not a
deliberate external API and not constructed by a sibling crate's tests. It removes
the mandate by correcting the visibility, not by suppressing a lint. A genuine
public API still earns real docs, but those describe the **contract**, not the
field name.

---

## 1. The Deletion Test (the one gate)

Replaces the older "Earned-Comment Test": that test asked "does this add
_anything_?" and almost everything passed. This one is biased toward deletion.

> **Delete the comment.** Now — reading only the code, would a competent engineer
> (a) introduce a bug, (b) "simplify" something correct into something wrong, or
> (c) be unable to recover a fact that lives **outside this file**?
> **If none → leave it deleted. If any → restore it, then cut to the single clause
> that triggers the "yes".**

Worked:

- `n_hydros` "Number of hydro plants…" → delete → nobody bugs out → **stays deleted.**
- a bare `if v < 0.0` rejecting a negative rate → delete the why → a maintainer
  removes the "over-strict" check → bug (b) → restore, **cut to one clause** naming
  the consumers that assume non-negativity.
- the "patch the three NCS sites identically" contract → delete → someone diverges
  them → wrong bound → **keep** (this is a §7 protected contract).

| If, with the comment deleted, the code…                          | Verdict                                  |
| ---------------------------------------------------------------- | ---------------------------------------- |
| would be mis-edited into a bug (wrong-but-compiling alternative) | **KEEP** — Contract (Voice 1), 1 clause  |
| would be "simplified" by removing a load-bearing choice          | **KEEP** — Rationale (Voice 2), 1 clause |
| loses a fact that lives outside this file (spec, sibling, test)  | **KEEP** — pointer by symbol             |
| reads exactly the same to a competent engineer                   | **DELETE**                               |
| would be clearer with a better name / type / extracted fn        | **REFACTOR** (§2), don't comment         |
| loses only "how it got here" / what the next line does           | **DELETE** (→ git / a test name)         |

When in doubt, **delete** and put the thought in the commit message. A missing
comment is cheap to add back the day someone actually needs it; a wrong comment
ships a lie.

### Where the test bites: not just struct fields

The Deletion Test applies to **every** comment site — module docs, **function
docs**, **function bodies**, **call sites**, and struct fields alike. Function code
is where the most-overlooked bloat hides; these forms almost always DELETE or
TIGHTEN (do not stop at struct fields):

- **A function doc that narrates its own body** ("then for each opening: evaluates
  X, computes Y, patches Z, solves, records") — the body already says this. Keep a
  one-line purpose plus `# Errors` / `# Panics`; delete the blow-by-blow.
- **A call-site comment that names the function it precedes** (`// Populate scratch
and load the LP` directly above `lb_init_rank0(...)`) — the call is
  self-documenting. DELETE; if it is _not_ self-documenting, **rename the function**
  (§2), don't annotate the call.
- **Pipeline labels** (`// Step 1`, `// Phase 3`) — the call order is visible at the
  call site and the function names carry the phases. DELETE.
- **An "explains why this is long/complex" note with no `#[allow]`** — if no lint is
  firing, the comment justifies nothing; it pre-empts a complaint nobody is making.
  DELETE. (A real `#[allow(...)]` keeps its D4 rationale.)
- **The same clause repeated down a function** ("constant across openings" on three
  separate lines) — state it once at the site that earns it, delete the echoes (§3).
- **A public-fn `# Arguments` list restating each param's name/type** — keep only the
  arguments that carry a nugget (a length contract, a rank-0 behaviour); the
  parameter list and the parameters' own type docs carry the rest.

`# Errors`, `# Panics`, `# Safety`, and a one-line purpose are the function-doc
**survivors** — they are the callable contract a reader cannot recover from the
body. Everything else in a function doc or body faces the same default-delete.

---

## 2. Refactor-first (the gate before a comment)

Before writing a comment, spend the same effort on the code:

- **Rename.** `t` → `total_stage_hours`; `check()` → `reject_negative_filling_rate()`.
  A comment explaining a name is a renamed-variable you skipped.
- **Extract.** A banner fencing a block inside a long function (N5) is an
  `extract_function` you skipped; the function name is the comment.
- **Introduce a type / newtype / enum.** A comment explaining what an `f64` means
  or which states a `bool` has is a type you skipped.

Only when none of these can carry the fact does a comment earn its place.

---

## 3. The repeated-clause rule (hoist to one owner)

If the **same clause** appears on more than one sibling — fields of a struct, arms
of a match, sites of a pattern — it does not belong on each item. State it **once**
at the enclosing scope (the struct/module doc, the match's preamble) and let the
per-item comments carry only what is **unique** to the item.

The tell is literal repetition: when "for stage 0" or "sourced from `StageContext`"
appears on ten fields, nine of them are noise and the tenth is the struct doc.

---

## 4. Relocation routing — where the "why" actually lives

Inline is the last resort. Route each fact to its durable home:

| The fact is about…                                | It lives in…                                   |
| ------------------------------------------------- | ---------------------------------------------- |
| why this **change** was made                      | the **commit message**                         |
| a durable cross-cutting **contract**              | `rules/*.md` / a module doc (§7)               |
| an expected **behaviour**                         | a **test name** (`test_<scenario>_<expected>`) |
| how the code **used to** work / who changed it    | **git** (never a comment)                      |
| a **line-local trap** none of the above can carry | a terse **inline** comment                     |

If you find yourself writing inline what is really the rationale for the diff,
stop — that belongs in the commit body. The inline comment keeps only the standing
invariant a future _reader_ (not reviewer of this diff) needs.

---

## 5. The Four Voices — the only comments that survive §1

Every surviving comment speaks in one of the first three voices; Voice 4 is a
tightly-bounded fourth. The Narrator voice is **not allowed** in shipped code.
These are the _survivors_ of the Deletion Test, not a menu of things you may add.

### Voice 1 — Contract (the house style for the load-bearing few)

A present-tense invariant **+ the wrong-but-compiling alternative it forbids +** a
citation of the owning symbol. `.claude/rules/sddp.md` is the gold standard.

> _Exemplar (`crates/novomodelo-sddp/src/backward.rs`):_ the subgradient coefficient is
> `rc_scaled / col_scale[col]` — **divided, not multiplied** — because the pin sets
> `v_scaled = v_orig / col_scale`. Stating the forbidden alternative (`* col_scale`)
> is what makes it load-bearing.

### Voice 2 — Rationale

_Why_ a non-obvious choice was made; what regresses if a maintainer "simplifies"
it. The `X instead of Y` form **is** rationale and is kept **only while Y is a
still-plausible wrong simplification**. One clause, not a paragraph.

> _Exemplar (`crates/novomodelo-sddp/src/forward.rs`):_ "Welford's online algorithm
> instead of the two-pass naive formula to avoid catastrophic cancellation when
> sum_sq ≈ n·mean²."

### Voice 3 — ~~Narrator~~ (NOT ALLOWED)

What the next line does; how it _used to_ work; which ticket/phase changed it;
project dates; byte-count deltas. Belongs in names, git, commit messages, or test
names — never a shipped comment.

### Voice 4 — Intent/Seam (the only licensed future-tense)

Why a currently-unused item exists and what will activate it — **pinned to the
`#[allow(dead_code)]` / `#[allow(unused_*)]`** it justifies. The suppression
attribute is the durable anchor: when the future caller lands, the lint fires and
forces the comment current.

> _Exemplar:_ "Pre-allocated here; read sites are not yet wired" (`workspace.rs`).

---

## 6. Intra-Comment Surgery (cut aggressively, protect the invariant line)

A comment block often welds several voices together — a contract sentence, a
rationale clause, a rot tail. **Adjudicate clause-by-clause, never block-by-block:
cut every clause that fails §1, but never delete the invariant line.**

Procedure:

1. Strip any plan token or drift-ref (`(F1-007 fix)`, `(see MEMORY.md)`, `:1555`).
2. Strip any clause that restates the code, narrates history, or copies a
   formula/number that a named symbol already owns.
3. A determinism/correctness sentence is a **Contract clause** → KEEP, tightened.
4. An `X instead of Y` clause is **Rationale** → KEEP while Y is still plausible.

The output is short. If the surgery leaves a paragraph, you under-cut — go again.

---

## 7. Protected Contracts (never delete — always considered)

A handful of comments are the difference between right and wrong bounds. A blunt
minimization drive that deletes one of these is **worse** than the bloat it
removes: bloat is boring, a deleted contract is a silent wrong-bound bug that
compiles and passes most tests. So these are **protected**, and protection is made
durable by living where it cannot be missed:

- The project `CLAUDE.md` (always loaded, independent of which file you touch)
  carries the hard rule: _never delete or weaken a load-bearing correctness
  contract; it is pinned to a named regression test and a `rules/*.md` entry._
- The contracts themselves live in `.claude/rules/sddp.md` (the numerical/algorithm
  invariants) and are each tied to a **named regression test or deterministic case**
  (`D06`, `D15`, …).

Because the authoritative statement lives in `sddp.md` + a test, the **inline**
copy can be terse — a pointer, not a re-derivation ("keep the three patch sites
identical — D15 contract"). When you tighten a contract comment, move the full
statement to its `rules/` home if it is not already there; never let tightening
**lose** the invariant.

---

## 8. The Comment-Skeptic Pass (very aggressive)

A dedicated review pass (distinct from the general code-reviewer) whose **only job
is to propose comment deletions**, biased hard toward removal. Its rubric is the
Deletion Test (§1) applied to every comment in the diff, defaulting to DELETE:

- Restatement of the adjacent declaration/line → **delete**.
- A clause repeated across siblings → **hoist** to the enclosing scope, delete the
  copies (§3).
- A copied formula/number a named symbol owns → **replace with the symbol** (§5
  Contract-mirroring; drift risk).
- A multi-line load-bearing comment → **cut to one clause** (§6).
- History/narration → **delete** (→ git / test name).
- A **function doc narrating its body**, a **call-site comment naming the callee**, a
  **pipeline label** (`Step 1`/`Phase 3`), or an **orphaned "why it's long" note with
  no `#[allow]`** → **delete** (keep only the purpose line + `# Errors`/`# Panics`).
- A **public-fn `# Arguments` list** restating each param → **cut to the nugget
  arguments** (length/rank/unit contracts); delete the name/type restatements.
- A `missing_docs`-driven restatement → **tighten to one terse line and hoist the
  shared context (§3)**; propose `pub(crate)` only when the item is genuinely
  internal and not test-exposed (§0). Never `#[allow(missing_docs)]` to delete it.

It keeps a comment **only** when deletion would cause (a) a bug, (b) a wrong
"simplification", or (c) loss of an out-of-file fact. Its default verdict is
"delete; justify what you keep" — the inverse of the writer's instinct. It runs at
epic/plan boundaries alongside the simplifier and reports proposed deletions as
data for the main session to apply → test → re-verify (it never deletes a §7
protected contract without surfacing it).

---

## 9. Lint suite (mechanical pressure — flag, don't auto-reject)

Lints cannot _judge_ a comment, but they create pressure where there is currently
none. Each flags candidates for the §8 skeptic pass; none auto-rejects, because
this codebase has legitimately dense contract comments. Implement as CI checks
(awk/ripgrep, alongside the existing comment/doc-integrity gates):

1. **Restatement heuristic** — a `///`/`//` line whose tokens (minus stopwords) are
   a subset of the immediately-following declaration's identifier tokens. Flags
   "Number of hydro plants" on `n_hydros`.
2. **Repeated-clause detector** — the same normalized clause on ≥3 sibling fields
   in one struct → flag to hoist (§3).
3. **Drift-copy detector** — a comment containing a formula or numeric literal that
   also appears verbatim in a named symbol in the same crate → flag (§5).
4. **Comment-to-code ratio** — per item (fn/struct), comment lines ÷ code lines
   above a threshold → flag for the skeptic pass (advisory only; contracts can be
   dense).
5. **`missing_docs`-restatement** — a `///` whose body matches heuristic 1 on a
   `pub` field → flag the `pub(crate)`/allow choice (§0).

Existing keyword gates (N2/N3/N4 plan tokens, `.claude/` paths, `file.rs:NNN`)
stay. The lints above _extend_ them from "banned tokens" to "low-value shapes".

---

## 10. Principles

### Durability — a shipped comment stays true with zero maintenance

Reference only things that cannot rot: a symbol name (`Mod::symbol`) or intra-doc
link (`[Symbol]`); a named regression **test**; a stable external spec anchor
(`output-schemas.md §5.1`) in a declared source-of-truth root (`novomodelo-docs`).
**Never** `file.rs:NNN`, commit hash, dead path, or `MEMORY.md` / `.claude/`. For a
`Symbol at file.rs:NNN` hybrid, keep the symbol, strip the `:NNN`. If a fact can
rot and cannot be made un-rottable, delete it.

### Provenance — history lives in git

Comments are present-tense. Convert story-tails to the durable fact; if a discovery
pins a contract via a living test or deterministic case, **name the test/case**.
Carve-outs (NOT history, KEEP): bibliographic years `Author (YYYY)`; calendar/
data-coverage years.

### Contract-mirroring beats DRY (but a mirror is shape, never a number)

Deliberately duplicated contracts (producer vs consumer, rustdoc vs spec) are
redundancy-with-purpose; do not DRY them away. But a mirror restates the **shape**
and references the owner by symbol — **never copies a magic number or formula**. A
drifted mirror is a lie; fix it or reduce it to the shape-only form.

### Single-owner

A comment naming the **sole owner** of a byte layout, or the **single hot-path
entry** ("never bypass"), is load-bearing — keep it.

### Length — the goal is the load-bearing clause, not brevity for its own sake

Length is not a virtue and not a vice; **every clause pays rent**. A comment 3× the
code it annotates is almost always restatement, repeated-across-siblings, or a
drift-copy — cut to the load-bearing clause (§6). The legitimately long comment is
rare: the **only** place an invariant is explained, or `schemars`-derived rustdoc
that **is** the `.schema.json` artifact (never delete that as redundant). Do not
use "length is not the metric" as a license to keep a paragraph where a clause
would do.

---

## 11. The Directive Set

### DO

- **D1 — Contracts.** When the obvious alternative is wrong, say so: invariant +
  forbidden alternative + owning symbol. (One clause.)
- **D2 — SAFETY.** Every `unsafe` block carries a multi-clause `// SAFETY:` mapping
  each Rust-side invariant to the C precondition it satisfies. (Reinforces
  `unsafe_code = "forbid"`.)
- **D3 — Units.** Annotate fields/consts carrying physical units, dimensionless
  factors, a sign convention, or a coefficient-evaluation order — **and the
  inverse-direction trap** ("divided by `col_scale`, not multiplied"). This is the
  one comment a self-documenting field name usually cannot carry.
- **D4 — Rationale above suppression.** Every `#[allow(...)]` for a refactor-decision
  lint on the **closed list** (`clippy::too_many_arguments`, `too_many_lines`,
  `type_complexity`, `dead_code`, `unused_*`) and every borrow-checker workaround
  carries a rationale: why the refactor that removes the lint is inappropriate.
  `// Rationale:`, `// RATIONALE:`, and inline-trailing forms count. For
  `dead_code`/`unused_*` it **is** Voice 4. The list is closed: numeric-cast lints
  (`cast_possible_truncation`, `cast_precision_loss`, `cast_sign_loss`,
  `cast_possible_wrap`), `needless_pass_by_value`, and `missing_docs` are **not** on
  it, so D4 mandates no rationale for a suppression of one of those — such an
  `#[allow]` may still be load-bearing (CI's zero-warning bar needs it), but a bare
  opener there is not a D4 violation. `missing_docs` in particular is never the
  escape hatch to delete an obvious field's doc; write the one terse line per §0.
- **D5 — Determinism.** Where solve/thread order is deliberately decoupled from
  aggregation order (stable sort after a parallel region, canonical iteration,
  online accumulator), say why — it upholds the declaration-order hard rule.

### DON'T

- **N1 — No what-narration.** Rename instead.
- **N2 — No history narration.** Ban `replaces`/`formerly`, byte-count deltas,
  commit hashes, project-event dates ("discovered 2026-06"), plan tokens. KEEP the
  present-tense fact. Carve-outs (not history): bibliographic years `Author (YYYY)`;
  calendar/data-coverage years; a deterministic regression-case id (`D06`/`D15`)
  naming a still-existing case that pins a contract.
- **N3 — No drift-prone refs.** Ban `file.rs:NNN` (incl. en-dash and `:N, :M`
  forms), commit hashes, dead paths, `MEMORY.md` / `.claude/` paths. Reference by
  symbol, intra-doc link, or named test. Stable external spec anchors (`§x.y`) into
  a source-of-truth root are allowed. For a symbol+line hybrid, strip the line.
- **N4 — No plan/workstream leakage** in shipped code, `README.md`, `CHANGELOG`, or
  inline test/bench comments. Ban `Epic`/`ticket`/`T0NN`/`sprint` and workstream
  forms `F-NNN`, `FN-NNN`, `W-N`. When a banned token is a trailing tag on a
  contract/rationale line, amputate only the tag, preserve the invariant. Plan refs
  in **test names** remain allowed.
- **N5 — No banners fencing groups inside one long function** — extract a function.
  Decorative dividers between top-level items, in `extern "C"` blocks, and in
  `#[cfg(test)]` modules are fine.
- **N6 — Don't duplicate a source-of-truth in prose** — but preserve
  contract-mirroring (§10). Exceptions: `schemars`-derived rustdoc (it _is_ the
  artifact) and single-owner byte tables.

### Special-case clauses

- **TODO/FIXME.** A shipped `TODO` MUST carry a durable behavioural tag
  (`TODO(historical-replay-non-monthly)`) and SHOULD name the guard/test enforcing
  the current limitation. Never a plan token (`TODO(Epic..)`). A bare ownerless
  `TODO` is discouraged.

---

## 12. Worked before/after exemplars (the new center of gravity is DELETE)

The exemplars above (Voices 1/2) defend _keeping_ the load-bearing few. These show
the common case: **deleting and tightening.**

### A — Verbose load-bearing comment → one clause (`crates/novomodelo-io/src/constraints/bounds.rs`)

```rust
// BEFORE — 8 lines, two drift-prone formula copies
// Mirror the entity-reader check in `validate_filling_configs`: a per-stage
// `filling_min_rate_m3s` override is non-negative. The filling-target backward
// fold (`build_filling_v_target`: `running -= ζ·rate`) and the sufficiency check
// (`check_filling_sufficiency`: `capacity += ζ·rate`) both assume `rate ≥ 0`; a
// negative override otherwise reverses the V_target floor and the feasibility
// budget. The finiteness gate above would pass a negative value straight through.

// AFTER — one clause; keeps the load-bearing "why", drops the formula copies
// build_filling_v_target and check_filling_sufficiency assume rate ≥ 0; a negative
// override silently inverts the V_target floor (validate_filling_configs enforces
// this for the entity; the finiteness gate above does not).
```

The error message already carries the _what_ (`"value must be >= 0.0"`); the
comment owes only the why-it-matters + the mirror pointer. The formula copies
(`running -= ζ·rate`) are deleted — the symbol names _are_ the reference.

### B — Restatement boilerplate on a struct → hoist + tighten (`LbEvalSpec`)

```rust
// BEFORE — a multi-line doc essay per pub field; "stage 0" / "sourced from
// StageContext" repeated ~10×; more comment than code.
/// Structural LP template for stage 0.
pub template: &'a StageTemplate,
/// Number of hydro plants with inflow noise.
pub n_hydros: usize,
/// ... (16 more, multi-line, mostly restating the field name) ...

// AFTER — cross-cutting facts hoist to the struct doc (§3); each field keeps ONE
// terse line (the missing_docs floor); nuggets stay; drift-copies (the LP column
// formula) go. NO `#[allow(missing_docs)]`.
/// Stage-0 inputs for `evaluate_lower_bound`. The lower bound evaluates stage 0
/// only; slice/range fields come from the stage-0 `StageContext`; NCS fields are
/// empty when no stochastic NCS exist.
pub struct LbEvalSpec<'a> {
    /// Stage-0 LP template.
    pub template: &'a StageTemplate,
    /// Hydros carrying inflow noise.
    pub n_hydros: usize,
    /// MW, id-sorted (the order `transform_ncs_noise` emits its bound buffers).
    pub ncs_max_gen: &'a [f64],
    /// Keep the forward, backward, and lower-bound patch sites identical — the
    /// "patch NCS identically" contract; a divergence understates the bound (D15).
    pub ncs_stochastic_windows: &'a [(Option<i32>, Option<i32>)],
    /// Always `0`: column-bound state pinning leaves no rows before the z-inflow block.
    pub z_inflow_row_start: usize,
    // ...
}
```

The multi-line essays collapse to one terse line each; the shared context lives
once in the struct doc; the drift-prone LP-column formula is gone; the D15 contract
survives (a §7 protected contract). `missing_docs` keeps each field's one line —
that floor is paid, not suppressed.

### C — Function doc + body → purpose + contract (`lb_evaluate_stage_0`)

The same default-delete applies inside functions — this is **not** struct-only.

```rust
// BEFORE — the doc narrates the loop body; an orphaned "why it's long" note with
// no #[allow]; call sites and body lines restate what the code does.
/// Step 2 — truncation precompute and per-opening LP evaluation.
/// Precomputes the PAR lag matrix and eta floor (constant across openings), then
/// iterates over all stage-0 openings. For each opening: evaluates PAR inflows,
/// computes effective eta, patches row bounds, patches NCS column bounds
/// (correctness-critical per-opening step), solves, and records the objective.
/// Writes the per-opening objectives into `scratch.objectives_buf`.
/// # Errors
/// Returns [`SddpError::Infeasible`] ... or [`SddpError::Solver`] ...
// The per-opening loop body accounts for the length; it cannot be meaningfully
// split without fragmenting correctness-critical sequential steps ...
fn lb_evaluate_stage_0(...) { ...
    // Build noise_buf and z_inflow_rhs_buf from effective eta.
    for (h, &eta_eff) in ...

// AFTER — one-line purpose + the # Errors contract; the narration, the orphaned
// rationale (no #[allow] is firing), and the call-naming body comment are gone.
/// Truncation precompute (PAR lag matrix + eta floor, constant across openings),
/// then a per-opening LP solve writing each objective into `scratch.objectives_buf`.
/// # Errors
/// Returns [`SddpError::Infeasible`] ... or [`SddpError::Solver`] ...
fn lb_evaluate_stage_0(...) { ...
    for (h, &eta_eff) in ...
```

And at the caller, the comment that merely names the callee is deleted:

```rust
// BEFORE
// Populate scratch buffers and perform append-only LP load.
lb_init_rank0(solver, fcf, spec, ...);
// AFTER
lb_init_rank0(solver, fcf, spec, ...);
```

What survives in a function doc: the one-line purpose and the `# Errors` /
`# Panics` / `# Safety` contract (a reader cannot recover those from the body). A
load-bearing in-body comment survives only when deleting it would let a maintainer
introduce a bug — e.g. "the NCS patch MUST stay inside the per-opening loop (D15)".

---

## E6 — Dual-owned wire formats checklist

The following formats are **dual-owned**: a serializer owns the byte layout and
one or more callers depend on exact round-trip fidelity and forward/backward
compatibility. Each must carry **both** tests:

1. A **round-trip `#[test]`** — serialise → deserialise → assert equality.
2. A **reject-old-version `#[test]`** — feed a byte payload with a stale version
   byte and assert the decoder returns an error (not silent corruption).

| Format                                   | Authoritative owner symbol                                                          | Reject-test note                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| ---------------------------------------- | ----------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `cut/wire.rs`                            | `serialize_cut` / `deserialize_cut`                                                 | Own format tag; direct reject test.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `policy/codec.rs`                        | policy encode/decode entry points in `policy/codec.rs`                              | No version byte — FlatBuffers uses schema-evolution FORWARD-COMPAT; the reject role is the legacy-slot-ignored test in `crates/novomodelo-io/tests/flatbuffers_schema_conformance.rs`. The boundary-cut load path layers its own CLEAN-BREAK reject on top: a resolved pool whose `cost_scale_factor` reads `None` (a pre-`id:8` buffer) fails loudly instead of tolerating the absence, pinned by `boundary_load_rejects_pre_self_describing_checkpoint` in `crates/novomodelo-sddp/tests/boundary_self_describing_clean_break.rs`. The `CheckpointManifest` root (`manifest.bin`, the `metadata.json` replacement) is the exception to the forward-compat pattern above: it carries its own `format_version` field and a DIRECT version reject (like `cut/wire.rs`), so `deserialize_checkpoint_manifest` fails a version mismatch before any payload parse — pinned by `checkpoint_manifest_rejects_stale_format_version` (conformance suite) and `read_policy_checkpoint_rejects_stale_manifest_version_before_parsing_payloads` (read path). |
| `workspace/workspace.rs` `CapturedBasis` | `CapturedBasis::to_broadcast_payload` / `CapturedBasis::try_from_broadcast_payload` | Own format tag; direct reject test.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `cut_sync`                               | `cut_sync` serialisation entry points                                               | No own format tag — the reject is DELEGATED to `cut::wire` (it serialises via `cut::wire`'s `CUT_WIRE_FORMAT_TAG`); `cut::wire`'s reject test discharges it.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |

Enforced by `cargo test` + review. Not a bespoke CI script. The reject-test note
records, per format, whether the "reject/tolerate old version" obligation is met by
its own version byte, by delegation, or by FlatBuffers forward-compat — so a future
reader does not file a false "missing test" finding for the delegated/forward-compat
rows.
