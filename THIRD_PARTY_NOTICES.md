# Third-Party Notices

Cobre is licensed under the Apache License 2.0 (see `LICENSE`). It bundles the
following third-party components as git submodules under `crates/*/vendor/` — the
solver libraries under `crates/cobre-solver/vendor/`, and qhull under
`crates/cobre-sddp/vendor/`. Each component retains its original license;
bundling does not relicense it.

These components are C/C++ libraries compiled and statically linked by the
owning crate's build script. They are **not** Rust crates and are therefore
**not** covered by cargo-deny's license checks (`deny.toml` governs only Cargo
dependencies). This file is the manual record of their license obligations.

## HiGHS — MIT

- Submodule: `crates/cobre-solver/vendor/HiGHS` (tag `v1.13.1`)
- Upstream: <https://github.com/ERGO-Code/HiGHS>
- Linked: statically, when the `highs` feature is enabled (enabled by default).
- License text: `crates/cobre-solver/vendor/HiGHS/LICENSE.txt`
- HiGHS bundles its own sub-dependencies; see
  `crates/cobre-solver/vendor/HiGHS/THIRD_PARTY_NOTICES.md` for the full list.

## Clp — Eclipse Public License 2.0 (EPL-2.0)

- Submodule: `crates/cobre-solver/vendor/Clp` (tag `releases/1.17.11`)
- Upstream: <https://github.com/coin-or/Clp>
- Linked: statically, **only** when the optional `clp` feature is enabled
  (disabled by default).
- License: <https://www.eclipse.org/legal/epl-2.0/>
- License text: `crates/cobre-solver/vendor/Clp/LICENSE`

EPL-2.0 source code is available via the upstream repository and via the
vendored submodule in this repository.

## CoinUtils — Eclipse Public License 2.0 (EPL-2.0)

- Submodule: `crates/cobre-solver/vendor/CoinUtils` (tag `releases/2.11.13`)
- Upstream: <https://github.com/coin-or/CoinUtils>
- Linked: statically, **only** when the optional `clp` feature is enabled
  (disabled by default).
- License: <https://www.eclipse.org/legal/epl-2.0/>
- License text: `crates/cobre-solver/vendor/CoinUtils/LICENSE`

EPL-2.0 source code is available via the upstream repository and via the
vendored submodule in this repository.

## Qhull — Qhull License

- Submodule: `crates/cobre-sddp/vendor/qhull` (tag `2020.2`)
- Upstream: <https://github.com/qhull/qhull>
- Linked: statically into `cobre-sddp`. Only the reentrant `libqhull_r` library
  is compiled (from `src/libqhull_r/`); the non-reentrant `libqhull`, the C++
  `libqhullcpp`, the CLI mains, tests, and docs in the submodule are not built.
- License: <http://www.qhull.org/COPYING.txt>
- License text: `crates/cobre-sddp/vendor/qhull/COPYING.txt`

## Rust crate dependencies

The C++ components above are recorded by hand because they are invisible to
Cargo. The Rust dependency graph is handled by two cooperating mechanisms:

- **License _checking_** — `deny.toml` (`[licenses] allow`) gates every crate's
  license against an allow-list, failing CI on anything outside it.
- **License _reproduction_** — `THIRD_PARTY_LICENSES.md` reproduces the full
  license text of every crate that ships in a binary (CLI release artifacts and
  Python wheels statically link them), satisfying the notice-retention clauses of
  whatever permissive licenses are present in the graph (MIT, Apache-2.0, BSD,
  ISC, Unicode, …). It is **generated** across all features and targets — a
  superset of any single distributed build — not hand-maintained; regenerate
  after any dependency change with:

  ```
  cargo about generate --all-features about.hbs -o THIRD_PARTY_LICENSES.md
  ```

  The accepted-license list in `about.toml` is kept in sync with `deny.toml`.

Two obligations are not captured by the generator and are handled explicitly:

- **Unicode-3.0** — `unicode-ident` is `(MIT OR Apache-2.0) AND Unicode-3.0`; the
  conjunctive Unicode license is reproduced in `THIRD_PARTY_LICENSES.md` along
  with the rest.
- **Apache-2.0 §4(d) NOTICE propagation** — the `arrow`/`arrow-*` and `parquet`
  crates ship their own `NOTICE`; that text is propagated in this repository's
  `NOTICE` file (cargo-about does not collect upstream NOTICE files).

No Rust dependency carries a copyleft license (no GPL/LGPL/MPL/EPL obligations);
`LGPL-2.1-or-later` appears only as an unused `OR` alternative in `r-efi`.
