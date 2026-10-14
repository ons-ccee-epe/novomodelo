---
paths:
  - "**/*.rs"
---

# Novomodelo Import Style Rules

Governs how every `.rs` file in this workspace refers to an item defined
outside the current module: `use` import vs. an inline fully-qualified path
(`crate::indexer::HydroSys`, `novomodelo_core::temporal::StageStateConfig`). Auto-loads
on the `**/*.rs` glob, sibling to `.claude/rules/comments.md`.

## 1. The default-import rule

**DEFAULT: import.** Every type, trait, or function referenced in a
`pub`/`pub(crate)` **signature** — a function's parameters/return type, a
`pub`/`pub(crate)` struct or enum field's type, a type alias's target, a
trait definition's method signatures, or a trait-impl method's signature
(publicly reachable via the trait even though Rust forbids writing `pub` on
it) — MUST be imported and referenced by its bare name. No qualified path in
a signature position unless it falls under §2.

A name used more than once in a file SHOULD be imported, even outside a
signature position; this is a guideline, not a gate — the enforcement below
targets signatures only.

## 2. Reserved qualified-path cases

A qualified (or module-qualified, see §3) path is legitimate, never a
violation, when:

- **(i) Genuine in-file name collision.** Two distinct types share a leaf
  name reachable in the same scope (a builder-local `Phase` vs a
  solver-local `Phase` enum). Qualify the locally rarer name; do not
  introduce an `as`-alias unless the alias is already the file's established
  idiom.
- **(ii) Single-use reference inside a large `#[cfg(test)]` module.** A path
  that would otherwise import a name used exactly once, deep inside a large
  test module, may stay qualified rather than adding a module-scoped import
  purely for that one line. Never applies to a signature position.
- **(iii) A position where an import cannot substitute.** Macro-internal
  token streams and some attribute arguments are not name-resolution
  positions; an import has nothing to attach to there.
- **(iv) The path is itself the documentation.** Rare; justify inline when it
  occurs (e.g. a doc example deliberately showing the fully-qualified form
  for a reader unfamiliar with the crate's re-export surface).

A qualified path that does not fall under (i)-(iv) and appears in a
`pub`/`pub(crate)` signature is a violation to fix, not a style choice.

## 3. The module-qualified resolution (the `fmt::Result` pattern)

Some collisions (§2.i) are with the **prelude itself**, not another
project type — most commonly `Result`, `Formatter`, `Display`, and similarly
prelude- or ambient-shaped leaf names. Bare-importing `std::fmt::Result` as
`Result` would shadow every other `Result<T, E>` use in the file; the fix is
never a raw `as`-alias, it is **importing the enclosing module and
qualifying one level**:

```rust
use std::fmt;

impl fmt::Display for Phase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // ...
    }
}
```

This is the established idiom across the workspace's `Display`/`Debug`
impls — prefer it over leaving `std::fmt::Result`/`std::fmt::Formatter`
fully spelled out, and over any `as`-alias.

## 4. Canonical public path

When importing a type re-exported at more than one depth, import the
**shallowest curated re-export**, never a deeper internal module path that
happens to also resolve:

```rust
// A curated re-export exists at crate root or a curated module:
use novomodelo_core::StageId;             // not novomodelo_core::model::temporal::StageId
use crate::indexer::HydroSys;        // crate::indexer re-exports it; not the
                                      // deeper file path that defines it
```

If no shallower re-export exists, the module path the item is actually
defined at (or re-exported through) IS the canonical path — there is nothing
to shorten.

## 5. Associated items are never `use`-importable

`Type::method(...)`, `Type::CONST`, and other associated fn/const access
through a concrete type are never legal targets of `use` — only free items
(types, traits, functions, consts, statics, modules) and enum **variants**
can be imported. Import the **type**, keep the qualified call:

```rust
use crate::local::HeapRegion;
// ...
HeapRegion::new(count)   // not a bare `new(count)` — `new` is not importable
```

An enum variant IS `use`-importable directly (`use BlockMode::Chronological;`
then bare `Chronological`); the distinguishing signal is the leaf's own
casing, not just the parent segment being a type — a variant is
UpperCamelCase like its enum, an associated fn is snake_case, an associated
const is `SCREAMING_SNAKE_CASE`.

## 6. Feature-gated imports mirror their usage's gate

When every reachable usage of an imported name sits behind the same
`#[cfg(feature = "...")]` (or `#[cfg(any(test, feature = "..."))]`,
`#[cfg(not(feature = "..."))]`) condition, the `use` statement carries the
identical `#[cfg(...)]` attribute — never an unconditional import for a
conditionally-compiled name. An import whose target crate/module is itself
feature-gated at its source (`pub use` behind `#[cfg(feature = "x")]`) is a
hard compile error under a build without that feature if left unconditional;
an import whose target exists unconditionally but is only ever _referenced_
from gated code is merely an unused-import warning there — both are bugs to
avoid, verified per-feature-combination, not just under the default/CI
feature superset.

## 7. Wildcard imports

Wildcard imports (`use foo::*;`) are banned in production code — the
workspace's `clippy::wildcard_imports` pedantic lint enforces this. `use
super::*;` inside a `#[cfg(test)] mod tests { ... }` block is the
lint's own sanctioned exception and the codebase's established test-module
idiom; it is not touched by the import-style normalization this rule
governs, which targets qualified-path-vs-import, not wildcard-vs-explicit.

## 8. What this rule does not require

Import-reshuffling a `#[cfg(test)]` module's or an integration test binary's
internal call sites is a `SHOULD`, not a `MUST` — the signature MUST in §1
is the enforced gate. A file's test module may continue to reference a
fixture/helper by its qualified path without that being a rule violation;
normalize it opportunistically when already touching the file for another
reason, not as a standing obligation.
