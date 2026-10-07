//! # cobre-python
//!
//! Python bindings for the [Cobre](https://github.com/cobre-rs/cobre) power systems solver.
//!
//! ## Constraints
//!
//! - **Single-process only** — this crate MUST NOT initialize MPI or depend
//!   on `ferrompi`. The GIL/MPI incompatibility makes it unsafe to combine
//!   MPI initialization with Python embedding. For distributed execution,
//!   launch `mpiexec cobre` as a subprocess.
//! - **GIL released during computation** — all Rust computation runs with
//!   the GIL released via `py.detach()`, allowing worker threads
//!   within `cobre-sddp` to run at full parallelism.
//! - **No Python callbacks in the hot loop** — all customization is
//!   via configuration structs, not Python callables.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

use pyo3::prelude::*;

mod convert;
mod errors;
mod io;
mod model;
mod policy;
mod results;
mod run;
mod schema;
mod study;
mod version;

#[pymodule]
#[pyo3(name = "model")]
fn model_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add(
        "__doc__",
        "Data model types for the Cobre power systems solver.",
    )?;
    m.add_class::<model::PySystem>()?;
    m.add_class::<model::PyBus>()?;
    m.add_class::<model::PyLine>()?;
    m.add_class::<model::PyThermal>()?;
    m.add_class::<model::PyHydro>()?;
    m.add_class::<model::PyEnergyContract>()?;
    m.add_class::<model::PyPumpingStation>()?;
    m.add_class::<model::PyNonControllableSource>()?;
    Ok(())
}

#[pymodule]
#[pyo3(name = "io")]
fn io_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__doc__", "I/O helpers for loading Cobre case directories.")?;
    m.add_function(wrap_pyfunction!(io::load_case, m)?)?;
    m.add_function(wrap_pyfunction!(io::validate, m)?)?;
    Ok(())
}

#[pymodule]
#[pyo3(name = "run")]
fn run_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add(
        "__doc__",
        "Solver execution entry points for training and simulation.",
    )?;
    m.add_function(wrap_pyfunction!(run::run, m)?)?;
    Ok(())
}

#[pymodule]
#[pyo3(name = "results")]
fn results_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add(
        "__doc__",
        "Result loading and inspection functions for Cobre output artifacts.",
    )?;
    m.add_function(wrap_pyfunction!(results::load_results, m)?)?;
    m.add_function(wrap_pyfunction!(results::load_convergence, m)?)?;
    m.add_function(wrap_pyfunction!(results::load_convergence_arrow, m)?)?;
    m.add_function(wrap_pyfunction!(results::load_simulation, m)?)?;
    m.add_function(wrap_pyfunction!(results::load_simulation_arrow, m)?)?;
    m.add_function(wrap_pyfunction!(results::load_policy, m)?)?;
    m.add_function(wrap_pyfunction!(results::load_stochastic, m)?)?;
    m.add_class::<results::Stochastic>()?;
    Ok(())
}

#[pymodule]
#[pyo3(name = "errors")]
fn errors_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    errors::register_errors(m)
}

#[pymodule]
#[pyo3(name = "schema")]
fn schema_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add(
        "__doc__",
        "JSON Schema export helpers for Cobre case directory input types.",
    )?;
    m.add_function(wrap_pyfunction!(schema::export, m)?)?;
    Ok(())
}

/// Register a submodule in `sys.modules` so `import cobre.foo` works: `PyO3`'s
/// `add_submodule` only attaches it as an attribute, leaving `import` to fail.
fn register_submodule<'py>(
    parent: &Bound<'py, PyModule>,
    child: &Bound<'py, PyModule>,
    parent_package: &str,
) -> PyResult<()> {
    let child_name = child.name()?;
    let full_name = format!("{parent_package}.{child_name}");
    parent
        .py()
        .import("sys")?
        .getattr("modules")?
        .set_item(full_name, child)?;
    parent.add_submodule(child)
}

/// The compiled extension module, imported privately as `cobre._native`.
/// The public `cobre` package (pure-Python `__init__.py`) re-exports everything
/// from here under the documented `cobre.*` names.
#[pymodule]
#[pyo3(name = "_native")]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    register_submodule(
        m,
        pyo3::wrap_pymodule!(model_module)(py).bind(py),
        "cobre._native",
    )?;
    register_submodule(
        m,
        pyo3::wrap_pymodule!(io_module)(py).bind(py),
        "cobre._native",
    )?;
    register_submodule(
        m,
        pyo3::wrap_pymodule!(run_module)(py).bind(py),
        "cobre._native",
    )?;
    register_submodule(
        m,
        pyo3::wrap_pymodule!(results_module)(py).bind(py),
        "cobre._native",
    )?;
    register_submodule(
        m,
        pyo3::wrap_pymodule!(errors_module)(py).bind(py),
        "cobre._native",
    )?;
    register_submodule(
        m,
        pyo3::wrap_pymodule!(schema_module)(py).bind(py),
        "cobre._native",
    )?;

    m.add_function(wrap_pyfunction!(version::version_info, m)?)?;
    m.add_function(wrap_pyfunction!(policy::write_policy_checkpoint, m)?)?;

    // Top-level classes (`cobre.Study`, `cobre.Policy`), not submodule members.
    m.add_class::<study::Study>()?;
    m.add_class::<study::Policy>()?;

    Ok(())
}
