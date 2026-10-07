//! Version and build-environment reporting for the top-level `cobre` module.
//!
//! [`version_info`] assembles the same fields the `cobre version` CLI prints.

use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Return a dict describing the running Cobre build.
///
/// The returned dict has the following keys:
///
/// * `"version"` — the `cobre-python` package version (`cobre.__version__`).
/// * `"solver"` — the active LP backend and its version, e.g. `"HiGHS 1.7.2"`
///   or `"CLP <v>"`.
/// * `"comm"` — always `"local"`; this crate is single-process and never
///   initializes MPI.
/// * `"zstd"` — `"enabled"` (compression support is always compiled in).
/// * `"arch"` — the target architecture and OS, e.g. `"x86_64-linux"`.
/// * `"build"` — `"debug"` or `"release"` depending on the build profile.
///
/// # Examples
///
/// ```python
/// import cobre
/// info = cobre.version_info()
/// assert info["version"] == cobre.__version__
/// assert info["comm"] == "local"
/// assert info["solver"].split()[0] in ("HiGHS", "CLP")
/// ```
#[pyfunction]
pub fn version_info(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item("version", env!("CARGO_PKG_VERSION"))?;
    dict.set_item(
        "solver",
        format!(
            "{} {}",
            cobre_solver::active_solver_name(),
            cobre_solver::active_solver_version()
        ),
    )?;
    dict.set_item("comm", "local")?;
    dict.set_item("zstd", "enabled")?;
    dict.set_item(
        "arch",
        format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS),
    )?;
    dict.set_item(
        "build",
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
    )?;
    Ok(dict.into())
}
