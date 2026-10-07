//! Factory function for creating the active communication backend.
//!
//! [`create_communicator`] is the single runtime entry point for constructing a
//! [`Communicator`](crate::Communicator). The caller passes a [`BackendKind`].
//! No configuration environment variable participates in the choice, but
//! [`BackendKind::Auto`] probes for an MPI launcher — a runtime fact set by
//! `mpiexec`/`mpirun`/`srun`, not a configuration channel — and selects the MPI
//! backend when one is detected.

use crate::BackendError;
#[cfg(not(feature = "mpi"))]
use crate::BackendError::BackendNotAvailable;
#[cfg(feature = "mpi")]
use crate::CommData;
#[cfg(feature = "mpi")]
use crate::CommError;
#[cfg(feature = "mpi")]
use crate::ExecutionTopology;
#[cfg(feature = "mpi")]
use crate::FerrompiBackend;
#[cfg(all(feature = "mpi", feature = "shared-memory"))]
use crate::HeapRegion;
use crate::LocalBackend;
#[cfg(all(feature = "mpi", feature = "shared-memory"))]
use crate::LocalCommKind;
#[cfg(feature = "mpi")]
use crate::ReduceOp;
/// Backend selector passed by the caller to [`create_communicator`].
///
/// [`BackendKind::Mpi`] and [`BackendKind::Local`] force a backend;
/// [`BackendKind::Auto`] detects one from the launch environment.
///
/// # Examples
///
/// ```rust
/// use cobre_comm::BackendKind;
///
/// let kind = BackendKind::Local;
/// let copy = kind;
/// assert_eq!(copy, kind);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    /// Detect the backend from the launch environment: the MPI backend when an
    /// MPI launcher is detected, otherwise the local backend. Also the label
    /// used when reconstructing an unrecognized persisted backend string.
    Auto,
    /// MPI backend; fails at runtime if the `mpi` feature is not compiled in.
    Mpi,
    /// Single-process local backend; always available.
    Local,
}

/// Enum-dispatched communicator wrapping any available concrete backend.
///
/// Enum dispatch (not `Box<dyn>`) because [`crate::Communicator`] carries
/// generic methods that make it non-object-safe; dispatch overhead is negligible
/// against the MPI collective or LP solve it wraps. `CommBackend: Send + Sync`
/// because all inner backends are. No-feature builds use [`crate::LocalBackend`]
/// directly.
#[cfg(feature = "mpi")]
pub enum CommBackend {
    /// MPI backend powered by ferrompi.
    Mpi(Box<FerrompiBackend>),

    /// Single-process local backend (always-available fallback).
    Local(LocalBackend),
}

#[cfg(feature = "mpi")]
// Rationale: never called; its monomorphisation is the compile-time check that
// `CommBackend: Send + Sync`, catching any future field that breaks it.
#[allow(dead_code)]
const fn _assert_comm_backend_send_sync() {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<CommBackend>();
}

#[cfg(feature = "mpi")]
impl crate::Communicator for CommBackend {
    fn allgatherv<T: CommData>(
        &self,
        send: &[T],
        recv: &mut [T],
        counts: &[usize],
        displs: &[usize],
    ) -> Result<(), CommError> {
        match self {
            Self::Mpi(backend) => backend.allgatherv(send, recv, counts, displs),
            Self::Local(backend) => backend.allgatherv(send, recv, counts, displs),
        }
    }

    fn allreduce<T: CommData>(
        &self,
        send: &[T],
        recv: &mut [T],
        op: ReduceOp,
    ) -> Result<(), CommError> {
        match self {
            Self::Mpi(backend) => backend.allreduce(send, recv, op),
            Self::Local(backend) => backend.allreduce(send, recv, op),
        }
    }

    fn broadcast<T: CommData>(&self, buf: &mut [T], root: usize) -> Result<(), CommError> {
        match self {
            Self::Mpi(backend) => backend.broadcast(buf, root),
            Self::Local(backend) => backend.broadcast(buf, root),
        }
    }

    fn barrier(&self) -> Result<(), CommError> {
        match self {
            Self::Mpi(backend) => backend.barrier(),
            Self::Local(backend) => backend.barrier(),
        }
    }

    fn rank(&self) -> usize {
        match self {
            Self::Mpi(backend) => backend.rank(),
            Self::Local(backend) => backend.rank(),
        }
    }

    fn size(&self) -> usize {
        match self {
            Self::Mpi(backend) => backend.size(),
            Self::Local(backend) => backend.size(),
        }
    }

    fn abort(&self, error_code: i32) -> ! {
        match self {
            Self::Mpi(backend) => backend.abort(error_code),
            Self::Local(backend) => backend.abort(error_code),
        }
    }
}

#[cfg(all(feature = "mpi", feature = "shared-memory"))]
impl crate::SharedMemoryProvider for CommBackend {
    /// `HeapRegion<T>` directly (not an enum wrapper): both inner backends
    /// already unify on it as their `Region<T>`.
    type Region<T: CommData> = HeapRegion<T>;

    fn create_shared_region<T: CommData>(
        &self,
        count: usize,
    ) -> Result<Self::Region<T>, CommError> {
        match self {
            Self::Mpi(backend) => backend.create_shared_region(count),
            Self::Local(backend) => backend.create_shared_region(count),
        }
    }

    fn split_local(&self) -> Result<LocalCommKind, CommError> {
        match self {
            Self::Mpi(backend) => backend.split_local(),
            Self::Local(backend) => backend.split_local(),
        }
    }

    fn is_leader(&self) -> bool {
        match self {
            Self::Mpi(backend) => backend.is_leader(),
            Self::Local(backend) => backend.is_leader(),
        }
    }
}

#[cfg(feature = "mpi")]
impl crate::TopologyProvider for CommBackend {
    fn topology(&self) -> &ExecutionTopology {
        match self {
            Self::Mpi(backend) => backend.topology(),
            Self::Local(backend) => backend.topology(),
        }
    }
}

/// Returns all backend names compiled into this binary.
///
/// Always includes `"local"`. Conditionally includes `"mpi"` (feature `mpi`).
///
/// # Examples
///
/// ```rust
/// use cobre_comm::available_backends;
///
/// let backends = available_backends();
/// assert!(backends.contains(&"local".to_string()));
/// ```
#[must_use]
#[allow(clippy::vec_init_then_push)] // cfg-gated push pattern
pub fn available_backends() -> Vec<String> {
    let mut backends = Vec::new();
    #[cfg(feature = "mpi")]
    backends.push("mpi".to_string());
    backends.push("local".to_string());
    backends
}

/// Returns `true` if any MPI launcher environment variable is present (checked
/// via `var_os`, so non-UTF-8 values still count).
///
/// Always compiled (no cfg gate) so it is testable in no-feature builds, where
/// it is unused outside tests — hence the dead-code allow.
#[cfg_attr(not(feature = "mpi"), allow(dead_code))]
fn mpi_launch_detected() -> bool {
    const MPI_ENV_VARS: [&str; 6] = [
        "PMI_RANK",
        "PMI_SIZE",
        "OMPI_COMM_WORLD_RANK",
        "OMPI_COMM_WORLD_SIZE",
        "MPI_LOCALRANKID",
        "SLURM_PROCID",
    ];
    MPI_ENV_VARS
        .iter()
        .any(|var| std::env::var_os(var).is_some())
}

/// So a run started under `mpiexec`/`mpirun`/`srun` distributes without an
/// explicit `--comm-backend mpi`.
#[cfg(feature = "mpi")]
fn auto_detect() -> Result<CommBackend, BackendError> {
    if mpi_launch_detected() {
        return Ok(CommBackend::Mpi(Box::new(FerrompiBackend::new()?)));
    }
    Ok(CommBackend::Local(LocalBackend))
}

/// Construct the active communication backend (no-feature build).
///
/// # Errors
///
/// - [`crate::BackendError::BackendNotAvailable`]: a known backend was requested
///   but not compiled into this binary.
///
/// # Examples
///
/// ```rust
/// # #[cfg(not(feature = "mpi"))]
/// # {
/// use cobre_comm::{create_communicator, BackendKind};
///
/// // With no distributed features, only the local backend is constructible.
/// let backend = create_communicator(BackendKind::Local).expect("local backend must succeed");
/// # use cobre_comm::Communicator;
/// assert_eq!(backend.rank(), 0);
/// assert_eq!(backend.size(), 1);
/// # }
/// ```
#[cfg(not(feature = "mpi"))]
pub fn create_communicator(kind: BackendKind) -> Result<LocalBackend, BackendError> {
    match kind {
        BackendKind::Local | BackendKind::Auto => Ok(LocalBackend),
        BackendKind::Mpi => Err(BackendNotAvailable {
            requested: "mpi".to_string(),
            available: available_backends(),
        }),
    }
}

/// Construct the active communication backend (MPI build).
///
/// # Errors
///
/// - [`crate::BackendError::InitializationFailed`]: the selected backend failed
///   to initialize (propagated from [`crate::FerrompiBackend::new`]).
///
/// # Examples
///
/// ```rust
/// # #[cfg(feature = "mpi")]
/// # {
/// use cobre_comm::{create_communicator, BackendKind, Communicator};
///
/// // The local backend is selected explicitly.
/// let backend = create_communicator(BackendKind::Local).expect("local backend must succeed");
/// assert_eq!(backend.rank(), 0);
/// # }
/// ```
#[cfg(feature = "mpi")]
pub fn create_communicator(kind: BackendKind) -> Result<CommBackend, BackendError> {
    match kind {
        BackendKind::Mpi => Ok(CommBackend::Mpi(Box::new(FerrompiBackend::new()?))),
        BackendKind::Local => Ok(CommBackend::Local(LocalBackend)),
        BackendKind::Auto => auto_detect(),
    }
}

#[cfg(test)]
mod tests {
    use super::BackendKind;

    use super::available_backends;

    /// `available_backends()` always contains `"local"` regardless of features.
    #[test]
    fn test_available_backends_contains_local() {
        let backends = available_backends();
        assert!(
            backends.contains(&"local".to_string()),
            "expected 'local' in {backends:?}"
        );
    }

    /// In a no-feature build `available_backends()` returns exactly `["local"]`.
    #[test]
    #[cfg(not(feature = "mpi"))]
    fn test_available_backends_no_feature_exact() {
        assert_eq!(available_backends(), vec!["local".to_string()]);
    }

    /// No-feature build: [`BackendKind::Local`] → `Ok(LocalBackend)` with
    /// rank 0 and size 1.
    #[test]
    #[cfg(not(feature = "mpi"))]
    fn test_create_communicator_no_feature_local() {
        use crate::Communicator;

        let backend = super::create_communicator(BackendKind::Local)
            .expect("LocalBackend construction must succeed");
        assert_eq!(backend.rank(), 0);
        assert_eq!(backend.size(), 1);
    }

    /// No-feature build: [`BackendKind::Auto`] resolves to the local backend
    /// (auto-detect cannot select MPI when none is compiled in).
    #[test]
    #[cfg(not(feature = "mpi"))]
    fn test_create_communicator_no_feature_auto_is_local() {
        use crate::Communicator;

        let backend = super::create_communicator(BackendKind::Auto)
            .expect("auto resolves to local with no MPI feature");
        assert_eq!(backend.rank(), 0);
        assert_eq!(backend.size(), 1);
    }

    /// MPI build: [`BackendKind::Auto`] resolves to the local backend when no
    /// MPI launcher is present. `cargo test` does not run under `mpiexec`, so
    /// no launcher variable is set; the guard skips the case where one is.
    #[test]
    #[cfg(feature = "mpi")]
    fn test_create_communicator_mpi_auto_local_without_launcher() {
        use crate::Communicator;

        if super::mpi_launch_detected() {
            return;
        }
        let backend = super::create_communicator(BackendKind::Auto)
            .expect("auto resolves to local without a launcher");
        assert_eq!(backend.rank(), 0);
        assert_eq!(backend.size(), 1);
    }

    /// No-feature build: [`BackendKind::Mpi`] → `Err(BackendNotAvailable)`
    /// where `requested == "mpi"` and `available` contains `"local"`.
    #[test]
    #[cfg(not(feature = "mpi"))]
    fn test_create_communicator_no_feature_unavailable() {
        use crate::BackendError::BackendNotAvailable;

        let err = super::create_communicator(BackendKind::Mpi)
            .expect_err("unavailable backend must return Err");
        assert!(
            matches!(err, BackendNotAvailable { .. }),
            "expected BackendNotAvailable, got {err:?}"
        );
        if let BackendNotAvailable {
            ref requested,
            ref available,
        } = err
        {
            assert_eq!(requested, "mpi");
            assert!(
                available.contains(&"local".to_string()),
                "available should contain 'local', got {available:?}"
            );
        }
    }

    /// Verify that `BackendKind` derives `Debug`, `Clone`, `Copy`, `PartialEq`, `Eq`.
    #[test]
    fn test_backend_kind_derives() {
        let kind = BackendKind::Auto;

        let s = format!("{kind:?}");
        assert!(s.contains("Auto"), "Debug output should contain 'Auto'");

        // Rationale: explicit Clone::clone on a Copy type proves Clone is derived.
        #[allow(clippy::clone_on_copy)]
        let cloned = kind.clone();
        assert_eq!(cloned, kind);

        let copied = kind;
        assert_eq!(copied, kind);

        assert_eq!(BackendKind::Mpi, BackendKind::Mpi);
        assert_ne!(BackendKind::Mpi, BackendKind::Local);
        assert_eq!(BackendKind::Local, BackendKind::Local);
    }

    #[cfg(feature = "mpi")]
    #[allow(clippy::float_cmp)]
    mod comm_backend {
        use super::super::CommBackend;
        use crate::{Communicator, LocalBackend, ReduceOp};

        #[cfg(feature = "shared-memory")]
        use crate::{LocalCommunicator, SharedMemoryProvider, SharedRegion};

        /// Compile-time assertion that `CommBackend: Send + Sync`.
        #[test]
        fn test_comm_backend_send_sync() {
            fn assert_send_sync<T: Send + Sync>() {}
            assert_send_sync::<CommBackend>();
        }

        #[test]
        fn test_comm_backend_local_rank_size() {
            let backend = CommBackend::Local(LocalBackend);
            assert_eq!(backend.rank(), 0);
            assert_eq!(backend.size(), 1);
        }

        #[test]
        fn test_comm_backend_local_barrier() {
            let backend = CommBackend::Local(LocalBackend);
            assert!(backend.barrier().is_ok());
        }

        #[test]
        fn test_comm_backend_local_allreduce() {
            let backend = CommBackend::Local(LocalBackend);
            let send = [1.0_f64, 2.0, 3.0];
            let mut recv = [0.0_f64; 3];
            backend.allreduce(&send, &mut recv, ReduceOp::Sum).unwrap();
            assert_eq!(recv, [1.0, 2.0, 3.0]);
        }

        #[test]
        fn test_comm_backend_local_allgatherv() {
            let backend = CommBackend::Local(LocalBackend);
            let send = [7.0_f64, 8.0, 9.0];
            let mut recv = [0.0_f64; 3];
            backend.allgatherv(&send, &mut recv, &[3], &[0]).unwrap();
            assert_eq!(recv, [7.0, 8.0, 9.0]);
        }

        #[test]
        fn test_comm_backend_local_broadcast() {
            let backend = CommBackend::Local(LocalBackend);
            let mut buf = [1.0_f64, 2.0];
            assert!(backend.broadcast(&mut buf, 0).is_ok());
            assert_eq!(buf, [1.0, 2.0]);
        }

        /// Covers `create_shared_region`, `split_local`, `is_leader`.
        #[cfg(feature = "shared-memory")]
        #[test]
        fn test_comm_backend_local_shared_memory() {
            let backend = CommBackend::Local(LocalBackend);

            let mut region = backend.create_shared_region::<f64>(10).unwrap();
            assert_eq!(region.as_slice().len(), 10);
            region.as_mut_slice().fill(42.0);
            assert_eq!(region.as_slice(), &[42.0_f64; 10]);
            assert!(region.fence().is_ok());

            let local_comm = backend.split_local().unwrap();
            assert_eq!(local_comm.rank(), 0);
            assert_eq!(local_comm.size(), 1);
            assert!(local_comm.barrier().is_ok());

            assert!(backend.is_leader());
        }
    }
}
