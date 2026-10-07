//! Integration tests for the `create_communicator()` factory and public APIs.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

/// Gated `not(feature = "mpi")`: there `create_communicator()` returns
/// `Result<LocalBackend, _>`, not the `CommBackend` of the `mpi` build.
#[cfg(not(feature = "mpi"))]
mod no_feature_factory {
    use cobre_comm::{BackendError, BackendKind, Communicator, create_communicator};

    #[test]
    fn test_factory_no_feature_local() {
        let backend = create_communicator(BackendKind::Local).expect("must succeed");
        assert_eq!(backend.rank(), 0);
        assert_eq!(backend.size(), 1);
    }

    #[test]
    fn test_factory_no_feature_auto_is_local() {
        let backend = create_communicator(BackendKind::Auto).expect("auto -> local");
        assert_eq!(backend.rank(), 0);
        assert_eq!(backend.size(), 1);
    }

    #[test]
    fn test_factory_no_feature_mpi_unavailable() {
        let err = create_communicator(BackendKind::Mpi).expect_err("must fail");
        assert!(
            matches!(
                err,
                BackendError::BackendNotAvailable {
                    ref requested,
                    ..
                } if requested == "mpi"
            ),
            "got {err:?}"
        );
        if let BackendError::BackendNotAvailable { ref available, .. } = err {
            assert!(available.contains(&"local".to_string()));
        }
    }
}

// ── any-feature factory tests ─────────────────────────────────────────────────

#[cfg(feature = "mpi")]
mod any_feature_factory {
    use cobre_comm::{BackendKind, CommBackend, Communicator, create_communicator};

    #[test]
    fn test_factory_any_feature_local() {
        let backend = create_communicator(BackendKind::Local).expect("must succeed");
        assert!(matches!(backend, CommBackend::Local(_)));
        assert_eq!(backend.rank(), 0);
    }

    /// `BackendKind::Auto` resolves to a concrete backend from the launch
    /// environment — local when `cargo test` runs outside an MPI launcher.
    #[test]
    fn test_factory_any_feature_auto_resolves() {
        let comm = create_communicator(BackendKind::Auto).expect("auto resolves");
        assert_eq!(comm.rank(), 0);
    }
}

// ── available_backends() tests ────────────────────────────────────────────────

mod available_backends_tests {
    use cobre_comm::available_backends;

    #[test]
    fn test_available_backends_contains_local() {
        let backends = available_backends();
        assert!(backends.contains(&"local".to_string()));
    }

    #[test]
    #[cfg(feature = "mpi")]
    fn test_available_backends_mpi_feature() {
        let backends = available_backends();
        assert!(backends.contains(&"mpi".to_string()));
    }
}

// ── compile-time checks ───────────────────────────────────────────────────────

mod compile_time_checks {
    #[test]
    #[cfg(feature = "mpi")]
    fn test_ferrompi_backend_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<cobre_comm::FerrompiBackend>();
    }

    #[test]
    #[cfg(feature = "mpi")]
    fn test_ferrompi_backend_communicator() {
        fn assert_communicator<T: cobre_comm::Communicator>() {}
        assert_communicator::<cobre_comm::FerrompiBackend>();
    }

    #[test]
    #[cfg(all(feature = "mpi", feature = "shared-memory"))]
    fn test_ferrompi_backend_shared_memory_provider() {
        fn assert_shared_memory_provider<T: cobre_comm::SharedMemoryProvider>() {}
        assert_shared_memory_provider::<cobre_comm::FerrompiBackend>();
    }
}

// ── error type checks ─────────────────────────────────────────────────────────

mod error_type_checks {
    use cobre_comm::{BackendError, CommError};

    #[test]
    fn test_comm_error_std_error_send_sync() {
        fn assert_error_send_sync<T: std::error::Error + Send + Sync>() {}
        assert_error_send_sync::<CommError>();
    }

    #[test]
    fn test_backend_error_std_error_send_sync() {
        fn assert_error_send_sync<T: std::error::Error + Send + Sync>() {}
        assert_error_send_sync::<BackendError>();
    }
}
