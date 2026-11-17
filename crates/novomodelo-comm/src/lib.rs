//! # novomodelo-comm
//!
//! Pluggable communication backend abstraction for the [Novomodelo](https://github.com/ons-ccee-epe/novomodelo)
//! ecosystem.
//!
//! This crate defines the [`Communicator`] trait that decouples distributed
//! computations from specific communication technologies. Backend implementations
//! are selected by Cargo feature flags:
//!
//! - **`local`** — single-process backend, always available without any feature
//!   flag. [`LocalBackend`] is a zero-sized type with identity collective
//!   semantics and zero overhead.
//! - **`mpi`** — MPI collectives via [ferrompi](https://github.com/cobre-rs/ferrompi)
//!   (Cargo feature). Provides the [`FerrompiBackend`] and the [`CommBackend`]
//!   enum-dispatched wrapper.
//! - **`numa`** — extends `mpi` with NUMA-aware topology information from the
//!   ferrompi NUMA extension (Cargo feature; implies `mpi`).
//! - **`shared-memory`** — experimental intra-node shared-memory region API
//!   (Cargo feature, off by default). Exposes [`SharedMemoryProvider`],
//!   [`SharedRegion`], [`LocalCommunicator`], and [`HeapRegion`]. The current
//!   implementation is a **heap-backed placeholder**: each rank allocates its
//!   own [`Vec<T>`] with no memory shared across processes. The real
//!   implementation should be built on ferrompi's `SharedWindow<T>`
//!   (`MPI_Win_allocate_shared` / `MPI_Win_shared_query`); see the tracking
//!   note in `traits.rs` for open design tensions.
//!
//! Available backends are determined at build time via Cargo feature flags; the
//! active backend is chosen at runtime by the explicit [`BackendKind`] passed to
//! [`create_communicator`].
//!
//! ## Design principles
//!
//! - **Zero-cost static dispatch**: generics over [`Communicator`] eliminate
//!   dynamic dispatch overhead on the hot path.
//! - **Additive features**: multiple backends can coexist in one binary.
//! - **Orthogonal to the solver**: algorithm crates depend on this crate for
//!   communication; it does not depend back.
//!
//! ## Status
//!
//! This crate is in early development. The API **will** change.
//!
//! See the [repository](https://github.com/ons-ccee-epe/novomodelo) for the current status.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

mod factory;
mod local;
pub mod topology;
mod traits;
mod types;

#[cfg(feature = "mpi")]
mod ferrompi;

pub use factory::{BackendKind, available_backends, create_communicator};
pub use local::LocalBackend;
pub use topology::{ExecutionTopology, HostInfo, MpiRuntimeInfo, SlurmJobInfo};
pub use traits::{CommData, Communicator, TopologyProvider};
pub use types::{BackendError, CommError, ReduceOp};

#[cfg(feature = "shared-memory")]
pub use local::HeapRegion;

#[cfg(feature = "shared-memory")]
pub use traits::{LocalCommKind, LocalCommunicator, SharedMemoryProvider, SharedRegion};

#[cfg(feature = "mpi")]
pub use ferrompi::FerrompiBackend;

#[cfg(all(feature = "mpi", feature = "shared-memory"))]
pub use ferrompi::FerrompiLocalComm;

#[cfg(feature = "mpi")]
pub use factory::CommBackend;

/// Even rank partition of `total` items across `num_ranks` ranks for an
/// `allgatherv`: the first `total % num_ranks` ranks each receive
/// `⌈total / num_ranks⌉`, the rest `⌊total / num_ranks⌋`; index `r` holds rank
/// `r`'s count. The single owner of the partition rule every `allgatherv` count
/// vector in one collective must agree on.
#[must_use]
pub fn per_rank_counts(total: usize, num_ranks: usize) -> Vec<usize> {
    let base = total / num_ranks;
    let remainder = total % num_ranks;
    (0..num_ranks)
        .map(|r| base + usize::from(r < remainder))
        .collect()
}

/// Exclusive prefix-sum displacements for a per-rank `counts` vector — the
/// `allgatherv` displacement companion to [`per_rank_counts`].
#[must_use]
pub fn prefix_displs(counts: &[usize]) -> Vec<usize> {
    let mut displs = vec![0usize; counts.len()];
    for r in 1..counts.len() {
        displs[r] = displs[r - 1] + counts[r - 1];
    }
    displs
}

/// Smallest and largest `value` across all ranks, as `(min, max)`, identical on
/// every rank. Every rank must call it together.
///
/// # Errors
///
/// Returns the [`CommError`] of the underlying [`Communicator::allreduce`].
pub fn min_max_across_ranks<C: Communicator>(
    comm: &C,
    value: u64,
) -> Result<(u64, u64), CommError> {
    if comm.size() == 1 {
        return Ok((value, value));
    }
    let mut out = [0_u64; 2];
    comm.allreduce(&[!value, value], &mut out, ReduceOp::Max)?;
    Ok((!out[0], out[1]))
}

#[cfg(test)]
mod tests {
    use std::any::Any;

    use super::{CommData, CommError, Communicator, LocalBackend, ReduceOp, min_max_across_ranks};

    #[test]
    fn min_max_across_ranks_on_one_rank_returns_the_value() {
        assert_eq!(
            min_max_across_ranks(&LocalBackend, 0x1234).unwrap(),
            (0x1234, 0x1234)
        );
    }

    #[test]
    fn min_max_across_ranks_reports_a_disagreeing_peer() {
        let comm = TwoRankStub { peer: 0x10 };
        assert_eq!(min_max_across_ranks(&comm, 0x20).unwrap(), (0x10, 0x20));
    }

    struct TwoRankStub {
        peer: u64,
    }

    impl Communicator for TwoRankStub {
        fn allgatherv<T: CommData>(
            &self,
            _send: &[T],
            _recv: &mut [T],
            _counts: &[usize],
            _displs: &[usize],
        ) -> Result<(), CommError> {
            unreachable!("min_max_across_ranks does not call allgatherv")
        }

        fn allreduce<T: CommData>(
            &self,
            send: &[T],
            recv: &mut [T],
            op: ReduceOp,
        ) -> Result<(), CommError> {
            assert_eq!(op, ReduceOp::Max, "the exchange must reduce with Max");
            let peer_send = [!self.peer, self.peer];
            assert_eq!(send.len(), peer_send.len());
            assert_eq!(recv.len(), peer_send.len());
            for ((s, r), p) in send.iter().zip(recv.iter_mut()).zip(peer_send) {
                *r = upcast_u64(downcast_u64(*s).max(p));
            }
            Ok(())
        }

        fn broadcast<T: CommData>(&self, _buf: &mut [T], _root: usize) -> Result<(), CommError> {
            unreachable!("min_max_across_ranks does not call broadcast")
        }

        fn barrier(&self) -> Result<(), CommError> {
            unreachable!("min_max_across_ranks does not call barrier")
        }

        fn rank(&self) -> usize {
            unreachable!("min_max_across_ranks does not call rank")
        }

        fn size(&self) -> usize {
            2
        }

        fn abort(&self, _error_code: i32) -> ! {
            unreachable!("min_max_across_ranks does not call abort")
        }
    }

    fn downcast_u64<T: CommData>(value: T) -> u64 {
        *(&value as &dyn Any)
            .downcast_ref::<u64>()
            .expect("the stub only reduces u64 values")
    }

    fn upcast_u64<T: CommData>(value: u64) -> T {
        *(&value as &dyn Any)
            .downcast_ref::<T>()
            .expect("the stub only reduces u64 values")
    }
}
