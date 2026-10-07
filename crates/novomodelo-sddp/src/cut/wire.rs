//! Wire format for MPI cut exchange.
//!
//! During the SDDP backward pass, each MPI rank broadcasts its newly generated
//! cuts to all other ranks via `allgatherv`. Because the coefficient count
//! (`n_state`) is a runtime value, `allgatherv` is called with `T = u8` and
//! records are packed into a contiguous byte buffer.
//!
//! ## Wire format — cut records only
//!
//! Every record is a single cut record. Byte 0 is the format-tag byte (must
//! equal `CUT_WIRE_FORMAT_TAG`). Byte 13 is `RECORD_TAG_CUT = 0` (zeroed
//! padding; reserved for future tag dispatch).
//!
//! ### Cut record byte layout
//!
//! Total size: `29 + n_state * 8` bytes.
//!
//! ```text
//! Offset  Size  Field
//! ------  ----  -----
//!  0       1   format_tag          (u8)
//!  1- 4    4   slot_index          (u32, native-endian)
//!  5- 8    4   iteration           (u32, native-endian)
//!  9-12    4   forward_pass_index  (u32, native-endian)
//! 13       1   record_tag          (u8 = RECORD_TAG_CUT = 0; padding)
//! 14-16    3   padding             (zeroed; reserved)
//! 17-20    4   node_id             (i32, native-endian; generating node)
//! 21-28    8   intercept           (f64, native-endian)
//! 29 ...   8*n coefficients[0..n]  (f64 each, native-endian)
//! ```
//!
//! `node_id` is the declared id (`NodeGraph::node_ids`) of the node that
//! GENERATED the cut, making each record self-describing rather than
//! positionally interpreted; on a shared-leaf pool, records from different
//! generating nodes carry their distinct `node_id`s (pool = container, node =
//! generator).
//!
//! ## Format-tag guard
//!
//! Receivers reject any record whose leading tag byte does not equal
//! `CUT_WIRE_FORMAT_TAG`. The tag is a fixed format identifier, not a negotiated
//! version: every rank in one run executes the same binary, so a cross-version
//! mismatch cannot occur. The guard rejects a corrupt or misaligned `allgatherv`
//! payload before it deserializes.

use crate::SddpError;

// ---------------------------------------------------------------------------
// Wire version and record-tag constants
// ---------------------------------------------------------------------------

/// Fixed format-tag byte at offset 0 of every cut record. A deserialize landing
/// on a non-matching leading byte is rejected as a corrupt or misaligned payload,
/// never a negotiated version (every rank in one run executes the same binary).
pub const CUT_WIRE_FORMAT_TAG: u8 = 2;

/// Record-tag value at offset 13 of every cut record. Zero (padding).
pub const RECORD_TAG_CUT: u8 = 0;

// ---------------------------------------------------------------------------
// CutWireHeader
// ---------------------------------------------------------------------------

/// Parsed header from a [`cut wire record`](self).
///
/// Not `#[repr(C)]`: byte conversion is explicit in [`serialize_cut`] /
/// [`deserialize_cut`], never a transmute.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CutWireHeader {
    /// Deterministic slot index in the target [`CutPool`].
    ///
    /// [`CutPool`]: crate::cut::CutPool
    pub slot_index: u32,

    /// Declared id (`NodeGraph::node_ids`) of the node that generated this cut.
    /// Distinct per generating node even when several nodes share one pool.
    pub node_id: i32,

    /// Training iteration counter when this cut was generated.
    pub iteration: u32,

    /// Forward pass index within the iteration when this cut was generated.
    pub forward_pass_index: u32,

    /// Intercept of the Benders cut (`α` in `α + β · x`).
    pub intercept: f64,
}

/// One cut's fields for the batch (de)serialization helpers, in wire order:
/// `(slot_index, node_id, iteration, forward_pass_index, intercept,
/// coefficients)`.
pub type CutWireTuple<'a> = (u32, i32, u32, u32, f64, &'a [f64]);

// ---------------------------------------------------------------------------
// cut_wire_size
// ---------------------------------------------------------------------------

/// Return the byte size of one cut wire record with `n_state` coefficients.
///
/// ```
/// use cobre_sddp::cut::wire::cut_wire_size;
///
/// assert_eq!(cut_wire_size(0), 29);
/// assert_eq!(cut_wire_size(1), 37);
/// assert_eq!(cut_wire_size(9), 101);
/// assert_eq!(cut_wire_size(2080), 16669);
/// ```
#[inline]
#[must_use]
pub fn cut_wire_size(n_state: usize) -> usize {
    29 + n_state * 8
}

/// Serialize one cut record into `buf` starting at offset 0 (module-doc layout).
///
/// `node_id` is the generating node's declared id (`NodeGraph::node_ids`).
///
/// # Panics (debug builds only)
///
/// Panics if `buf.len() < cut_wire_size(coefficients.len())`.
pub fn serialize_cut(
    buf: &mut [u8],
    slot_index: u32,
    node_id: i32,
    iteration: u32,
    forward_pass_index: u32,
    intercept: f64,
    coefficients: &[f64],
) {
    debug_assert!(
        buf.len() >= cut_wire_size(coefficients.len()),
        "buffer too small: {} < {}",
        buf.len(),
        cut_wire_size(coefficients.len())
    );

    buf[0] = CUT_WIRE_FORMAT_TAG;
    buf[1..5].copy_from_slice(&slot_index.to_ne_bytes());
    buf[5..9].copy_from_slice(&iteration.to_ne_bytes());
    buf[9..13].copy_from_slice(&forward_pass_index.to_ne_bytes());
    buf[13] = RECORD_TAG_CUT;
    buf[14] = 0;
    buf[15] = 0;
    buf[16] = 0;
    buf[17..21].copy_from_slice(&node_id.to_ne_bytes());
    buf[21..29].copy_from_slice(&intercept.to_ne_bytes());

    for (i, &coeff) in coefficients.iter().enumerate() {
        let start = 29 + i * 8;
        buf[start..start + 8].copy_from_slice(&coeff.to_ne_bytes());
    }
}

/// Deserialize one cut record from `buf`, expecting `n_state` coefficients.
///
/// # Errors
///
/// Returns `Err(SddpError::Validation(_))` if:
/// - The leading tag byte does not equal [`CUT_WIRE_FORMAT_TAG`]. The message
///   contains `"cut_wire: unsupported version"`.
/// - The record tag at offset 13 does not equal [`RECORD_TAG_CUT`]. The
///   message contains `"cut_wire: expected cut record"`.
///
/// # Panics (debug builds only)
///
/// Panics if `buf.len() < cut_wire_size(n_state)`.
pub fn deserialize_cut(buf: &[u8], n_state: usize) -> Result<(CutWireHeader, Vec<f64>), SddpError> {
    debug_assert!(
        buf.len() >= cut_wire_size(n_state),
        "buffer too small: {} < {}",
        buf.len(),
        cut_wire_size(n_state)
    );

    let version = buf[0];
    if version != CUT_WIRE_FORMAT_TAG {
        return Err(SddpError::Validation(format!(
            "cut_wire: unsupported version {version}"
        )));
    }

    let record_tag = buf[13];
    if record_tag != RECORD_TAG_CUT {
        return Err(SddpError::Validation(format!(
            "cut_wire: expected cut record (tag {RECORD_TAG_CUT}), got tag {record_tag}"
        )));
    }

    let slot_index = u32::from_ne_bytes([buf[1], buf[2], buf[3], buf[4]]);
    let iteration = u32::from_ne_bytes([buf[5], buf[6], buf[7], buf[8]]);
    let forward_pass_index = u32::from_ne_bytes([buf[9], buf[10], buf[11], buf[12]]);
    let node_id = i32::from_ne_bytes([buf[17], buf[18], buf[19], buf[20]]);
    let intercept = f64::from_ne_bytes([
        buf[21], buf[22], buf[23], buf[24], buf[25], buf[26], buf[27], buf[28],
    ]);

    let header = CutWireHeader {
        slot_index,
        node_id,
        iteration,
        forward_pass_index,
        intercept,
    };

    let coefficients: Vec<f64> = (0..n_state)
        .map(|i| {
            let s = 29 + i * 8;
            f64::from_ne_bytes([
                buf[s],
                buf[s + 1],
                buf[s + 2],
                buf[s + 3],
                buf[s + 4],
                buf[s + 5],
                buf[s + 6],
                buf[s + 7],
            ])
        })
        .collect();

    Ok((header, coefficients))
}

/// Serialize multiple cuts into a freshly allocated contiguous byte buffer.
///
/// Each element of `cuts` is a tuple `(slot_index, node_id, iteration,
/// forward_pass_index, intercept, coefficients)`; all must have the same
/// `n_state` coefficient count.
///
/// Allocates on every call: off-hot-path only (tests, policy export). The
/// MPI hot path serializes into a pre-allocated buffer via [`CutSyncBuffers`].
///
/// # Panics (debug builds only)
///
/// Panics if any coefficient slice has length != `n_state`.
///
/// [`CutSyncBuffers`]: crate::cut::CutSyncBuffers
#[cold]
#[must_use]
pub fn serialize_cuts_to_buffer(cuts: &[CutWireTuple<'_>], n_state: usize) -> Vec<u8> {
    let record_size = cut_wire_size(n_state);
    let mut buf = vec![0u8; cuts.len() * record_size];

    for (i, &(slot_index, node_id, iteration, forward_pass_index, intercept, coefficients)) in
        cuts.iter().enumerate()
    {
        debug_assert!(
            coefficients.len() == n_state,
            "cut {i} coefficient length {} != n_state {n_state}",
            coefficients.len()
        );
        let start = i * record_size;
        serialize_cut(
            &mut buf[start..start + record_size],
            slot_index,
            node_id,
            iteration,
            forward_pass_index,
            intercept,
            coefficients,
        );
    }

    buf
}

/// Deserialize all cuts from a contiguous byte buffer, preserving record order.
///
/// # Errors
///
/// Returns `Err(SddpError::Validation(_))` if any cut record contains an
/// unrecognised version byte.
///
/// # Panics
///
/// Panics if `buf.len()` is not a multiple of `cut_wire_size(n_state)` (when
/// `n_state > 0`).
pub fn deserialize_cuts_from_buffer(
    buf: &[u8],
    n_state: usize,
) -> Result<Vec<(CutWireHeader, Vec<f64>)>, SddpError> {
    if buf.is_empty() {
        return Ok(Vec::new());
    }

    let record_size = cut_wire_size(n_state);
    assert!(
        buf.len().is_multiple_of(record_size),
        "buffer length {} is not a multiple of record size {record_size}",
        buf.len()
    );

    let n_cuts = buf.len() / record_size;
    (0..n_cuts)
        .map(|i| {
            let start = i * record_size;
            deserialize_cut(&buf[start..start + record_size], n_state)
        })
        .collect()
}

/// Deserialize all cuts from a contiguous byte buffer into caller-provided
/// pre-allocated scratch buffers.
///
/// On return, `coefficients_flat_out` holds all coefficients concatenated as a
/// flat `SoA` layout (cut `i`'s values at `[i * n_state .. (i+1) * n_state]`),
/// `headers_out` one [`CutWireHeader`] per record. Both buffers are cleared at
/// the start of each call and retain their heap allocation for reuse.
///
/// # Errors
///
/// Returns `Err(SddpError::Validation(_))` if any cut record contains an
/// unrecognised version byte. On error, the output buffers are in an
/// unspecified partial state.
///
/// # Panics
///
/// Panics if `buf.len()` is not a multiple of `cut_wire_size(n_state)` (when
/// `n_state > 0`).
pub fn deserialize_cuts_from_buffer_into(
    buf: &[u8],
    n_state: usize,
    headers_out: &mut Vec<CutWireHeader>,
    coefficients_flat_out: &mut Vec<f64>,
) -> Result<(), SddpError> {
    headers_out.clear();
    coefficients_flat_out.clear();

    if buf.is_empty() {
        return Ok(());
    }

    let record_size = cut_wire_size(n_state);
    assert!(
        buf.len().is_multiple_of(record_size),
        "buffer length {} is not a multiple of record size {record_size}",
        buf.len()
    );

    let n_cuts = buf.len() / record_size;
    headers_out.reserve(n_cuts);
    coefficients_flat_out.reserve(n_cuts * n_state);

    for i in 0..n_cuts {
        let start = i * record_size;
        let (header, coefficients) = deserialize_cut(&buf[start..start + record_size], n_state)?;
        headers_out.push(header);
        coefficients_flat_out.extend_from_slice(&coefficients);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::cast_possible_truncation, // loop indices are small constants in tests
        clippy::cast_precision_loss,      // usize cast to f64 is intentional in tests
        clippy::cast_lossless,            // i32→f64 is lossless but clippy prefers From
        clippy::unwrap_used,              // unwrap is acceptable in tests
        clippy::expect_used,              // expect is acceptable in tests
    )]

    use super::{
        CUT_WIRE_FORMAT_TAG, CutWireHeader, CutWireTuple, RECORD_TAG_CUT, cut_wire_size,
        deserialize_cut, deserialize_cuts_from_buffer, deserialize_cuts_from_buffer_into,
        serialize_cut, serialize_cuts_to_buffer,
    };
    use crate::SddpError;

    #[test]
    fn cut_wire_size_zero_state_returns_29() {
        assert_eq!(cut_wire_size(0), 29);
    }

    #[test]
    fn cut_wire_size_one_state_returns_37() {
        assert_eq!(cut_wire_size(1), 37);
    }

    #[test]
    fn cut_wire_size_three_hydro_ar2_returns_101() {
        // 3-hydro AR(2) system: n_state = 9 → 29 + 9 * 8 = 101
        assert_eq!(cut_wire_size(9), 101);
    }

    #[test]
    fn cut_wire_size_production_scale_returns_16669() {
        // Production-scale: n_state = 2080 → 29 + 2080 * 8 = 16669
        assert_eq!(cut_wire_size(2080), 16669);
    }

    #[test]
    fn round_trip_all_fields_match_exactly() {
        let n_state = 3;
        let coefficients = [1.0_f64, 2.0, 3.0];
        let mut buf = vec![0u8; cut_wire_size(n_state)];

        serialize_cut(&mut buf, 5, -9, 3, 7, 42.0, &coefficients);
        let (header, recovered) = deserialize_cut(&buf, n_state).unwrap();

        assert_eq!(header.slot_index, 5);
        assert_eq!(header.node_id, -9);
        assert_eq!(header.iteration, 3);
        assert_eq!(header.forward_pass_index, 7);
        assert_eq!(header.intercept, 42.0);
        assert_eq!(recovered, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn round_trip_verifies_bit_for_bit_coefficient_integrity() {
        // Use values that are not exactly representable in f64 to verify
        // that round-trip preserves the bit pattern exactly.
        let n_state = 4;
        let val = 1.0_f64 / 3.0;
        let coefficients = [val, -val, val * 2.0, f64::MIN_POSITIVE];
        let mut buf = vec![0u8; cut_wire_size(n_state)];

        serialize_cut(&mut buf, 1, 7, 10, 2, f64::MAX, &coefficients);
        let (header, recovered) = deserialize_cut(&buf, n_state).unwrap();

        assert_eq!(header.node_id, 7);
        assert_eq!(header.intercept.to_bits(), f64::MAX.to_bits());
        for (orig, got) in coefficients.iter().zip(&recovered) {
            assert_eq!(orig.to_bits(), got.to_bits(), "coefficient mismatch");
        }
    }

    #[test]
    fn byte_offsets_match_wire_format_spec() {
        let coefficients = [1.0_f64, 2.0, 3.0];
        let mut buf = vec![0u8; cut_wire_size(3)];

        serialize_cut(&mut buf, 5, -13, 3, 7, 42.0, &coefficients);

        assert_eq!(buf[0], CUT_WIRE_FORMAT_TAG, "version at offset 0");
        assert_eq!(
            u32::from_ne_bytes(buf[1..5].try_into().unwrap()),
            5u32,
            "slot_index at offset 1"
        );
        assert_eq!(
            u32::from_ne_bytes(buf[5..9].try_into().unwrap()),
            3u32,
            "iteration at offset 5"
        );
        assert_eq!(
            u32::from_ne_bytes(buf[9..13].try_into().unwrap()),
            7u32,
            "forward_pass_index at offset 9"
        );
        assert_eq!(buf[13], RECORD_TAG_CUT, "record tag at offset 13");
        assert_eq!(
            &buf[14..17],
            &[0u8; 3],
            "padding at offsets 14-16 must be zero"
        );
        assert_eq!(
            i32::from_ne_bytes(buf[17..21].try_into().unwrap()),
            -13i32,
            "node_id at offset 17"
        );
        assert_eq!(
            f64::from_ne_bytes(buf[21..29].try_into().unwrap()),
            42.0_f64,
            "intercept at offset 21"
        );
        assert_eq!(
            f64::from_ne_bytes(buf[29..37].try_into().unwrap()),
            1.0_f64,
            "coefficient[0] at offset 29"
        );
    }

    #[test]
    fn round_trip_production_scale_n_state_2080() {
        let n_state = 2080;
        let coefficients: Vec<f64> = (0..n_state).map(|i| i as f64 * 0.001).collect();
        let mut buf = vec![0u8; cut_wire_size(n_state)];

        serialize_cut(&mut buf, 100, 42, 50, 3, 999.0, &coefficients);
        let (header, recovered) = deserialize_cut(&buf, n_state).unwrap();

        assert_eq!(header.slot_index, 100);
        assert_eq!(header.node_id, 42);
        assert_eq!(header.iteration, 50);
        assert_eq!(header.forward_pass_index, 3);
        assert_eq!(header.intercept, 999.0);
        assert_eq!(recovered.len(), n_state);
        for (i, (orig, got)) in coefficients.iter().zip(&recovered).enumerate() {
            assert_eq!(orig.to_bits(), got.to_bits(), "mismatch at coefficient {i}");
        }
    }

    #[test]
    fn edge_case_n_state_zero_header_only_29_bytes() {
        let mut buf = vec![0u8; cut_wire_size(0)];
        assert_eq!(buf.len(), 29);

        serialize_cut(&mut buf, 1, 4, 2, 3, -1.0, &[]);
        let (header, coefficients) = deserialize_cut(&buf, 0).unwrap();

        assert_eq!(header.slot_index, 1);
        assert_eq!(header.node_id, 4);
        assert_eq!(header.iteration, 2);
        assert_eq!(header.forward_pass_index, 3);
        assert_eq!(header.intercept, -1.0);
        assert!(coefficients.is_empty());
    }

    #[test]
    fn edge_case_n_state_one_produces_37_byte_record() {
        let mut buf = vec![0u8; cut_wire_size(1)];
        assert_eq!(buf.len(), 37);

        // Use 2.5 (exactly representable in f64) as a non-PI coefficient.
        let coeff = 2.5_f64;
        serialize_cut(&mut buf, 0, 0, 0, 0, 7.0, &[coeff]);
        let (header, coefficients) = deserialize_cut(&buf, 1).unwrap();

        assert_eq!(header.intercept, 7.0);
        assert_eq!(coefficients.len(), 1);
        assert_eq!(coefficients[0].to_bits(), coeff.to_bits());
    }

    #[test]
    fn padding_bytes_at_offset_13_to_16_are_zero() {
        let mut buf = vec![0xFFu8; cut_wire_size(2)]; // Pre-fill with 0xFF
        serialize_cut(&mut buf, 1, 1, 1, 1, 1.0, &[1.0, 2.0]);
        assert_eq!(&buf[13..17], &[0u8; 4], "padding bytes must be zero");
    }

    #[test]
    fn multi_cut_five_cuts_round_trip_all_match() {
        let n_state = 3;
        let coefficients: Vec<[f64; 3]> = (0..5u32).map(|i| [f64::from(i); 3]).collect();
        let cuts: Vec<CutWireTuple<'_>> = coefficients
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let idx = i as u32;
                (
                    idx,
                    idx as i32,
                    idx * 2,
                    idx,
                    f64::from(idx) * 10.0,
                    c.as_slice(),
                )
            })
            .collect();

        let buf = serialize_cuts_to_buffer(&cuts, n_state);
        assert_eq!(buf.len(), 5 * cut_wire_size(n_state));

        let recovered = deserialize_cuts_from_buffer(&buf, n_state).unwrap();
        assert_eq!(recovered.len(), 5);

        for (i, (header, coeffs)) in recovered.iter().enumerate() {
            let idx = i as u32;
            assert_eq!(header.slot_index, idx, "slot_index mismatch at cut {i}");
            assert_eq!(header.node_id, idx as i32, "node_id mismatch at cut {i}");
            assert_eq!(header.iteration, idx * 2, "iteration mismatch at cut {i}");
            assert_eq!(
                header.forward_pass_index, idx,
                "forward_pass_index mismatch at cut {i}"
            );
            assert_eq!(
                header.intercept,
                f64::from(idx) * 10.0,
                "intercept mismatch at cut {i}"
            );
            for (j, &c) in coeffs.iter().enumerate() {
                assert_eq!(c, f64::from(idx), "coefficient[{j}] mismatch at cut {i}");
            }
        }
    }

    #[test]
    fn multi_cut_ten_cuts_round_trip_order_preserved() {
        let n_state = 2;
        let all_coefficients: Vec<Vec<f64>> = (0..10u32)
            .map(|i| vec![f64::from(i), -f64::from(i)])
            .collect();
        let cuts: Vec<CutWireTuple<'_>> = all_coefficients
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let idx = i as u32;
                (idx, -(idx as i32), 0u32, idx, f64::from(idx), c.as_slice())
            })
            .collect();

        let buf = serialize_cuts_to_buffer(&cuts, n_state);
        let recovered = deserialize_cuts_from_buffer(&buf, n_state).unwrap();

        assert_eq!(recovered.len(), 10);
        for (i, (header, coeffs)) in recovered.iter().enumerate() {
            let idx = i as u32;
            assert_eq!(header.slot_index, idx);
            assert_eq!(header.node_id, -(idx as i32));
            assert_eq!(coeffs[0].to_bits(), f64::from(idx).to_bits());
            assert_eq!(coeffs[1].to_bits(), (-f64::from(idx)).to_bits());
        }
    }

    #[test]
    fn deserialize_cuts_from_empty_buffer_returns_empty_vec() {
        let result = deserialize_cuts_from_buffer(&[], 5).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn cut_wire_header_derives_debug_clone_copy_partialeq() {
        let h = CutWireHeader {
            slot_index: 1,
            node_id: -2,
            iteration: 2,
            forward_pass_index: 3,
            intercept: 4.0,
        };
        let cloned = h;
        assert_eq!(h, cloned);
        let debug_str = format!("{h:?}");
        assert!(!debug_str.is_empty());
    }

    #[test]
    fn deserialize_cuts_from_buffer_into_populates_buffers() {
        // Serialize 3 cuts with n_state=2, then verify that
        // deserialize_cuts_from_buffer_into produces values bit-for-bit
        // identical to those from deserialize_cuts_from_buffer.
        let n_state = 2usize;
        let cuts_data: &[CutWireTuple<'_>] = &[
            (0, 0, 1, 0, 10.0, &[1.0, 2.0]),
            (1, 1, 2, 1, 20.0, &[3.0, 4.0]),
            (2, 2, 3, 2, 30.0, &[5.0, 6.0]),
        ];
        let buf = serialize_cuts_to_buffer(cuts_data, n_state);

        let mut headers_out: Vec<CutWireHeader> = Vec::new();
        let mut coefficients_flat_out: Vec<f64> = Vec::new();
        deserialize_cuts_from_buffer_into(
            &buf,
            n_state,
            &mut headers_out,
            &mut coefficients_flat_out,
        )
        .unwrap();

        assert_eq!(headers_out.len(), 3, "must produce exactly 3 headers");
        assert_eq!(
            coefficients_flat_out.len(),
            3 * n_state,
            "flat coefficient buffer must have 3 * n_state entries"
        );

        let reference = deserialize_cuts_from_buffer(&buf, n_state).unwrap();
        assert_eq!(reference.len(), 3);

        for (i, (ref_header, ref_coeffs)) in reference.iter().enumerate() {
            assert_eq!(headers_out[i], *ref_header, "header mismatch at cut {i}");
            let start = i * n_state;
            for j in 0..n_state {
                assert_eq!(
                    coefficients_flat_out[start + j].to_bits(),
                    ref_coeffs[j].to_bits(),
                    "coefficient[{j}] mismatch at cut {i}"
                );
            }
        }
    }

    #[test]
    fn deserialize_cuts_from_buffer_into_reuses_capacity() {
        // Call twice with the same buffers and verify that after the second
        // call the capacity is at least as large as after the first (proving
        // the Vec allocation is retained between calls).
        let n_state = 3usize;
        let cuts_data: &[CutWireTuple<'_>] = &[
            (0, 0, 1, 0, 1.0, &[1.0, 2.0, 3.0]),
            (1, 0, 1, 1, 2.0, &[4.0, 5.0, 6.0]),
            (2, 0, 1, 2, 3.0, &[7.0, 8.0, 9.0]),
        ];
        let buf = serialize_cuts_to_buffer(cuts_data, n_state);

        let mut headers_out: Vec<CutWireHeader> = Vec::new();
        let mut coefficients_flat_out: Vec<f64> = Vec::new();

        deserialize_cuts_from_buffer_into(
            &buf,
            n_state,
            &mut headers_out,
            &mut coefficients_flat_out,
        )
        .unwrap();
        let cap_headers_after_first = headers_out.capacity();
        let cap_coeffs_after_first = coefficients_flat_out.capacity();

        assert!(
            cap_headers_after_first >= 3,
            "headers capacity must be >= 3 after first call, got {cap_headers_after_first}"
        );

        deserialize_cuts_from_buffer_into(
            &buf,
            n_state,
            &mut headers_out,
            &mut coefficients_flat_out,
        )
        .unwrap();

        assert!(
            headers_out.capacity() >= cap_headers_after_first,
            "headers capacity must not shrink between calls"
        );
        assert!(
            coefficients_flat_out.capacity() >= cap_coeffs_after_first,
            "coefficients capacity must not shrink between calls"
        );
        assert_eq!(
            headers_out.len(),
            3,
            "second call must still produce 3 headers"
        );
    }

    // ── Version byte and record-tag tests ─────────────────────────────────────

    #[test]
    fn serialize_cut_writes_version_at_offset_zero() {
        let n_state = 3;
        let mut buf = vec![0u8; cut_wire_size(n_state)];
        serialize_cut(&mut buf, 5, 0, 3, 7, 42.0, &[1.0, 2.0, 3.0]);
        assert_eq!(
            buf[0], CUT_WIRE_FORMAT_TAG,
            "version byte at offset 0 must equal CUT_WIRE_FORMAT_TAG"
        );
        assert_eq!(
            &buf[13..17],
            &[0u8; 4],
            "padding at offset 13-16 must be zero"
        );
    }

    #[test]
    fn deserialize_cut_rejects_wrong_version_byte() {
        let n_state = 3;
        let mut buf = vec![0u8; cut_wire_size(n_state)];
        serialize_cut(&mut buf, 5, 0, 3, 7, 42.0, &[1.0, 2.0, 3.0]);

        // Overwrite the version byte with wire version 1 (the removed format).
        buf[0] = 1_u8;

        let result = deserialize_cut(&buf, n_state);
        match result {
            Err(SddpError::Validation(msg)) => {
                assert!(
                    msg.contains("unsupported version"),
                    "error message must contain 'unsupported version', got: {msg}"
                );
            }
            other => panic!("expected Err(SddpError::Validation(_)), got: {other:?}"),
        }
    }

    #[test]
    fn cut_wire_size_matches_29_plus_n_state_times_8_spec() {
        assert_eq!(cut_wire_size(0), 29);
        assert_eq!(cut_wire_size(1), 37);
        assert_eq!(cut_wire_size(9), 101);
        assert_eq!(cut_wire_size(2080), 16669);
    }

    // ── Wire-format constant + version-reject tests ──────────────────────────

    #[test]
    fn cut_wire_format_tag_value() {
        assert_eq!(CUT_WIRE_FORMAT_TAG, 2);
    }

    #[test]
    fn serialize_cut_writes_record_tag_zero_at_offset_13() {
        let n_state = 2;
        let mut buf = vec![0xFFu8; cut_wire_size(n_state)];
        serialize_cut(&mut buf, 7, 0, 1, 3, 5.0, &[1.0, 2.0]);
        assert_eq!(
            buf[13], RECORD_TAG_CUT,
            "byte 13 must equal RECORD_TAG_CUT after serialize_cut"
        );
    }

    #[test]
    fn deserialize_cut_rejects_wrong_version() {
        let n_state = 2;
        let mut buf = vec![0u8; cut_wire_size(n_state)];
        // Write a structurally valid cut record but stamp it as wire version 1 (old format).
        buf[0] = 1; // wrong wire version
        buf[1..5].copy_from_slice(&10u32.to_ne_bytes());
        buf[5..9].copy_from_slice(&1u32.to_ne_bytes());
        buf[9..13].copy_from_slice(&0u32.to_ne_bytes());
        buf[13] = RECORD_TAG_CUT;
        buf[21..29].copy_from_slice(&1.0f64.to_ne_bytes());

        let result = deserialize_cut(&buf, n_state);
        match result {
            Err(SddpError::Validation(msg)) => {
                assert!(
                    msg.contains("wire") || msg.contains("version") || msg.contains("unsupported"),
                    "error message must describe a wire version problem, got: {msg}"
                );
            }
            other => panic!("expected Err(SddpError::Validation(_)), got: {other:?}"),
        }
    }

    #[test]
    fn record_tag_cut_constant_is_zero() {
        assert_eq!(RECORD_TAG_CUT, 0u8);
    }
}
