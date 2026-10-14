//! Canonical simplex basis status shared across LP backends.
//!
//! `HiGHS` and CLP number the same logical basis states differently
//! (`crate::ffi::highs` vs `crate::ffi::clp`); [`BasisStatus`] is the
//! backend-neutral pivot the `to_*_code`/`from_*_code` pairs convert through.
//! A `HIGHS_BASIS_STATUS_*` value and a CLP per-element status code are never
//! interchangeable even when they share a numeral.

use crate::ffi::clp::{
    CLP_BASIS_AT_LOWER, CLP_BASIS_AT_UPPER, CLP_BASIS_BASIC, CLP_BASIS_FIXED, CLP_BASIS_FREE,
    CLP_BASIS_SUPERBASIC,
};
use crate::ffi::highs::{
    HIGHS_BASIS_STATUS_BASIC, HIGHS_BASIS_STATUS_LOWER, HIGHS_BASIS_STATUS_NONBASIC,
    HIGHS_BASIS_STATUS_UPPER, HIGHS_BASIS_STATUS_ZERO,
};

/// Canonical simplex basis status covering every basis state either backend
/// can produce.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BasisStatus {
    /// At the lower bound.
    Lower,
    /// In the basis.
    Basic,
    /// At the upper bound.
    Upper,
    /// Free and at value zero (`HiGHS` `kZero`).
    Zero,
    /// General nonbasic state for a bounded variable at neither bound (`HiGHS` `kNonbasic`).
    Nonbasic,
    /// Nonbasic and at neither bound (CLP `superBasic`).
    Superbasic,
    /// Nonbasic at a fixed bound, where the lower and upper bounds coincide (CLP `isFixed`).
    Fixed,
}

impl BasisStatus {
    /// Maps to the native `HIGHS_BASIS_STATUS_*` code; `Superbasic` and
    /// `Fixed` fold onto `Nonbasic`'s and `Lower`'s codes respectively, since
    /// `HiGHS` has no distinct representation for either.
    #[must_use]
    pub fn to_highs_code(self) -> i32 {
        match self {
            Self::Lower | Self::Fixed => HIGHS_BASIS_STATUS_LOWER,
            Self::Basic => HIGHS_BASIS_STATUS_BASIC,
            Self::Upper => HIGHS_BASIS_STATUS_UPPER,
            Self::Zero => HIGHS_BASIS_STATUS_ZERO,
            Self::Nonbasic | Self::Superbasic => HIGHS_BASIS_STATUS_NONBASIC,
        }
    }

    /// Maps a `HiGHS` native basis-status code to the canonical status,
    /// falling back to [`Self::Nonbasic`] for any code outside `0..=4`.
    #[must_use]
    pub fn from_highs_code(code: i32) -> Self {
        match code {
            HIGHS_BASIS_STATUS_LOWER => Self::Lower,
            HIGHS_BASIS_STATUS_BASIC => Self::Basic,
            HIGHS_BASIS_STATUS_UPPER => Self::Upper,
            HIGHS_BASIS_STATUS_ZERO => Self::Zero,
            _ => Self::Nonbasic,
        }
    }

    /// Maps to the native `CLP_BASIS_*` code (`ffi::clp`); `Nonbasic` folds
    /// onto the `CLP_BASIS_SUPERBASIC` code, since CLP has no distinct
    /// representation for it.
    #[must_use]
    pub fn to_clp_code(self) -> i32 {
        match self {
            Self::Zero => CLP_BASIS_FREE,
            Self::Basic => CLP_BASIS_BASIC,
            Self::Upper => CLP_BASIS_AT_UPPER,
            Self::Lower => CLP_BASIS_AT_LOWER,
            Self::Superbasic | Self::Nonbasic => CLP_BASIS_SUPERBASIC,
            Self::Fixed => CLP_BASIS_FIXED,
        }
    }

    /// Maps a CLP native per-element basis-status code to the canonical
    /// status, falling back to [`Self::Nonbasic`] for any code outside the
    /// `CLP_BASIS_*` range.
    #[must_use]
    pub fn from_clp_code(code: i32) -> Self {
        match code {
            CLP_BASIS_FREE => Self::Zero,
            CLP_BASIS_BASIC => Self::Basic,
            CLP_BASIS_AT_UPPER => Self::Upper,
            CLP_BASIS_AT_LOWER => Self::Lower,
            CLP_BASIS_SUPERBASIC => Self::Superbasic,
            CLP_BASIS_FIXED => Self::Fixed,
            _ => Self::Nonbasic,
        }
    }

    /// Maps to the injective canonical discriminant (`0..=6`), the one code
    /// space that distinguishes all seven variants losslessly — unlike
    /// [`Self::to_highs_code`]/[`Self::to_clp_code`], which fold the variants a
    /// backend cannot represent. Use this for a durable cross-backend encoding
    /// where a CLP-only `Superbasic`/`Fixed` must survive a round-trip; the
    /// explicit arms make the code space independent of enum declaration order.
    #[must_use]
    pub fn to_discriminant_code(self) -> u8 {
        match self {
            Self::Lower => 0,
            Self::Basic => 1,
            Self::Upper => 2,
            Self::Zero => 3,
            Self::Nonbasic => 4,
            Self::Superbasic => 5,
            Self::Fixed => 6,
        }
    }

    /// Inverse of [`Self::to_discriminant_code`]; any code outside `0..=6`
    /// falls back to [`Self::Nonbasic`] (non-panicking decode).
    #[must_use]
    pub fn from_discriminant_code(code: u8) -> Self {
        match code {
            0 => Self::Lower,
            1 => Self::Basic,
            2 => Self::Upper,
            3 => Self::Zero,
            5 => Self::Superbasic,
            6 => Self::Fixed,
            _ => Self::Nonbasic,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::BasisStatus;
    use crate::ffi::clp::CLP_BASIS_SUPERBASIC;
    use crate::ffi::highs::{
        HIGHS_BASIS_STATUS_BASIC, HIGHS_BASIS_STATUS_LOWER, HIGHS_BASIS_STATUS_NONBASIC,
        HIGHS_BASIS_STATUS_UPPER, HIGHS_BASIS_STATUS_ZERO,
    };

    const HIGHS_REPRESENTABLE: [BasisStatus; 5] = [
        BasisStatus::Lower,
        BasisStatus::Basic,
        BasisStatus::Upper,
        BasisStatus::Zero,
        BasisStatus::Nonbasic,
    ];

    const CLP_REPRESENTABLE: [BasisStatus; 6] = [
        BasisStatus::Zero,
        BasisStatus::Basic,
        BasisStatus::Upper,
        BasisStatus::Lower,
        BasisStatus::Superbasic,
        BasisStatus::Fixed,
    ];

    const ALL_VARIANTS: [BasisStatus; 7] = [
        BasisStatus::Lower,
        BasisStatus::Basic,
        BasisStatus::Upper,
        BasisStatus::Zero,
        BasisStatus::Nonbasic,
        BasisStatus::Superbasic,
        BasisStatus::Fixed,
    ];

    #[test]
    fn highs_native_code_round_trips_for_every_valid_code() {
        for code in 0..=4 {
            assert_eq!(
                BasisStatus::from_highs_code(code).to_highs_code(),
                code,
                "HiGHS code {code} did not round-trip"
            );
        }
    }

    #[test]
    fn clp_native_code_round_trips_for_every_valid_code() {
        for code in 0..=5 {
            assert_eq!(
                BasisStatus::from_clp_code(code).to_clp_code(),
                code,
                "CLP code {code} did not round-trip"
            );
        }
    }

    #[test]
    fn highs_canonical_status_round_trips_for_every_representable_variant() {
        for status in HIGHS_REPRESENTABLE {
            assert_eq!(
                BasisStatus::from_highs_code(status.to_highs_code()),
                status,
                "{status:?} did not round-trip through the HiGHS code space"
            );
        }
    }

    #[test]
    fn clp_canonical_status_round_trips_for_every_representable_variant() {
        for status in CLP_REPRESENTABLE {
            assert_eq!(
                BasisStatus::from_clp_code(status.to_clp_code()),
                status,
                "{status:?} did not round-trip through the CLP code space"
            );
        }
    }

    #[test]
    fn to_highs_code_matches_named_constants() {
        assert_eq!(BasisStatus::Lower.to_highs_code(), HIGHS_BASIS_STATUS_LOWER);
        assert_eq!(BasisStatus::Basic.to_highs_code(), HIGHS_BASIS_STATUS_BASIC);
        assert_eq!(BasisStatus::Upper.to_highs_code(), HIGHS_BASIS_STATUS_UPPER);
        assert_eq!(BasisStatus::Zero.to_highs_code(), HIGHS_BASIS_STATUS_ZERO);
        assert_eq!(
            BasisStatus::Nonbasic.to_highs_code(),
            HIGHS_BASIS_STATUS_NONBASIC
        );
    }

    #[test]
    fn to_highs_code_folds_clp_only_variants() {
        assert_eq!(
            BasisStatus::Superbasic.to_highs_code(),
            HIGHS_BASIS_STATUS_NONBASIC
        );
        assert_eq!(BasisStatus::Fixed.to_highs_code(), HIGHS_BASIS_STATUS_LOWER);
    }

    #[test]
    fn to_clp_code_folds_highs_only_variant() {
        assert_eq!(BasisStatus::Nonbasic.to_clp_code(), CLP_BASIS_SUPERBASIC);
    }

    #[test]
    fn from_highs_code_out_of_range_falls_back_without_panicking() {
        assert_eq!(BasisStatus::from_highs_code(99), BasisStatus::Nonbasic);
    }

    #[test]
    fn from_clp_code_out_of_range_falls_back_without_panicking() {
        assert_eq!(BasisStatus::from_clp_code(99), BasisStatus::Nonbasic);
    }

    #[test]
    fn discriminant_code_round_trips_for_every_variant() {
        for status in ALL_VARIANTS {
            assert_eq!(
                BasisStatus::from_discriminant_code(status.to_discriminant_code()),
                status,
                "{status:?} did not round-trip through the discriminant code space"
            );
        }
    }

    #[test]
    fn discriminant_code_is_injective() {
        let codes: Vec<u8> = ALL_VARIANTS
            .iter()
            .map(|s| s.to_discriminant_code())
            .collect();
        let mut sorted = codes.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            sorted.len(),
            ALL_VARIANTS.len(),
            "discriminant codes must be distinct across all variants: {codes:?}"
        );
    }

    #[test]
    fn from_discriminant_code_out_of_range_falls_back_without_panicking() {
        assert_eq!(
            BasisStatus::from_discriminant_code(99),
            BasisStatus::Nonbasic
        );
    }

    #[test]
    fn discriminant_and_highs_decode_agree_over_pre_existing_checkpoint_range() {
        for code in 0..=4u8 {
            assert_eq!(
                BasisStatus::from_discriminant_code(code),
                BasisStatus::from_highs_code(i32::from(code)),
                "canonical and HiGHS decodes must agree over the legacy 0..=4 range \
                 a pre-existing checkpoint carries"
            );
        }
    }
}
