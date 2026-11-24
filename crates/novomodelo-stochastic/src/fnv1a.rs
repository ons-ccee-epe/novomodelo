//! 64-bit FNV-1a folding shared by the crate's fingerprints.

pub(crate) const FNV1A_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV1A_PRIME: u64 = 0x0000_0100_0000_01b3;

pub(crate) fn fnv1a_fold(hash: u64, bytes: &[u8]) -> u64 {
    bytes.iter().fold(hash, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(FNV1A_PRIME)
    })
}
