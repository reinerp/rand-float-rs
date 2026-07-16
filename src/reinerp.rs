//! Reiner Pope's hardware-int-to-float variants.
//!
//! These use hardware u64-to-f64 conversion instructions (cvtsi2sd/ucvtf)
//! to give faster fast paths. The hardware conversion instructions do the
//! work of both count-leading-zeros and bit packing all in one, thus saving
//! some instructions.
//!
//! In the slow path, any algorithm will suffice; as implemented here it
//! falls back to pekkizen.
//!
//! This concept was introduced in
//! <https://github.com/smol-rs/fastrand/pull/129#issuecomment-4827111778>
//! and commit
//! <https://github.com/reinerp/fastrand/commit/bdee1e766d5ab103992af719bf6de358e70624ef>.
//! Explanation of the approach is in
//! <https://github.com/smol-rs/fastrand/pull/129#issuecomment-4827802889>.
//! As implemented here, we fall back to a slow path using integer arithmetic
//! rather than floating point arithmetic, to simplify the slow path.
//! Performance on the slow path is irrelevant, as it runs in <0.1% of cases.

use crate::cold::cold_barrier;

/// Interprets an unbounded stream of random bits as the binary expansion
/// of a real number in [0 . . 1) and rounded down to the largest representable
/// f64 below it. Its output is identical to [`crate::pekkizen::f64_full`].
#[inline(always)]
pub fn f64_round_down(mut bits: impl FnMut() -> u64) -> f64 {
    let mut u = bits();
    if IS_AVX512 {
        // On x86 with AVX512F, we can directly ask for the rounding mode we want.
        if u & MASK52 != 0 {
            // >99.975% of cases
            return avx512f_cvt_round_down(u) * POW64;
        }
    } else if U64_TO_F64_IS_SLOW {
        // On other platforms, we emulate rounding down with arithmetic.
        //
        // Keep the top 53 bits starting from the leading one.
        let u1 = u >> 1;
        let z1 = u1.leading_zeros() as u64;
        if z1 <= 11 {
            // >99.95% of cases
            return (u1 & (MASK_TOP53 >> z1)) as i64 as f64 * POW63;
        }
    } else {
        let z = u.leading_zeros() as u64;
        if z <= 11 {
            // >99.975% of cases
            return (u & (MASK_TOP53 >> z)) as f64 * POW64;
        }
    }

    cold_barrier();

    // Slow path, for <0.05% of cases. Any algorithm will do. We use
    // the pekkizen slow path.
    let mut exp = 0u64;
    let mut z = u.leading_zeros() as u64;
    while u == 0 {
        u = bits();
        z = u.leading_zeros() as u64;
        exp += 64;
        if exp + z >= 1074 {
            return 0.0;
        }
    }
    // The Go original computes `bits() >> (64 - z)` with z possibly 0 after
    // the loop, which Go defines as 0; `checked_shr` + `unwrap_or` mirrors
    // that.
    let u = u << z | bits().checked_shr(64 - z as u32).unwrap_or(0);
    exp += z;
    if exp < 1022 {
        return f64::from_bits((1022 - exp) << 52 | (u << 1) >> 12);
    }
    // The 2⁵² subnormal floats.
    f64::from_bits(u >> (exp - 1022) >> 12)
}

#[inline]
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
fn avx512f_cvt_round_down(u: u64) -> f64 {
    use core::arch::x86_64::*;

    // SAFETY: this function is only compiled when AVX-512F is enabled. SSE2
    // is part of the x86-64 baseline, so all required target features are
    // available.
    unsafe {
        _mm_cvtsd_f64(_mm_cvt_roundu64_sd(
            _mm_setzero_pd(),
            u,
            _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC,
        ))
    }
}

#[inline]
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
fn avx512f_cvt_round_down(_: u64) -> f64 {
    unreachable!("AVX512-only path")
}

const IS_X86: bool = cfg!(any(target_arch = "x86", target_arch = "x86_64"));
const IS_AVX512: bool = cfg!(all(target_arch = "x86_64", target_feature = "avx512f"));
const U64_TO_F64_IS_SLOW: bool = IS_X86 && !IS_AVX512;

const POW64: f64 = const { 1.0 / (1u128 << 64) as f64 };
const POW63: f64 = const { 1.0 / (1u128 << 63) as f64 };
const MASK52: u64 = (1u64 << 52).wrapping_neg();
const MASK_TOP53: u64 = (1u64 << 11).wrapping_neg();

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pekkizen;

    fn replay(words: &[u64]) -> impl FnMut() -> u64 + '_ {
        let mut iter = words.iter();
        move || *iter.next().expect("source exhausted")
    }

    fn shifted_pattern(pattern: u64, shift: usize) -> Vec<u64> {
        let whole_words = shift / 64;
        let intra_word = shift % 64;
        let mut words = vec![0; whole_words];
        if intra_word == 0 {
            words.push(pattern);
        } else {
            words.push(pattern >> intra_word);
            words.push(pattern << (64 - intra_word));
        }
        // Reference implementations may fetch a lookahead word to fill the
        // significand even when all its useful bits are already available.
        words.extend([0; 2]);
        words
    }

    #[test]
    fn test_dynamic_range_against_reference_implementations() {
        let mut patterns = vec![
            0x8000_0000_0000_0000, // 1.000...000
            0x8000_0000_0000_0001, // 1.000...001
            0xffff_ffff_ffff_ffff, // 1.111...111
            0xc000_0000_0000_0000, // 1.100...000
            0xaaaa_aaaa_aaaa_aaaa,
            0xdead_beef_cafe_babe,
        ];
        // Exercise every possible position of a second set bit in the
        // normalized significand, including the rounding and sticky bits.
        patterns.extend((0..63).map(|bit| 1 << 63 | 1 << bit));

        for pattern in patterns {
            for shift in 0..=1140 {
                let words = shifted_pattern(pattern, shift);

                let down = f64_round_down(replay(&words));
                let expected_down = pekkizen::f64_full(replay(&words));
                assert_eq!(
                    down.to_bits(),
                    expected_down.to_bits(),
                    "round down: pattern={pattern:#018x}, shift={shift}"
                );
            }
        }
    }
}
