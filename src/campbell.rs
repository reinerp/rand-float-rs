//! Taylor R. Campbell’s correctly rounded uniform doubles.
//!
//! A Rust port of `binary64fast.c` (Campbell, 2014–2026). These functions
//! return an `f64` distributed as a uniform real in [0, 1] **correctly
//! rounded to nearest**, up to events of probability 2⁻¹²⁸: every float in
//! [2⁻¹²⁸, 1], and 0, can occur, each with probability equal to the measure
//! of the reals that round to it.
//!
//! The construction draws one word `u` for the significand and one word `m`
//! for the exponent: the number of trailing zeros of `m` is geometrically
//! distributed and picks the binade, while `u`, forced odd, is a
//! round-to-odd representative of a uniform real in [2⁶³, 2⁶⁴] that the
//! final integer→float conversion correctly rounds (ties are impossible on
//! an odd integer). If `m` is zero — probability 2⁻⁶⁴ — the scale drops by
//! another 2⁻⁶⁴ and a third word is drawn.
//!
//! Two variants handle the zero-`m` event without data-dependent branching,
//! for use where timing side channels matter; they unconditionally consume
//! three words per call. In both, the `(t - 1) * 0x1p-64` rescaling of the
//! original C is computed in *signed* arithmetic,
//! `((double)(int64_t)t - 1) * 0x1p-64`, so that t = 0 yields −1 rather
//! than 2⁶⁴ − 1, and so that the integer→double conversion is signed
//! (avoiding the branchy unsigned conversion sequence on pre-AVX-512 x86).

/// 2⁻⁶⁴ (Rust has no hex float literals; built via the exponent field).
const TWO_M64: f64 = f64::from_bits((1023 - 64) << 52);
/// 2³², used by the x86-only split unsigned→double conversion.
#[cfg(target_arch = "x86_64")]
const TWO_P32: f64 = 4294967296.0;

/// Port of Campbell’s `uniformbinary64_fastdet`, the deterministic core of
/// this module: turns an exponent scale `f` ∈ {2⁻⁶⁴, 2⁻¹²⁸}, a geometric
/// word `m` and a significand word `u` into a correctly rounded (to
/// nearest) uniform binary64.
#[inline(always)]
pub fn fastdet(f: f64, m: u64, u: u64) -> f64 {
    // Largest power-of-two divisor of m, with bit 63 forced as a backstop
    // against a broken all-zero source; exactly representable, so the
    // conversion below is exact.
    let m = m | (1 << 63);
    let m = m & m.wrapping_neg();
    // On x86_64 there is no unsigned 64-bit → double instruction until
    // AVX-512; split into halves so the compiler emits branch-free signed
    // conversions, as in the C original.
    #[cfg(target_arch = "x86_64")]
    let d = ((m >> 32) as f64) * TWO_P32 + ((m & 0xFFFF_FFFF) as f64);
    #[cfg(not(target_arch = "x86_64"))]
    let d = m as f64;

    // Uniform odd integer in (2⁶³, 2⁶⁴): round-to-odd of a uniform real in
    // [2⁶³, 2⁶⁴]. The conversion rounds to nearest; ties are impossible.
    let u = u | (1 << 63) | 1;
    let s = u as f64;

    // Scale the significand into [1/2, 1] and apply the geometric exponent.
    s * f / d
}

/// Port of `uniformbinary64_fast`: a correctly rounded (to nearest) uniform
/// real in [0, 1], branching on the 2⁻⁶⁴-probability zero-`m` event.
///
/// Consumes two 64-bit words, plus a third with probability 2⁻⁶⁴.
#[inline]
pub fn fast(mut bits: impl FnMut() -> u64) -> f64 {
    let u = bits();
    let mut f = TWO_M64;
    let mut m = bits();
    if m == 0 {
        // unlikely
        f *= TWO_M64;
        m = bits();
    }
    fastdet(f, m, u)
}

/// Shared tail of the const-time variants: given the flag t ∈ {0, 1}
/// (t = 1 iff m ≠ 0), rescale `f` and substitute `m2` for a zero `m`,
/// both branch-free. See the [module documentation](self) for the
/// signed-arithmetic form of the rescaling.
#[inline(always)]
fn consttime_tail(t: u64, u: u64, m: u64, m2: u64) -> f64 {
    let mut f = TWO_M64;
    let tf = t as i64 as f64;
    f *= tf - (tf - 1.0) * TWO_M64;
    let m = m | (m2 & t.wrapping_sub(1));
    fastdet(f, m, u)
}

/// Port of `uniformbinary64_consttime_if`: like [`fast`], but the zero-`m`
/// event is handled branch-free, with the flag computed as `m != 0`
/// (which compiles to a branch-free flag-set).
///
/// Always consumes three 64-bit words.
#[inline]
pub fn consttime_if(mut bits: impl FnMut() -> u64) -> f64 {
    let (u, m, m2) = (bits(), bits(), bits());
    let t = (m != 0) as u64;
    consttime_tail(t, u, m, m2)
}

/// Port of `uniformbinary64_consttime_smear`: like [`consttime_if`], but
/// the flag is computed by smearing every set bit of `m` down to bit 0 —
/// branch-free at the source level, independent of the optimizer.
///
/// Always consumes three 64-bit words.
#[inline]
pub fn consttime_smear(mut bits: impl FnMut() -> u64) -> f64 {
    let (u, m, m2) = (bits(), bits(), bits());
    let mut t = m;
    t |= t >> 1;
    t |= t >> 2;
    t |= t >> 4;
    t |= t >> 8;
    t |= t >> 16;
    t |= t >> 32;
    t &= 1;
    consttime_tail(t, u, m, m2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sources::Weyl;

    /// 2⁻¹²⁸.
    const TWO_M128: f64 = f64::from_bits((1023 - 128) << 52);

    /// A source that replays a fixed sequence of words, then panics.
    fn replay(words: &[u64]) -> impl FnMut() -> u64 + '_ {
        let mut iter = words.iter();
        move || *iter.next().expect("source exhausted")
    }

    #[test]
    fn stays_in_closed_unit_interval() {
        let mut rng = Weyl(42);
        for _ in 0..100_000 {
            let x = fast(|| rng.next_u64());
            assert!(x > 0.0 && x <= 1.0, "fast: {x}");
            let x = consttime_if(|| rng.next_u64());
            assert!(x > 0.0 && x <= 1.0, "consttime_if: {x}");
            let x = consttime_smear(|| rng.next_u64());
            assert!(x > 0.0 && x <= 1.0, "consttime_smear: {x}");
        }
    }

    /// The two const-time variants must agree on every input.
    #[test]
    fn consttime_variants_agree() {
        let mut rng = Weyl(0xDEAD_BEEF);
        for _ in 0..100_000 {
            let words = [rng.next_u64(), rng.next_u64(), rng.next_u64()];
            assert_eq!(
                consttime_if(replay(&words)),
                consttime_smear(replay(&words))
            );
        }
    }

    /// With a nonzero m, the const-time variants must agree with the
    /// branchy variant (which then ignores the third word).
    #[test]
    fn consttime_agrees_with_fast() {
        let mut rng = Weyl(0xBADC_0FFE);
        for _ in 0..100_000 {
            let words = [rng.next_u64(), rng.next_u64().max(1), rng.next_u64()];
            assert_eq!(fast(replay(&words[..2])), consttime_if(replay(&words)));
        }
    }

    /// The signed-arithmetic fix: with m = 0 the scale must become 2⁻¹²⁸
    /// (not go negative as with the unsigned `(t - 1)` of the original C).
    #[test]
    fn consttime_zero_mantissa_rescales() {
        let u = 0x0123_4567_89AB_CDEF;
        let m2 = 0x8000_0000_0000_0000u64; // power of two: d = 2^63
        let x = consttime_if(replay(&[u, 0, m2]));
        assert!(x > 0.0, "scale went negative: {x}");
        // s ≈ 2^63, f = 2^-128, d = 2^63  ⇒  x ≈ 2^-128.
        assert_eq!(x, fastdet(TWO_M128, m2, u));
        // And with m ≠ 0, m2 must be ignored.
        assert_eq!(consttime_if(replay(&[u, 3, m2])), fastdet(TWO_M64, 3, u));
    }
}
