// Derived from binary64fast.c and random_real.c by Taylor R. Campbell,
// distributed under the following license:
//
// Copyright (c) 2014-2026 Taylor R. Campbell
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
// OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
// SUCH DAMAGE.

//! Taylor R. Campbell’s correctly rounded uniform doubles.
//!
//! A Rust port of `binary64fast.c` (Campbell, 2014–2026) and of the
//! `random_real` function of
//! [`random_real.c`](https://mumble.net/~campbell/2014/04/28/random_real.c)
//! (Campbell, 2014). These functions return an `f64` distributed as a uniform
//! real in [0 . . 1] correctly rounded to nearest. The `binary64fast.c`
//! functions ([`fast`] and the const-time variants) stop after two exponent
//! words, so they reach every float in [2⁻¹²⁸ . . 1], and 0, each with
//! probability equal to the measure of the reals that round to it; [`real`]
//! keeps consuming words, and reaches every float in [0 . . 1], including all
//! subnormals.
//!
//! Two variants handle the zero-`m` event without data-dependent branching, for
//! use where timing side channels matter; they unconditionally consume three
//! words per call. The `cmov` variant is a modification of mine in which the
//! “bit smearing“ technique of the original C is replaced by a comparison that
//! the compiler can turn into a conditional-move instruction (Taylor does not
//! want to rely on the compiler for the absence of tests, but the smearing is
//! much slower).
//!
//! If you compare generated code with older copies of `binary64fast.c` you
//! might find a signed where an unsigned integer-to-double conversion
//! instruction was present: Taylor fixed the code after I reported the
//! possibility of a branchy unsigned conversion on pre-AVX-512 x86.

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

    // Uniform odd integer in (2⁶³..2⁶⁴): round-to-odd of a uniform real in
    // [2⁶³..2⁶⁴]. The conversion rounds to nearest; ties are impossible.
    let u = u | (1 << 63) | 1;
    let s = u as f64;

    // Scale the significand into [1/2..1] and apply the geometric exponent.
    s * f / d
}

/// Port of `uniformbinary64_fast`: a correctly rounded (to nearest) uniform
/// real in [0 . . 1], branching on the 2⁻⁶⁴-probability zero-`m` event.
///
/// Consumes two 64-bit words, plus a third with probability 2⁻⁶⁴.
#[inline(always)]
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
/// event is handled branch-free, with the flag computed as `m != 0`.
///
/// This is Sebastiano Vigna's variant: it relies on the compiler turning
/// the comparison into a conditional-move–style instruction (`setne`,
/// `cset`) rather than a branch. Campbell prefers the bit-smearing variant
/// ([`consttime`]) because, for security reasons, it guarantees at the
/// source level that no test can be inserted by the compiler, whereas here
/// the absence of a branch is at the optimizer's discretion.
///
/// Always consumes three 64-bit words.
#[inline(always)]
pub fn consttime_cmove(mut bits: impl FnMut() -> u64) -> f64 {
    let (u, m, m2) = (bits(), bits(), bits());
    let t = (m != 0) as u64;
    consttime_tail(t, u, m, m2)
}

/// Port of `uniformbinary64_consttime_smear`: like [`consttime_cmove`], but
/// the flag is computed by smearing every set bit of `m` down to bit 0 —
/// branch-free at the source level, independent of the optimizer. This is
/// the variant Campbell prefers for security-sensitive uses, since no
/// compiler-inserted test can leak the (secret) value of `m` through a
/// timing side channel.
///
/// Always consumes three 64-bit words.
#[inline(always)]
pub fn consttime(mut bits: impl FnMut() -> u64) -> f64 {
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

/// 2⁻⁹⁶⁰, the exact first stage of the two-step scaling that replaces
/// the C original's `ldexp`.
const TWO_M960: f64 = f64::from_bits((1023 - 960) << 52);

/// Port of `random_real` from `random_real.c`: interprets an unbounded
/// stream of random bits as the binary expansion of a real number in
/// [0 . . 1] and rounds it to the nearest `f64`, with a sticky bit avoiding
/// ties. Every float in [0 . . 1] is reachable, including all subnormals;
/// 0 itself is returned only after 1088 zero bits, i.e., only if the
/// source is broken.
///
/// Consumes one 64-bit word, plus one more with probability 1/2 (when the
/// first word has leading zeros), plus one word per 64 leading zero bits.
/// The expected number of words per call is 1.5 + ε.
///
/// The C original scales by `ldexp(significand, exponent)`; Rust has no
/// `ldexp` in the standard library, and a single multiplication cannot
/// replace it, since 2^exponent can be as small as 2⁻¹⁰⁸⁷, which is not
/// representable. The port multiplies by 2⁻⁹⁶⁰ first — exact, since the
/// intermediate result stays normal — and then by 2^(exponent + 960),
/// which is always representable, so the result is rounded once, exactly
/// as in `ldexp`.
#[inline(always)]
pub fn real(mut bits: impl FnMut() -> u64) -> f64 {
    let mut exponent = -64i32;

    // Read zeros into the exponent until we hit a one; the rest will go
    // into the significand.
    let mut significand = bits();
    while significand == 0 {
        exponent -= 64;
        // Below -1074 = emin + 1 - p the result is guaranteed to round to
        // zero; this can happen in realistic terms only if the source is
        // broken.
        if exponent < -1074 {
            return 0.0;
        }
        significand = bits();
    }

    // If there are leading zeros, shift them into the exponent and refill
    // the less-significant bits of the significand.
    let shift = significand.leading_zeros();
    if shift != 0 {
        exponent -= shift as i32;
        significand <<= shift;
        significand |= bits() >> (64 - shift);
    }

    // Set the sticky bit, since there is almost surely another 1 in the
    // bit stream; otherwise an apparent tie might round to even when,
    // almost surely, a further 1 would break it.
    significand |= 1;

    // ldexp(significand as f64, exponent), in two multiplications: the
    // first is exact, the second rounds (to a subnormal, possibly).
    significand as f64 * TWO_M960 * f64::from_bits(((1023 + 960 + exponent) as u64) << 52)
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
            let x = consttime_cmove(|| rng.next_u64());
            assert!(x > 0.0 && x <= 1.0, "consttime_cmove: {x}");
            let x = consttime(|| rng.next_u64());
            assert!(x > 0.0 && x <= 1.0, "consttime: {x}");
        }
    }

    /// The two const-time variants must agree on every input.
    #[test]
    fn consttime_variants_agree() {
        let mut rng = Weyl(0xDEAD_BEEF);
        for _ in 0..100_000 {
            let words = [rng.next_u64(), rng.next_u64(), rng.next_u64()];
            assert_eq!(consttime_cmove(replay(&words)), consttime(replay(&words)));
        }
    }

    /// With a nonzero m, the const-time variants must agree with the
    /// branchy variant (which then ignores the third word).
    #[test]
    fn consttime_agrees_with_fast() {
        let mut rng = Weyl(0xBADC_0FFE);
        for _ in 0..100_000 {
            let words = [rng.next_u64(), rng.next_u64().max(1), rng.next_u64()];
            assert_eq!(fast(replay(&words[..2])), consttime_cmove(replay(&words)));
        }
    }

    /// The signed-arithmetic fix: with m = 0 the scale must become 2⁻¹²⁸
    /// (not go negative as with the unsigned `(t - 1)` of the original C).
    #[test]
    fn consttime_zero_mantissa_rescales() {
        let u = 0x0123_4567_89AB_CDEF;
        let m2 = 0x8000_0000_0000_0000u64; // power of two: d = 2^63
        let x = consttime_cmove(replay(&[u, 0, m2]));
        assert!(x > 0.0, "scale went negative: {x}");
        // s ≈ 2^63, f = 2^-128, d = 2^63  ⇒  x ≈ 2^-128.
        assert_eq!(x, fastdet(TWO_M128, m2, u));
        // And with m ≠ 0, m2 must be ignored.
        assert_eq!(consttime_cmove(replay(&[u, 3, m2])), fastdet(TWO_M64, 3, u));
    }

    /// Known values of `real`, pinning the two-step `ldexp` replacement
    /// (verified bit-for-bit against the compiled C original).
    #[test]
    fn real_known_values() {
        // Top bit set: one word, no refill; 2^63 | 1 rounds to 2^63.
        assert_eq!(real(replay(&[1 << 63])), 0.5);
        // All ones rounds up to 2^64: the maximum output, exactly 1.
        assert_eq!(real(replay(&[!0])), 1.0);
        // Smallest first word: 63 leading zeros shifted out and refilled.
        assert_eq!(real(replay(&[1, 0])), f64::from_bits((1023 - 64) << 52)); // 2^-64
        // 15 zero words then a 1: deep subnormal, 2^-1024.
        let mut words = [0u64; 17];
        words[15] = 1;
        assert_eq!(real(replay(&words)), f64::from_bits(1 << 50));
        // 16 zero words: the exponent falls below -1074 and the result is 0.
        assert_eq!(real(replay(&[0u64; 16])), 0.0);
    }

    #[test]
    fn real_stays_in_closed_unit_interval() {
        let mut rng = Weyl(7);
        for _ in 0..100_000 {
            let x = real(|| rng.next_u64());
            assert!(x > 0.0 && x <= 1.0, "real: {x}");
        }
    }
}
