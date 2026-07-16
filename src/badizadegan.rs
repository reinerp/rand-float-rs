// Derived from the fp-rand reference implementations (fp_rand.hpp /
// fpRand.go), at https://github.com/specbranch/fp-rand/, distributed under
// the following license:
//
// MIT License
//
// Copyright (c) 2025 Nima Badizadegan
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

//! Nima Badizadegan's “perfect” conversion.
//!
//! A Rust port of the round-down variant of
//! [Nima Badizadegan's algorithm].
//!
//! The functions in this module return a value distributed exactly as if a real
//! number had been drawn uniformly from (0 . . 1) and then rounded down (toward
//! −∞) to a representable floating-point value. Every float in [0 . . 1),
//! including every subnormal and 0 itself, is returned with probability equal
//! to the measure of the interval of reals that rounds down to it.
//!
//! ```
//! let mut src = rand_float::sources::Weyl(42);
//!
//! let x = rand_float::badizadegan::f64_down(|| src.next_u64());
//! assert!((0.0..1.0).contains(&x));
//! let y = rand_float::badizadegan::f32_down(|| src.next_u64());
//! assert!((0.0..1.0).contains(&y));
//! ```
//!
//! [Nima Badizadegan's algorithm]: https://github.com/specbranch/fp-rand/

/// Number of explicit mantissa bits of an IEEE 754 binary64.
const F64_MBITS: u32 = 52;
/// Exponent bias of an IEEE 754 binary64.
const F64_EBIAS: u32 = 1023;

/// Number of explicit mantissa bits of an IEEE 754 binary32.
const F32_MBITS: u32 = 23;
/// Exponent bias of an IEEE 754 binary32.
const F32_EBIAS: u32 = 127;

/// A buffer of random bits drawn from a 64-bit entropy source.
///
/// [`get_bits`] hands out `n` bits at a time, drawing a new 64-bit word
/// from the source only when the buffered bits run out; the remainder of
/// each draw is kept for subsequent requests. A pool lives for a single
/// top-level generation call.
///
/// [`get_bits`]: Self::get_bits
struct EntropyPool<F> {
    src: F,
    pool: u64,
    nbits: u32,
}

impl<F: FnMut() -> u64> EntropyPool<F> {
    #[inline]
    fn new(mut src: F) -> Self {
        // Draw the first word eagerly: every conversion starts by requesting
        // more bits than an empty pool holds, so this draw is never wasted,
        // and it makes the refill in [`get_bits`](Self::get_bits) a genuinely
        // rare path, as its cold hint promises.
        let pool = src();
        Self {
            src,
            pool,
            nbits: 64,
        }
    }

    /// Returns `n` (< 64) uniform random bits in the low bits of the result.
    #[inline]
    fn get_bits(&mut self, n: u32) -> u64 {
        debug_assert!(n < 64);
        let mut result = self.pool;

        if self.nbits < n {
            // Kluge; see [`crate::cold::cold_barrier`].
            crate::cold::cold_barrier();
            let needed = n - self.nbits;
            self.pool = (self.src)();
            result |= self.pool << self.nbits;
            self.pool >>= needed;
            self.nbits = 64 - needed;
        } else {
            self.nbits -= n;
            self.pool >>= n;
        }

        result & ((1u64 << n) - 1)
    }
}

/// Stage 1 for `f64`: locates the result’s binade and produces the partial
/// result.
///
/// Returns the IEEE 754 representation of the partial result and the number
/// of vacant trailing significand bits that stage 2 must backfill.
#[inline]
fn seek64<F: FnMut() -> u64>(pool: &mut EntropyPool<F>) -> (u64, u32) {
    let mut a = pool.get_bits(F64_MBITS);
    let mut base = 1.0f64.to_bits();
    let mut nb = 0;

    // Zoom down, one 52-binade window at a time, while the mantissa is zero.
    while a == 0 {
        if base < ((F64_MBITS as u64) << F64_MBITS) {
            // Bottom window, reaching the subnormals: only EBIAS mod MBITS
            // bits remain, drawn aligned to the top of the mantissa field.
            const B: u32 = F64_EBIAS % F64_MBITS;
            nb = F64_MBITS - B;
            a = pool.get_bits(B) << nb;
            break;
        }

        a = pool.get_bits(F64_MBITS);
        base -= (F64_MBITS as u64) << F64_MBITS;
    }

    // Add the bits to the base as an integer, subtract the base in floating
    // point: the difference is the (renormalized) partial result, and the
    // exponent drop tells how many trailing bits were left vacant.
    a += base;
    let b = f64::from_bits(a) - f64::from_bits(base);
    nb += (base >> F64_MBITS) as u32 - (b.to_bits() >> F64_MBITS) as u32;
    (b.to_bits(), nb)
}

/// Stage 1 for `f32`; see [`seek64`].
#[inline]
fn seek32<F: FnMut() -> u64>(pool: &mut EntropyPool<F>) -> (u32, u32) {
    let mut a = pool.get_bits(F32_MBITS) as u32;
    let mut base = 1.0f32.to_bits();
    let mut nb = 0;

    while a == 0 {
        if base < (F32_MBITS << F32_MBITS) {
            const B: u32 = F32_EBIAS % F32_MBITS;
            nb = F32_MBITS - B;
            a = (pool.get_bits(B) as u32) << nb;
            break;
        }

        a = pool.get_bits(F32_MBITS) as u32;
        base -= F32_MBITS << F32_MBITS;
    }

    a += base;
    let b = f32::from_bits(a) - f32::from_bits(base);
    nb += (base >> F32_MBITS) - (b.to_bits() >> F32_MBITS);
    (b.to_bits(), nb)
}

/// Returns a random `f64` distributed as a uniform real in (0 . . 1) rounded
/// **down** (toward −∞) to the nearest representable value.
///
/// The result lies in [0 . . 1); every `f64` in that range, including every
/// subnormal and 0, is returned with probability equal to the measure of
/// the reals that round down to it. See the [module documentation] for
/// the algorithm.
///
/// `bits` must return independent uniform random 64-bit words. Exactly one
/// word is consumed with probability ≈ 1 − 2⁻¹²; the expected number of
/// words per call is 1 + 2⁻¹² + O(2⁻⁵²).
///
/// To call it repeatedly on the same source, pass the source by mutable
/// reference:
///
/// ```
/// // A Weyl sequence: statistically very poor, for illustration only.
/// let mut src = rand_float::sources::Weyl(1);
/// let mut next = || src.next_u64();
/// let x = rand_float::badizadegan::f64_down(&mut next);
/// let y = rand_float::badizadegan::f64_down(&mut next);
/// assert!(x != y);
/// ```
///
/// [module documentation]: self
#[inline]
pub fn f64_down(bits: impl FnMut() -> u64) -> f64 {
    let mut pool = EntropyPool::new(bits);
    let (partial, nb) = seek64(&mut pool);
    // Round down: backfill the vacant trailing bits with random bits. The
    // low nb bits of `partial` are zero, so the addition never carries.
    f64::from_bits(pool.get_bits(nb) + partial)
}

/// Returns a random `f32` distributed as a uniform real in (0 . . 1) rounded
/// **down** (toward −∞) to the nearest representable value.
///
/// The `f32` counterpart of [`f64_down`]; the result lies in [0 . . 1) and
/// every `f32` in that range is reachable, including subnormals and 0.
#[inline]
pub fn f32_down(bits: impl FnMut() -> u64) -> f32 {
    let mut pool = EntropyPool::new(bits);
    let (partial, nb) = seek32(&mut pool);
    f32::from_bits(pool.get_bits(nb) as u32 + partial)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sources::Weyl;

    /// A source that replays a fixed sequence of words, then panics.
    fn replay(words: &[u64]) -> impl FnMut() -> u64 + '_ {
        let mut iter = words.iter();
        move || *iter.next().expect("source exhausted")
    }

    #[test]
    fn test_known_values_f64() {
        // Mantissa 2^51 -> partial 0.5, one vacant bit, backfilled with the
        // leftover bit 52 of the same word.
        assert_eq!(f64_down(replay(&[1 << 51])), 0.5);
        assert_eq!(
            f64_down(replay(&[(1 << 51) | (1 << 52)])),
            f64::from_bits(0.5f64.to_bits() + 1)
        );
        // Full mantissa: partial is 1 - 2^-52, i.e. the largest f64 below 1,
        // with one vacant bit... which is bit 52 of the word.
        assert_eq!(
            f64_down(replay(&[(1 << 52) - 1])),
            f64::from_bits(1.0f64.to_bits() - 2)
        );
        assert_eq!(
            f64_down(replay(&[(1 << 53) - 1])),
            f64::from_bits(1.0f64.to_bits() - 1)
        );
        // Mantissa 1: partial 2^-52, 52 vacant bits backfilled from the 12
        // leftover bits plus 40 bits of a second word.
        assert_eq!(
            f64_down(replay(&[1, 0])),
            f64::from_bits(0x3CB0000000000000)
        );
    }

    #[test]
    fn test_all_zeros_terminates_and_yields_zero() {
        // A broken all-zero source must walk down all exponent windows and
        // come out with exactly 0.0 (and must not loop forever).
        let mut n_calls = 0u32;
        let zero_src = || {
            n_calls += 1;
            assert!(n_calls < 64, "zoom loop does not terminate");
            0u64
        };
        assert_eq!(f64_down(zero_src), 0.0);
    }

    #[test]
    fn test_all_zeros_terminates_and_yields_zero_f32() {
        assert_eq!(f32_down(replay(&[0, 0, 0])), 0.0);
    }

    #[test]
    fn test_all_ones_source() {
        let x = f64_down(replay(&[!0u64]));
        assert_eq!(x, f64::from_bits(1.0f64.to_bits() - 1));
    }

    #[test]
    fn test_range_and_moments_f64() {
        let mut src = Weyl(0xDEADBEEF);
        let n = 1_000_000;
        let mut sum = 0.0;
        let mut top_binade = 0u32;
        for _ in 0..n {
            let x = f64_down(|| src.next_u64());
            assert!((0.0..1.0).contains(&x), "out of range: {x}");
            sum += x;
            if x >= 0.5 {
                top_binade += 1;
            }
        }
        let mean = sum / n as f64;
        // Standard error of the mean is ~2.9e-4; 5 sigma.
        assert!((mean - 0.5).abs() < 1.5e-3, "mean {mean}");
        // P(x >= 1/2) = 1/2; 5 sigma is ~2.5e-3.
        let frac = top_binade as f64 / n as f64;
        assert!((frac - 0.5).abs() < 2.5e-3, "top-binade fraction {frac}");
    }

    #[test]
    fn test_range_and_moments_f32() {
        let mut src = Weyl(0xC0FFEE);
        let n = 1_000_000;
        let mut sum = 0.0f64;
        for _ in 0..n {
            let x = f32_down(|| src.next_u64());
            assert!((0.0..1.0).contains(&x), "out of range: {x}");
            sum += x as f64;
        }
        let mean = sum / n as f64;
        assert!((mean - 0.5).abs() < 1.5e-3, "mean {mean}");
    }

    #[test]
    fn test_low_binades_are_reachable_and_correctly_distributed() {
        // With 10^6 samples, P(x < 2^-10) should be ~2^-10 (~977 hits).
        let mut src = Weyl(42);
        let n = 1_000_000;
        let threshold = 1.0 / 1024.0;
        let mut hits = 0u32;
        for _ in 0..n {
            if f64_down(|| src.next_u64()) < threshold {
                hits += 1;
            }
        }
        // Binomial(10^6, 2^-10): mean ~977, sigma ~31; allow 5 sigma.
        assert!((820..1140).contains(&hits), "hits {hits}");
    }

    #[test]
    fn test_entropy_pool_bit_accounting() {
        // 64 bits requested 8 at a time must reproduce the word exactly.
        let word = 0x0123_4567_89AB_CDEFu64;
        let words = [word];
        let mut pool = EntropyPool::new(replay(&words));
        for i in 0..8 {
            assert_eq!(pool.get_bits(8), (word >> (8 * i)) & 0xFF);
        }
        // Zero-width requests are free.
        assert_eq!(pool.get_bits(0), 0);
    }
}
