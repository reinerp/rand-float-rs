// Derived from the uniFloats wiki of https://github.com/pekkizen/prng
// (function Float64_64), distributed under the following license:
//
// MIT License
//
// Copyright (c) 2020 pekkizen <pekkizen@gmail.com>
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

//! pekkizen’s leading-zeros technique.
//!
//! A Rust port of `Float64_64` from
//! [pekkizen’s uniFloats wiki](https://github.com/pekkizen/prng/wiki/uniFloats)
//! (explicit bit-building form). One 64-bit word is interpreted as a
//! uniform 64-bit fixed-point real in [0 . . 1): the count of leading zeros
//! picks the binade (a geometric distribution), and the remaining bits are
//! shifted into the mantissa.
//!
//! Every float in [2⁻¹² . . 1) is reachable, and below 2⁻¹² the technique
//! returns the 2⁵² multiples of 2⁻⁶⁴, that is, the value is a uniform real
//! rounded down to the 2⁻⁶⁴ grid. It consumes exactly one word per call and
//! costs only a couple of operations more than [`standard`](crate::standard)
//! scaling, delivering about 2¹¹ times as many distinct values.

/// Returns a random `f64` distributed as a uniform 64-bit fixed-point real
/// in [0 . . 1) rounded down to the nearest representable value: every float
/// in [2⁻¹² . . 1), and the 2⁵² multiples of 2⁻⁶⁴ below 2⁻¹².
///
/// The Go original computes `u << z` with z possibly 64, which Go defines
/// as 0; Rust declares 64-bit shifts overflow, so the shift is split as
/// `(u << (z - 1)) << 1` (z ≥ 1 always).
#[inline]
pub fn f64_64(mut bits: impl FnMut() -> u64) -> f64 {
    let u = bits();
    if u == 0 {
        return 0.0;
    }
    let z = u.leading_zeros() as u64 + 1;
    f64::from_bits((1023 - z) << 52 | ((u << (z - 1)) << 1) >> 12)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sources::Weyl;

    /// The bit-building form must agree with the wiki’s division form,
    /// `float64(u << z >> 11) / 2^53 / 2^z` with z = leadingZeros(u).
    #[test]
    fn matches_reference_division_form() {
        let division_form = |u: u64| {
            let z = u.leading_zeros();
            let m = ((u << z) >> 11) as f64;
            m / (1u64 << 53) as f64 / (1u64 << z) as f64
        };
        let mut rng = Weyl(7);
        for _ in 0..100_000 {
            let u = rng.0.wrapping_add(0x9E3779B97F4A7C15);
            assert_eq!(f64_64(|| rng.next_u64()), division_form(u));
        }
    }

    #[test]
    fn range() {
        let mut rng = Weyl(42);
        for _ in 0..100_000 {
            let x = f64_64(|| rng.next_u64());
            assert!((0.0..1.0).contains(&x), "f64_64: {x}");
        }
    }

    /// Extremes of the leading-zeros count.
    #[test]
    fn edge_cases() {
        assert_eq!(f64_64(|| 1 << 63), 0.5);
        assert_eq!(f64_64(|| u64::MAX), f64::from_bits(1.0f64.to_bits() - 1));
        assert_eq!(f64_64(|| 1), f64::from_bits((1023 - 64) << 52)); // 2^-64
        assert_eq!(f64_64(|| 0), 0.0);
    }
}
