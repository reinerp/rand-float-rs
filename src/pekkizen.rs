// Derived from the uniFloats wiki https://github.com/pekkizen/prng (functions
// Float64_64, Float64_117 and Float64full), distributed under the following
// license:
//
// Copyright (c) 2020, Pekka Pulkkinen
//
// Copying and distribution of this article, ideas and source code are permitted
// worldwide, without royalty, in any medium, provided the copyright notice is
// preserved and a reference to this article is given.

//! Pekka Pulkkinen’s leading-zeros technique.
//!
//! A Rust port of `Float64_64`, `Float64_117` and `Float64full` from
//! [Pekka Pulkkinen’s uniFloats wiki]. One 64-bit word is
//! interpreted as a uniform fixed-point real in [0 . . 1): the count of leading
//! zeros picks the binade (a geometric distribution), and the remaining bits
//! are shifted into the mantissa.
//!
//! The three variants differ in how far down the fixed point extends:
//!
//! - [`f64_64`] stops at the first word: a uniform real rounded down to the
//!   2⁻⁶⁴ grid, complete (every float reachable) in [2⁻¹² . . 1);
//! - [`f64_117`] draws a second word when the first has 12 or more leading
//!   zeros (probability 2⁻¹²), extending completeness to [2⁻⁶⁵ . . 1)
//!   with a 2⁻¹¹⁷ grid below;
//! - [`f64_full`] additionally keeps drawing words while they are zero,
//!   reaching every float in [0 . . 1), subnormals included.
//!
//! All variants consume one word per call with probability 1 − 2⁻¹² and, in
//! the typical case, cost only a couple of operations more than
//! [`division`] scaling.
//!
//! [Pekka Pulkkinen’s uniFloats wiki]: https://github.com/pekkizen/prng/wiki/uniFloats
//! [`division`]: crate::division

/// Returns a random `f64` distributed as a uniform 64-bit fixed-point real
/// in [0 . . 1) rounded down to the nearest representable value: every float
/// in [2⁻¹² . . 1), and the 2⁵² multiples of 2⁻⁶⁴ below 2⁻¹².
#[inline]
pub fn f64_64(mut bits: impl FnMut() -> u64) -> f64 {
    let u = bits();
    if u == 0 {
        return 0.0;
    }
    let z = u.leading_zeros() as u64;
    // The Go original defines z as the leading-zeros count plus one and
    // computes `u << z` with z possibly 64, which Go defines as 0; Rust
    // declares 64-bit shifts overflow, so here z is the plain leading-zeros
    // count (≤ 63, as u ≠ 0) and the last bit of the shift, which drops the
    // implicit leading one, is applied separately.
    f64::from_bits((1022 - z) << 52 | ((u << z) << 1) >> 12)
}

/// Returns 2⁻ⁿ as an `f64`; `n` must be at most 1022.
#[inline]
const fn two_to_minus(n: u64) -> f64 {
    debug_assert!(n <= 1022);
    f64::from_bits((1023 - n) << 52)
}

/// Returns a random `f64` distributed as a uniform 117-bit fixed-point real
/// in [0 . . 1) rounded down to the nearest representable value: every
/// float in [2⁻⁶⁵ . . 1), and the multiples of 2⁻¹¹⁷ below 2⁻⁶⁵.
///
/// A port of `Float64_117`: as [`f64_64`], except that when the first word
/// has 12 or more leading zeros (probability 2⁻¹²) a second word extends the
/// fixed point to 117 bits.
#[inline]
pub fn f64_117(mut bits: impl FnMut() -> u64) -> f64 {
    let u = bits();
    let z = u.leading_zeros() as u64;
    if z <= 11 {
        // 99.975% of cases; see [`f64_64`] for the shift structure.
        return f64::from_bits((1022 - z) << 52 | ((u << z) << 1) >> 12);
    }
    // Kluge; see [`crate::cold::cold_barrier`].
    crate::cold::cold_barrier();
    // The Go original computes `u << z` with z possibly 64 (first word 0),
    // which Go defines as 0; `checked_shl` + `unwrap_or` mirrors that
    // (as would `unbounded_shl`, which however requires Rust 1.87).
    let u = u.checked_shl(z as u32).unwrap_or(0) | bits() >> (64 - z);
    (u >> 11) as f64 * two_to_minus(53 + z)
}

/// Returns a random `f64` distributed as a uniform real in [0 . . 1)
/// rounded down to the nearest representable value: every float in
/// [0 . . 1) is reachable, subnormals and 0 included, with probability
/// equal to the measure of the reals that round down to it.
///
/// A port of `Float64full`: as [`f64_117`], except that zero words keep the
/// zoom going, 64 binades at a time, down to the bottom of the subnormals.
#[inline]
pub fn f64_full(mut bits: impl FnMut() -> u64) -> f64 {
    let mut u = bits();
    let mut z = u.leading_zeros() as u64;
    if z <= 11 {
        // 99.975% of cases; see [`f64_64`] for the shift structure.
        return f64::from_bits((1022 - z) << 52 | ((u << z) << 1) >> 12);
    }
    // Kluge; see [`crate::cold::cold_barrier`].
    crate::cold::cold_barrier();
    let mut exp = 0u64;
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
    // that (as would `unbounded_shr`, which however requires Rust 1.87).
    let u = u << z | bits().checked_shr(64 - z as u32).unwrap_or(0);
    exp += z;
    if exp < 1022 {
        return f64::from_bits((1022 - exp) << 52 | (u << 1) >> 12);
    }
    // The 2⁵² subnormal floats.
    f64::from_bits(u >> (exp - 1022) >> 12)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sources::Weyl;

    /// The bit-building form must agree with the wiki’s division form,
    /// `float64(u << z >> 11) / 2^53 / 2^z` with z = leadingZeros(u).
    #[test]
    fn test_matches_reference_division_form() {
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
    fn test_range() {
        let mut rng = Weyl(42);
        for _ in 0..100_000 {
            let x = f64_64(|| rng.next_u64());
            assert!((0.0..1.0).contains(&x), "f64_64: {x}");
        }
    }

    /// Extremes of the leading-zeros count.
    #[test]
    fn test_edge_cases() {
        assert_eq!(f64_64(|| 1 << 63), 0.5);
        assert_eq!(f64_64(|| u64::MAX), f64::from_bits(1.0f64.to_bits() - 1));
        assert_eq!(f64_64(|| 1), f64::from_bits((1023 - 64) << 52)); // 2^-64
        assert_eq!(f64_64(|| 0), 0.0);
    }
    /// A source that replays a fixed sequence of words, then panics.
    fn replay(words: &[u64]) -> impl FnMut() -> u64 + '_ {
        let mut iter = words.iter();
        move || *iter.next().expect("source exhausted")
    }

    /// Golden values produced by the verbatim Go code of the wiki, one per
    /// branch (fast path, slow path, extreme leading-zero counts, zero
    /// words).
    #[test]
    fn test_known_values_117() {
        for (words, expected) in [
            (&[0x8000000000000000, 0][..], 0x3fe0000000000000u64),
            (&[0xDEADBEEFDEADBEEF, 0][..], 0x3febd5b7ddfbd5b7),
            (&[1 << 52, 0][..], 0x3f30000000000000),
            (&[1 << 51, 0xC0FFEE0DDF00D5ED][..], 0x3f20000000000001),
            (&[1, 0xC0FFEE0DDF00D5ED][..], 0x3bfc0ffee0ddf00d),
            (&[3, 0xFFFFFFFFFFFFFFFF][..], 0x3c0fffffffffffff),
            (&[0, 0xC0FFEE0DDF00D5ED][..], 0x3be81ffdc1bbe01a),
            (&[0, 0][..], 0),
        ] {
            assert_eq!(
                f64_117(replay(words)).to_bits(),
                expected,
                "words {words:x?}"
            );
        }
    }

    /// Golden values produced by the verbatim Go code of the wiki; the deep
    /// sequences exercise the zero-word zoom, the subnormal construction and
    /// the all-zeros cutoff (which fires before the final exponent add).
    #[test]
    fn test_known_values_full() {
        for (words, expected) in [
            (vec![0x8000000000000000], 0x3fe0000000000000u64),
            (vec![0xDEADBEEFDEADBEEF], 0x3febd5b7ddfbd5b7),
            (vec![1 << 52], 0x3f30000000000000),
            (vec![1 << 51, 0xC0FFEE0DDF00D5ED], 0x3f20000000000001),
            (vec![1, 0xC0FFEE0DDF00D5ED], 0x3bfc0ffee0ddf00d),
            (vec![3, 0xFFFFFFFFFFFFFFFF], 0x3c0fffffffffffff),
            (vec![0, 0xC0FFEE0DDF00D5ED, 0xABCD], 0x3be81ffdc1bbe01a),
            (vec![0, 0, 0xC0FFEE0DDF00D5ED, 0xAB], 0x37e81ffdc1bbe01a),
            (
                [vec![0; 15], vec![2, u64::MAX]].concat(),
                0x000bffffffffffff,
            ),
            (
                [vec![0; 15], vec![1, u64::MAX]].concat(),
                0x0007ffffffffffff,
            ),
            (
                [vec![0; 16], vec![1 << 50, 0xFFFFFFFFFFFFF]].concat(),
                0x0000001000000000,
            ),
            ([vec![0; 16], vec![3, 0]].concat(), 0),
            (vec![0; 18], 0),
        ] {
            assert_eq!(
                f64_full(replay(&words)).to_bits(),
                expected,
                "words {words:x?}"
            );
        }
    }

    /// XOR-rotate hash over 10⁶ draws, cross-checked against the verbatim Go
    /// code of the wiki driven by the same Weyl source. The two variants
    /// agree on any zero-free source stream.
    #[test]
    fn test_matches_go_reference_hash() {
        let mut rng = Weyl(42);
        let mut h = 0u64;
        for _ in 0..1_000_000 {
            h = h.rotate_left(1) ^ f64_117(|| rng.next_u64()).to_bits();
        }
        assert_eq!(h, 0xf9b6db6017240be7);

        let mut rng = Weyl(42);
        let mut h = 0u64;
        for _ in 0..1_000_000 {
            h = h.rotate_left(1) ^ f64_full(|| rng.next_u64()).to_bits();
        }
        assert_eq!(h, 0xf9b6db6017240be7);
    }

    #[test]
    fn test_range_117_full() {
        let mut rng = Weyl(42);
        for _ in 0..100_000 {
            let x = f64_117(|| rng.next_u64());
            assert!((0.0..1.0).contains(&x), "f64_117: {x}");
            let x = f64_full(|| rng.next_u64());
            assert!((0.0..1.0).contains(&x), "f64_full: {x}");
        }
    }
}
