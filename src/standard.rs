//! The standard shift-and-scale conversion.
//!
//! The technique used by most language runtimes and libraries: take as many
//! bits as the target type has of precision (53 for `f64`, 24 for `f32`)
//! and scale them by the corresponding power of two. Every arithmetic step
//! is exact, so the result is an equispaced lattice: the 2⁵³ multiples of
//! 2⁻⁵³ in [0, 1) for `f64`, each with probability 2⁻⁵³.
//!
//! This is by far the cheapest conversion — a shift, an integer→float
//! conversion, and a multiplication by a constant — but most representable
//! floats never occur: below 2⁻¹ the lattice is coarser than the
//! floating-point grid (in the binade [2⁻ᵏ⁻¹, 2⁻ᵏ) only every 2ᵏ-th float
//! is reachable), no nonzero value below 2⁻⁵³ can appear, and 0 has the
//! same probability 2⁻⁵³ as any other lattice point.

/// 2⁻⁵³.
const TWO_M53: f64 = 1.0 / (1u64 << 53) as f64;
/// 2⁻²⁴.
const TWO_M24: f32 = 1.0 / (1u32 << 24) as f32;

/// Returns a random `f64` uniform on the 2⁵³ multiples of 2⁻⁵³ in [0, 1):
/// the top 53 bits of one word, scaled by 2⁻⁵³.
#[inline]
pub fn f64_53bits(mut bits: impl FnMut() -> u64) -> f64 {
    (bits() >> 11) as f64 * TWO_M53
}

/// Returns a random `f32` uniform on the 2²⁴ multiples of 2⁻²⁴ in [0, 1):
/// the top 24 bits of one word, scaled by 2⁻²⁴.
#[inline]
pub fn f32_24bits(mut bits: impl FnMut() -> u64) -> f32 {
    (bits() >> 40) as f32 * TWO_M24
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sources::Weyl;

    #[test]
    fn range_and_lattice() {
        let mut rng = Weyl(42);
        for _ in 0..100_000 {
            let x = f64_53bits(|| rng.next_u64());
            assert!((0.0..1.0).contains(&x), "f64_53bits: {x}");
            // Every value is a multiple of 2^-53.
            assert_eq!(x, (x / TWO_M53).round() * TWO_M53);
            let y = f32_24bits(|| rng.next_u64());
            assert!((0.0..1.0).contains(&y), "f32_24bits: {y}");
            assert_eq!(y, (y / TWO_M24).round() * TWO_M24);
        }
    }

    #[test]
    fn extremes() {
        assert_eq!(f64_53bits(|| 0), 0.0);
        assert_eq!(f64_53bits(|| !0u64), 1.0 - TWO_M53);
        assert_eq!(f32_24bits(|| 0), 0.0);
        assert_eq!(f32_24bits(|| !0u64), 1.0 - TWO_M24);
    }
}
