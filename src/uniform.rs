//! The technique of choice, plus integration with the [`rand`] crate.
//!
//! This module is an alias for [`pekkizen`], whose
//! leading-zeros technique provides complete coverage of [0 . . 1) at
//! almost the cost of the standard 53-bit scaling, consuming one 64-bit
//! word per call with probability 1 − 2⁻¹².
//!
//! The [`unif_01`] function converts any source of random 64-bit words;
//! with the `rand` feature (enabled by default), the [`Unif01Ext`]
//! extension trait additionally endows every generator implementing
//! [`rand_core::Rng`] with the same conversion as a [`unif_01`][method]
//! method.
//!
//! [`rand`]: https://crates.io/crates/rand
//! [`pekkizen`]: crate::pekkizen
//! [method]: Unif01Ext::unif_01

pub use crate::pekkizen::*;

/// Returns a random `f64` distributed as a uniform real in [0 . . 1)
/// rounded down to the nearest representable value: every float in
/// [0 . . 1) is reachable, subnormals and 0 included, with probability
/// equal to the measure of the reals that round down to it.
///
/// An alias for [`f64_full`], the technique of choice; the `rand` feature
/// provides the same conversion as a method ([`Unif01Ext::unif_01`]).
///
/// # Examples
///
/// ```
/// let mut src = rand_float::sources::Weyl(42);
/// let x = rand_float::uniform::unif_01(|| src.next_u64());
/// assert!((0.0..1.0).contains(&x));
/// ```
#[inline(always)]
pub fn unif_01(bits: impl FnMut() -> u64) -> f64 {
    f64_full(bits)
}

/// Extension trait adding a [`unif_01`][method] method to every generator
/// implementing [`rand_core::Rng`] (in particular, to the generators of the
/// [`rand`] crate).
///
/// [`rand`]: https://crates.io/crates/rand
/// [method]: Self::unif_01
#[cfg(feature = "rand")]
pub trait Unif01Ext {
    /// Returns a random `f64` distributed as a uniform real in [0 . . 1)
    /// rounded down to the nearest representable value: every float in
    /// [0 . . 1) is reachable, subnormals and 0 included, with probability
    /// equal to the measure of the reals that round down to it.
    ///
    /// This is [`f64_full`] applied to the generator.
    ///
    /// # Examples
    ///
    /// ```
    /// use rand_float::uniform::Unif01Ext;
    ///
    /// let x = rand::rng().unif_01();
    /// assert!((0.0..1.0).contains(&x));
    /// ```
    fn unif_01(&mut self) -> f64;
}

#[cfg(feature = "rand")]
impl<R: rand_core::Rng + ?Sized> Unif01Ext for R {
    #[inline(always)]
    fn unif_01(&mut self) -> f64 {
        f64_full(|| self.next_u64())
    }
}

#[cfg(all(test, feature = "rand"))]
mod tests {
    use super::*;
    use crate::sources::Weyl;

    /// [`Weyl`] as a [`rand_core::Rng`] (via an infallible
    /// [`rand_core::TryRng`]), for testing only.
    struct WeylRng(Weyl);

    impl rand_core::TryRng for WeylRng {
        type Error = core::convert::Infallible;

        fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
            Ok(self.0.next_u64() as u32)
        }

        fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
            Ok(self.0.next_u64())
        }

        fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Self::Error> {
            for chunk in dst.chunks_mut(8) {
                let bytes = self.0.next_u64().to_le_bytes();
                chunk.copy_from_slice(&bytes[..chunk.len()]);
            }
            Ok(())
        }
    }

    /// Method and function form of `unif_01` must agree with `f64_full`
    /// on the same word stream.
    #[test]
    fn test_unif_01_matches_f64_full() {
        let mut rng = WeylRng(Weyl(42));
        let mut src = Weyl(42);
        let mut src2 = Weyl(42);
        for _ in 0..100_000 {
            let expected = f64_full(|| src.next_u64());
            assert_eq!(rng.unif_01(), expected);
            assert_eq!(unif_01(|| src2.next_u64()), expected);
        }
    }
}
