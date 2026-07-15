//! Trivial deterministic 64-bit generator.
//!
//! [`Weyl`] is not a recommended general-purpose generator: it has insufficient
//! statistical quality for anything but feeding a benchmark or a test. It is
//! here so that all conversion techniques can be driven by the same cheap,
//! inlinable bit stream.

/// A Weyl (additive) generator: the state advances by the golden-ratio
/// constant 0x9E3779B97F4A7C15 and is returned as is.
pub struct Weyl(pub u64);

impl Weyl {
    /// Returns the next 64-bit word of the sequence.
    #[inline(always)]
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        self.0
    }
}
