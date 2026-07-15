//! Trivial deterministic 64-bit generators used by the tests, the examples
//! and the benchmarks.
//!
//! These are *not* recommended general-purpose generators: [`Weyl`] in
//! particular has grossly insufficient statistical quality for anything but
//! feeding a benchmark. They are here so that all conversion techniques can
//! be driven by the same cheap, inlinable bit stream.

/// The additive Weyl sequence used as `uniform64()` by the benchmark
/// harness of Campbell's `binary64fast.c`: the state advances by the
/// golden-ratio constant 0x9E3779B97F4A7C15 and is returned as is.
///
/// The public field is the current state (the seed, before the first call).
pub struct Weyl(pub u64);

impl Weyl {
    /// Returns the next 64-bit word of the sequence.
    #[inline(always)]
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        self.0
    }
}

/// Vigna's SplitMix64: a Weyl sequence passed through a two-round
/// xor-shift-multiply finalizer. Statistically sound, equidistributed,
/// and cheap; commonly used to seed other generators.
///
/// The public field is the current state (the seed, before the first call).
pub struct SplitMix64(pub u64);

impl SplitMix64 {
    /// Returns the next 64-bit word of the sequence.
    #[inline(always)]
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}
