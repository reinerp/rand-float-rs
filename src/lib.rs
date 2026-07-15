//! Techniques for generating uniform random floating-point numbers in
//! [0, 1) from a stream of random bits.
//!
//! This crate collects, documents, and benchmarks against one another
//! several techniques that turn a source of uniform random 64-bit words
//! (any `FnMut() -> u64`) into uniformly distributed floating-point
//! numbers. No technique is preferred: they make different tradeoffs
//! between speed, entropy consumption, and which floating-point values
//! they can produce, with which distribution.
//!
//! | Module | Distribution | Reachable values | Words per `f64` |
//! |--------|--------------|------------------|-----------------|
//! | [`standard`] | equispaced lattice | the 2⁵³ multiples of 2⁻⁵³ in [0, 1) | 1 |
//! | [`pekkizen`] | uniform real rounded down to a 2⁻⁶⁴ grid | every float in [2⁻¹², 1); 2⁵² values spaced 2⁻⁶⁴ below 2⁻¹² | 1 |
//! | [`campbell`] | uniform real in [0, 1] rounded **to nearest** | every float in [2⁻¹²⁸, 1] and 0 | 2 or 3 |
//! | [`perfect`] | uniform real in (0, 1) rounded **down** | every float in [0, 1), including all subnormals | 1 + ≈2⁻¹² |
//!
//! All techniques are implemented as pure transformations of the bit
//! stream, so they can be driven by any generator; the [`sources`] module
//! provides the two trivial deterministic generators used by the tests,
//! the examples and the benchmarks.
//!
//! The Criterion benchmark (`cargo bench`) measures every technique both
//! per call and filling an array of 1024 doubles, against the baseline
//! cost of the bit source itself.

#![deny(missing_docs)]

pub mod campbell;
pub mod pekkizen;
pub mod perfect;
pub mod sources;
pub mod standard;
