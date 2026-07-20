# Change Log

## unreleased

- Following a suggestion in a pull request by Reiner Pope, the code for
  pekkizen has been slightly simplified.

## [0.2.0] - 2026-07-17

### Changed

- The `standard` module has been renamed `division`: the technique is
  now named after what it does (extracting 53 bits and dividing by 2⁵³)
  rather than after its ubiquity.

### New

- A `gaussian_tail` example: a multithreaded run, driven by per-block MWC192
  streams, generating Gaussian deviates by inversion with the division technique
  and with `unif_01`: in the first case, statistically impossible collisions
  appear due to the limited precision of the division technique.

## [0.1.2] - 2026-07-16

### New

- A `uniform` module aliasing `pekkizen`, the technique of choice, with a
  `unif_01` entry-point function; with the `rand` feature (enabled by
  default) its `Unif01Ext` extension trait provides the same conversion as
  a `unif_01` method on every `rand` generator.

## [0.1.1] - 2026-07-16

### New

- On x86-64 compiled with AVX-512F, `campbell::real` scales its result with
  the hardware `ldexp` (a single `vscalefsd` instruction, via inline
  assembly) instead of two multiplications, so performance should be
  identical to the original code.

- The `cold` module is now public, making the documentation of the barrier
  against if-conversion linkable from the README.

### Fixed

- Leftover references to the pre-rename crate name `rand_float_rs` in the
  README example, in the `badizadegan` doctests, and in the benchmarks; the
  `documentation` metadata now points to `docs.rs/rand-float`.

## [0.1.0] - 2026-07-16

### New

- First release.
