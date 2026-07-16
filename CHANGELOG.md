# Change Log

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
