# Change Log

## [0.1.1] - 2026-07-16

### New

- On x86-64 compiled with AVX-512F, `campbell::real` scales its result with
  the hardware `ldexp` (a single `vscalefsd` instruction, via inline
  assembly) instead of two multiplications, so performance should be
  identical to the original code.

## [0.1.0] - 2026-07-16

### New

- First release.
