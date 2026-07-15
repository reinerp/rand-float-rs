# perfect-fp-rs

Perfect uniform floating-point random number generation for Rust: a port of
the *round-down* variant of the [fp-rand](https://github.com/specbranch/fp-rand/)
algorithm (C++/Go reference implementations and paper).

`f64_down` (and its `f32_down` counterpart) returns a value distributed
exactly as if a real number had been drawn uniformly from (0, 1) and then
rounded down (toward −∞) to a representable floating-point value. The result
lies in [0, 1), and *every* float in that range — including every subnormal
and 0 — is returned with probability equal to the measure of the reals that
round down to it. This is a strict upgrade over the usual
`(bits >> 11) * 2⁻⁵³` technique, which can only produce 2⁵³ equally spaced
values.

The generator is a pure transformation of a stream of uniform 64-bit words:
it takes any `FnMut() -> u64` and, thanks to an internal entropy pool that
recycles leftover bits, consumes a single word per `f64` except with
probability ≈ 2⁻¹².

```rust
let mut state = 0x9E3779B97F4A7C15u64;
let mut next = move || {
    // SplitMix64
    state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
};

let x = perfect_fp_rs::f64_down(&mut next);
assert!((0.0..1.0).contains(&x));
```

The port is validated bit-for-bit against the reference Go implementation
(see `examples/crosscheck.rs`).

## Benchmarks

`cargo bench` compares the conversion speed of this crate against:

- Taylor R. Campbell's `uniformbinary64_fast` and its two constant-time
  variants (`binary64fast.c`), which produce correctly rounded
  (to-nearest) uniform doubles — with the `(t - 1) * 0x1p-64` rescaling
  computed in signed arithmetic to avoid a branchy unsigned
  integer→double conversion on older x86;
- the standard `(bits >> 11) * 2⁻⁵³` technique;
- the leading-zeros technique from
  [pekkizen's uniFloats](https://github.com/pekkizen/prng/wiki/uniFloats)
  (`Float64_64`), which covers every float in [2⁻¹², 1).

All contenders are fed by the same Weyl-sequence generator used by the
benchmark harness of `binary64fast.c`.
