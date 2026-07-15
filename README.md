# rand-float-rs

A comparison of techniques for generating uniform random floating-point
numbers in [0, 1) from a stream of random bits.

Every technique is implemented as a pure transformation of a source of
uniform random 64-bit words (any `FnMut() -> u64`), documented with its
exact output distribution, and benchmarked against the others. No technique
is preferred: they make different tradeoffs between speed, entropy
consumption, and which floating-point values they can produce.

| Module | Origin | Distribution | Reachable values | Words per `f64` |
|--------|--------|--------------|------------------|-----------------|
| `standard` | folklore | equispaced lattice | the 2⁵³ multiples of 2⁻⁵³ in [0, 1) | 1 |
| `pekkizen` | [pekkizen's uniFloats](https://github.com/pekkizen/prng/wiki/uniFloats) (`Float64_64`) | uniform real rounded down to a 2⁻⁶⁴ grid | every float in [2⁻¹², 1); 2⁵² values spaced 2⁻⁶⁴ below 2⁻¹² | 1 |
| `campbell` | Taylor R. Campbell's `binary64fast.c` | uniform real in [0, 1] rounded **to nearest** | every float in [2⁻¹²⁸, 1] and 0 | 2 (or 3, const-time) |
| `perfect` | [fp-rand](https://github.com/specbranch/fp-rand/) (round-down variant) | uniform real in (0, 1) rounded **down** | every float in [0, 1), including all subnormals | 1 + ≈2⁻¹² |

Notes on the ports:

- `perfect` is validated bit-for-bit against the reference Go implementation
  (see `examples/crosscheck.rs`); `f32` generation is also provided.
- In `campbell`, the `(t - 1) * 0x1p-64` rescaling of the const-time
  variants is computed in signed arithmetic, so that it is correct and
  branch-free on pre-AVX-512 x86, where the unsigned integer→double
  conversion is branchy.
- `pekkizen` uses the explicit bit-building form, validated against the
  wiki's division form.

```rust
use rand_float_rs::{campbell, pekkizen, perfect, standard, sources::SplitMix64};

let mut src = SplitMix64(42);
let a = standard::f64_53bits(|| src.next_u64());
let b = pekkizen::f64_64(|| src.next_u64());
let c = campbell::fast(|| src.next_u64());
let d = perfect::f64_down(|| src.next_u64());
```

## Benchmarks

`cargo bench` drives every technique with the same Weyl-sequence source
(the one used by the benchmark harness of `binary64fast.c`, whose baseline
cost is measured separately) in two settings: one conversion per call, and
filling an array of 1024 doubles per iteration. `contrib/x86-bench.sh`
additionally runs, pinned to a single CPU, the Go benchmarks of the fp-rand
repository together with a corrected version of their baseline.
