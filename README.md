# rand-float

This crate implements several techniques for generating uniform random
floating-point numbers in [0 . . 1) from a stream of random bits.

A [relatively innocuous pull
request](https://github.com/smol-rs/fastrand/pull/129) ended up in a rabbit hole
that led to this crate. The motivation was the state of disarray of the
literature on the topic: academic papers, GitHub repositories, and free-floating
sources, often lacking comparison with other approaches, a few bugs, and in some
cases definitely hostile notation. By gathering all techniques in the same
place, we have a common ground for comparison and benchmarking, and cross
references for future implementations.

Every technique is implemented as a pure transformation of a source of
uniform random 64-bit words (any `FnMut() -> u64`), documented with its
exact output distribution, and benchmarked against the others.

The translation from the sources in different languages was operated by
Anthropic's Fable and extensively checked against the original implementation.
In the process, we isolated a few bugs in the original code, that have been
reported to the authors.

| Module        | Origin                                                                                         | Distribution                                     | Reachable values                                               | Words per `f64`             |
| ------------- | ---------------------------------------------------------------------------------------------- | ------------------------------------------------ | -------------------------------------------------------------- | --------------------------- |
| `standard`    | folklore                                                                                       | equispaced                                       | the 2⁵³ multiples of 2⁻⁵³ in [0 . . 1)                         | 1                           |
| `pekkizen`    | [Pekka Pulkkinen's uniFloats](https://github.com/pekkizen/prng/wiki/uniFloats) (`Float64_64`)  | uniform real rounded **down** to a 2⁻⁶⁴ grid     | every float in [2⁻¹² . . 1); 2⁵² values spaced 2⁻⁶⁴ below 2⁻¹² | 1                           |
| `pekkizen`    | [Pekka Pulkkinen's uniFloats](https://github.com/pekkizen/prng/wiki/uniFloats) (`Float64_117`) | uniform real rounded **down** to a 2⁻¹¹⁷ grid    | every float in [2⁻⁶⁵ . . 1); multiples of 2⁻¹¹⁷ below 2⁻⁶⁵     | 1 + ≈2⁻¹²                   |
| `pekkizen`    | [Pekka Pulkkinen's uniFloats](https://github.com/pekkizen/prng/wiki/uniFloats) (`Float64full`) | uniform real in [0 . . 1) rounded **down**       | every float in [0 . . 1), including all subnormals             | 1 + ≈2⁻¹²                   |
| `campbell`    | Taylor R. Campbell's `binary64fast.c`                                                          | uniform real in [0 . . 1] rounded **to nearest** | every float in [2⁻¹²⁸ . . 1]                                   | 2 (or 3, for constant time) |
| `campbell`    | Taylor R. Campbell's [`random_real.c`](https://mumble.net/~campbell/2014/04/28/random_real.c)  | uniform real in [0 . . 1] rounded **to nearest** | every float in [2⁻¹⁰²⁴ . . 1], and 0                           | ≈1.5                        |
| `badizadegan` | [fp-rand](https://github.com/specbranch/fp-rand/) (round-down variant)                         | uniform real in (0 . . 1) rounded **down**       | every float in [0 . . 1), including all subnormals             | 1 + ≈2⁻¹²                   |

A few observations:

- Some of the techniques have variants: for example, both Badizadegan's
  and Pulkkinen's can return values in [0 . . 1] by rounding to nearest, but the
  semi-open interval is our target as it is the right one to do inversion (e.g.,
  if you do inversion of a discrete distribution on [0 . . 1] you'll be in
  trouble).

- Campbell's constant-time code is geared towards shielding from timing
  side-channel attacks, which is a non-goal for us.

- The documentation of `campbell::real` differs from the comments in
  Campbell's `random_real.c`: while porting it we found a minor bug (the
  all-zeros cutoff fires one 64-bit word too early, so the subnormals below
  2⁻¹⁰²⁴ are unreachable and 0 is returned slightly too often), which we
  reported to the author. The code is a faithful port; our documentation
  describes what the code actually does.

## Examples

```rust
use rand_float::{badizadegan, campbell, pekkizen, standard};

struct Xoroshiro128pp([u64; 2]);

impl Xoroshiro128pp {
    fn next_u64(&mut self) -> u64 {
        let [s0, mut s1] = self.0;
        let result = s0.wrapping_add(s1).rotate_left(17).wrapping_add(s0);
        s1 ^= s0;
        self.0 = [s0.rotate_left(49) ^ s1 ^ (s1 << 21), s1.rotate_left(28)];
        result
    }
}

let mut src = Xoroshiro128pp([0x243F6A8885A308D3, 0x13198A2E03707344]);
let a = standard::f64_53bits(|| src.next_u64());
let b = pekkizen::f64_64(|| src.next_u64());
let c = campbell::fast(|| src.next_u64());
let d = badizadegan::f64_down(|| src.next_u64());
```

## Benchmarks

`cargo bench` drives every technique with the same Weyl-sequence source in two
settings: one conversion per call, and filling an array of 1024 doubles per
iteration.
All builds use `-C target-cpu=native` (set in `.cargo/config.toml`), so the
benchmarks measure code generated for the machine they run on.

On CPUs with heterogeneous cores, pin the run to one core (build first so
compilation stays parallel):

```sh
cargo bench --no-run
taskset -c 2 cargo bench   # Linux; pick a performance core
```

To run the benchmarks and turn the results into a bar chart (ns per
generated `f64`, single-call and array-fill bars side by side; requires
Python with matplotlib) in one line:

```sh
cargo bench 2>&1 | python3 python/plot_bench.py -o bench.pdf
```

The output format follows the extension (`.pdf`, `.png`, `.svg`, ...), and
the script also accepts a previously saved log: `python3
python/plot_bench.py bench.txt -o bench.pdf`.

## Results

Here we display the results of the benchmarks on some architectures. Note that
because of a quirk in LLVM, we had to add (at least for the time being) a [cold
barrier](https://docs.rs/rand-float/latest/rand_float/cold/) preventing some
un-optimization of the array case due to interference with the underlying Weyl
generator. The barrier sometimes however disturbs a bit the single-call case. We
hope to remove it in the future.

### AMD Ryzen 9 5950X

![AMD Ryzen 9 5950X](https://raw.githubusercontent.com/vigna/rand-float-rs/main/img/Zen4.png)

### 12th Gen Intel® Core™ i7-12700KF @3.60 GHz

![12th Gen Intel® Core™ i7-12700KF @3.60 GHz](https://raw.githubusercontent.com/vigna/rand-float-rs/main/img/i7.png)

### Intel® Xeon® X5660 @2.80 GHz

![Intel® Xeon® X5660 @2.80 GHz](https://raw.githubusercontent.com/vigna/rand-float-rs/main/img/XeonX5660.png)

### Apple M1 Max @2.50 GHz

![Apple M1 Max @2.50 GHz](https://raw.githubusercontent.com/vigna/rand-float-rs/main/img/M1.png)

### Apple M5 Max @4.00 GHz

![Apple M5 Max @4.00 GHz](https://raw.githubusercontent.com/vigna/rand-float-rs/main/img/M5.png)

## Acknowledgments

I would like to thank Nima Badizadegan, Taylor R. Campbell, Frédéric Goualard,
and Reiner Pope for interesting discussions and a lot of useful pointers.

## Licensing

Original code in this crate is dual-licensed under Apache-2.0 or MIT. The
ported techniques retain the licenses of their authors, reproduced in the
header of the respective source files:

- `src/badizadegan.rs` — MIT, Copyright (c) 2025 Nima Badizadegan
  ([fp-rand](https://github.com/specbranch/fp-rand/));
- `src/campbell.rs` — BSD-2-Clause, Copyright (c) 2014-2026 Taylor R. Campbell
  (from `binary64fast.c` and `random_real.c`);
- `src/pekkizen.rs` — Copyright (c) 2020 Pekka Pulkkinen, distribution permitted
  ([uniFloats](https://github.com/pekkizen/prng/wiki/uniFloats));
