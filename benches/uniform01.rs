//! Speed comparison of `u64` → uniform `f64` in [0, 1) conversion techniques.
//!
//! Contenders:
//!
//! - `perfect_fp_down`: this crate — the fp-rand “perfect” generator,
//!   round-down variant (<https://github.com/specbranch/fp-rand/>).
//! - `campbell_fast`: Taylor R. Campbell’s `uniformbinary64_fast` from
//!   `binary64fast.c` — correctly rounded (to nearest) uniform reals, with a
//!   branch on the improbable zero mantissa.
//! - `campbell_consttime_if` / `campbell_consttime_smear`: Campbell’s
//!   branch-free variants. The `(t - 1) * 0x1p-64` term of the original is
//!   computed here in *signed* arithmetic, `((double)(int64_t)t - 1) * 0x1p-64`,
//!   so that t = 0 yields −1 rather than 2⁶⁴ − 1, and so that the
//!   integer→double conversion is signed (avoiding the branchy unsigned
//!   conversion sequence on pre-AVX-512 x86).
//! - `std_53bits`: the standard `(bits >> 11) * 0x1p-53`.
//! - `pekkizen_64`: the leading-zeros technique from
//!   <https://github.com/pekkizen/prng/wiki/uniFloats> (`Float64_64`).
//!
//! All contenders draw from the same Weyl-sequence generator used by the
//! benchmark harness in `binary64fast.c`, so the per-word entropy cost is
//! identical; differences reflect the conversion (and the number of words
//! each technique consumes: one for `std_53bits`, `pekkizen_64` and — almost
//! always — `perfect_fp_down`; two for `campbell_fast`; three for the
//! const-time variants). A `weyl_baseline` entry measures the bare generator.

use criterion::{Criterion, criterion_group, criterion_main};

const TWO_M53: f64 = 1.0 / (1u64 << 53) as f64;
/// 2⁻⁶⁴ (Rust has no hex float literals; built via the exponent field).
const TWO_M64: f64 = f64::from_bits((1023 - 64) << 52);
/// 2⁻¹²⁸ (used by the tests and by the x86-only unlikely path).
#[allow(dead_code)]
const TWO_M128: f64 = f64::from_bits((1023 - 128) << 52);
/// 2³².
#[cfg(target_arch = "x86_64")]
const TWO_P32: f64 = 4294967296.0;

/// The additive Weyl sequence used as `uniform64()` in `binary64fast.c`.
struct Weyl(u64);

impl Weyl {
    #[inline(always)]
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        self.0
    }
}

/// Port of Campbell’s `uniformbinary64_fastdet`: turns an exponent scale
/// `f` ∈ {2⁻⁶⁴, 2⁻¹²⁸}, a geometric word `m` and a significand word `u`
/// into a correctly rounded (to nearest) uniform binary64.
#[inline(always)]
fn campbell_fastdet(f: f64, m: u64, u: u64) -> f64 {
    // Largest power-of-two divisor of m, with bit 63 forced as a backstop
    // against a broken all-zero source; exactly representable, so the
    // conversion below is exact.
    let m = m | (1 << 63);
    let m = m & m.wrapping_neg();
    // On x86_64 there is no unsigned 64-bit → double instruction until
    // AVX-512; split into halves so the compiler emits branch-free signed
    // conversions, as in the C original.
    #[cfg(target_arch = "x86_64")]
    let d = ((m >> 32) as f64) * TWO_P32 + ((m & 0xFFFF_FFFF) as f64);
    #[cfg(not(target_arch = "x86_64"))]
    let d = m as f64;

    // Uniform odd integer in (2⁶³, 2⁶⁴): round-to-odd of a uniform real in
    // [2⁶³, 2⁶⁴]. The conversion rounds to nearest; ties are impossible.
    let u = u | (1 << 63) | 1;
    let s = u as f64;

    // Scale the significand into [1/2, 1] and apply the geometric exponent.
    s * f / d
}

/// Port of `uniformbinary64_fast` (branches on the 2⁻⁶⁴-probability event).
#[inline(always)]
fn campbell_fast(rng: &mut Weyl) -> f64 {
    let u = rng.next();
    let mut f = TWO_M64;
    let mut m = rng.next();
    if m == 0 {
        // unlikely
        f *= TWO_M64;
        m = rng.next();
    }
    campbell_fastdet(f, m, u)
}

/// Shared tail of the const-time variants: given the flag t ∈ {0, 1}
/// (t = 1 iff m ≠ 0), rescale `f` and substitute `m2` for a zero `m`,
/// both branch-free.
///
/// The original C computes `t*0x1p0 - (t - 1)*0x1p-64` with *unsigned*
/// t, which makes `(t - 1)` equal to 2⁶⁴ − 1 rather than −1 when t = 0;
/// here the subtraction is done after a signed integer → double
/// conversion, `((double)(int64_t)t - 1) * 0x1p-64`, which is both correct
/// and branch-free on older x86 (CVTSI2SD is signed).
#[inline(always)]
fn campbell_consttime_tail(t: u64, u: u64, m: u64, m2: u64) -> f64 {
    let mut f = TWO_M64;
    let tf = t as i64 as f64;
    f *= tf - (tf - 1.0) * TWO_M64;
    let m = m | (m2 & t.wrapping_sub(1));
    campbell_fastdet(f, m, u)
}

/// Port of `uniformbinary64_consttime_if`: the flag comes from `m != 0`,
/// which compiles to a branch-free flag-set.
#[inline(always)]
fn campbell_consttime_if(u: u64, m: u64, m2: u64) -> f64 {
    let t = (m != 0) as u64;
    campbell_consttime_tail(t, u, m, m2)
}

/// Port of `uniformbinary64_consttime_smear`: the flag is computed by
/// smearing every set bit of `m` down to bit 0 — branch-free at the source
/// level, independent of the optimizer.
#[inline(always)]
fn campbell_consttime_smear(u: u64, m: u64, m2: u64) -> f64 {
    let mut t = m;
    t |= t >> 1;
    t |= t >> 2;
    t |= t >> 4;
    t |= t >> 8;
    t |= t >> 16;
    t |= t >> 32;
    t &= 1;
    campbell_consttime_tail(t, u, m, m2)
}

/// The standard technique: top 53 bits scaled by 2⁻⁵³.
#[inline(always)]
fn std_53bits(rng: &mut Weyl) -> f64 {
    (rng.next() >> 11) as f64 * TWO_M53
}

/// pekkizen’s `Float64_64` (explicit bit-building form): count leading
/// zeros to pick the binade, shift the remaining bits into the mantissa.
/// Covers every float in [2⁻¹², 1) and 2⁵² evenly spaced values below 2⁻¹².
///
/// The Go original computes `u << z` with z possibly 64, which Go defines
/// as 0; Rust declares 64-bit shifts overflow, so the shift is split as
/// `(u << (z - 1)) << 1` (z ≥ 1 always).
#[inline(always)]
fn pekkizen_64(rng: &mut Weyl) -> f64 {
    let u = rng.next();
    if u == 0 {
        return 0.0;
    }
    let z = u.leading_zeros() as u64 + 1;
    f64::from_bits((1023 - z) << 52 | ((u << (z - 1)) << 1) >> 12)
}

fn bench(c: &mut Criterion) {
    let mut g = c.benchmark_group("u64_to_f64");
    g.warm_up_time(std::time::Duration::from_secs(1));
    g.measurement_time(std::time::Duration::from_secs(3));

    g.bench_function("perfect_fp_down", |b| {
        let mut rng = Weyl(0x0123_4567_89AB_CDEF);
        b.iter(|| perfect_fp_rs::f64_down(|| rng.next()))
    });

    g.bench_function("campbell_fast", |b| {
        let mut rng = Weyl(0x0123_4567_89AB_CDEF);
        b.iter(|| campbell_fast(&mut rng))
    });

    g.bench_function("campbell_consttime_if", |b| {
        let mut rng = Weyl(0x0123_4567_89AB_CDEF);
        b.iter(|| {
            let (c0, c1, c2) = (rng.next(), rng.next(), rng.next());
            campbell_consttime_if(c0, c1, c2)
        })
    });

    g.bench_function("campbell_consttime_smear", |b| {
        let mut rng = Weyl(0x0123_4567_89AB_CDEF);
        b.iter(|| {
            let (c0, c1, c2) = (rng.next(), rng.next(), rng.next());
            campbell_consttime_smear(c0, c1, c2)
        })
    });

    g.bench_function("std_53bits", |b| {
        let mut rng = Weyl(0x0123_4567_89AB_CDEF);
        b.iter(|| std_53bits(&mut rng))
    });

    g.bench_function("pekkizen_64", |b| {
        let mut rng = Weyl(0x0123_4567_89AB_CDEF);
        b.iter(|| pekkizen_64(&mut rng))
    });

    g.bench_function("weyl_baseline", |b| {
        let mut rng = Weyl(0x0123_4567_89AB_CDEF);
        b.iter(|| rng.next())
    });

    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);

/// Sanity checks for the contenders implemented in this file. Since
/// `harness = false` benches never run `#[test]` functions, this module is
/// pulled into the test harness by `tests/bench_contenders.rs`.
#[cfg(test)]
mod tests {
    // Unused when the bench target itself is compiled with cfg(test) but
    // without the test harness (which strips `#[test]` functions).
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn contenders_stay_in_unit_interval() {
        let mut rng = Weyl(42);
        for _ in 0..100_000 {
            let x = campbell_fast(&mut rng);
            assert!(x > 0.0 && x <= 1.0, "campbell_fast: {x}");
            let (c0, c1, c2) = (rng.next(), rng.next(), rng.next());
            let x = campbell_consttime_if(c0, c1, c2);
            assert!(x > 0.0 && x <= 1.0, "consttime_if: {x}");
            let x = campbell_consttime_smear(c0, c1, c2);
            assert!(x > 0.0 && x <= 1.0, "consttime_smear: {x}");
            let x = std_53bits(&mut rng);
            assert!((0.0..1.0).contains(&x), "std_53bits: {x}");
            let x = pekkizen_64(&mut rng);
            assert!((0.0..1.0).contains(&x), "pekkizen_64: {x}");
        }
    }

    /// The two const-time variants and the branchy variant must agree.
    #[test]
    fn campbell_variants_agree() {
        let mut rng = Weyl(0xDEAD_BEEF);
        for _ in 0..100_000 {
            let (c0, c1, c2) = (rng.next(), rng.next(), rng.next());
            assert_eq!(
                campbell_consttime_if(c0, c1, c2),
                campbell_consttime_smear(c0, c1, c2)
            );
        }
    }

    /// The signed-arithmetic fix: with m = 0 the scale must become 2⁻¹²⁸
    /// (not go negative as with the unsigned `(t - 1)` of the original C).
    #[test]
    fn consttime_zero_mantissa_rescales() {
        let u = 0x0123_4567_89AB_CDEF;
        let m2 = 0x8000_0000_0000_0000u64; // power of two: d = 2^63
        let x = campbell_consttime_if(u, 0, m2);
        assert!(x > 0.0, "scale went negative: {x}");
        // s ≈ 2^63, f = 2^-128, d = 2^63  ⇒  x ≈ 2^-128.
        assert_eq!(x, campbell_fastdet(TWO_M128, m2, u));
        // And with m ≠ 0, m2 must be ignored.
        assert_eq!(
            campbell_consttime_if(u, 3, m2),
            campbell_fastdet(TWO_M64, 3, u)
        );
    }

    /// pekkizen’s bit-building form must agree with the wiki’s division
    /// form, `float64(u << z >> 11) / 2^53 / 2^z` with z = leadingZeros(u).
    #[test]
    fn pekkizen_matches_reference_division_form() {
        let division_form = |u: u64| {
            let z = u.leading_zeros();
            let m = ((u << z) >> 11) as f64;
            m / (1u64 << 53) as f64 / (1u64 << z) as f64
        };
        let mut rng = Weyl(7);
        for _ in 0..100_000 {
            let u = rng.0.wrapping_add(0x9E3779B97F4A7C15);
            assert_eq!(pekkizen_64(&mut rng), division_form(u));
        }
        // Edge cases: extremes of the leading-zeros count.
        let f = |w: u64| pekkizen_64(&mut Weyl(w.wrapping_sub(0x9E3779B97F4A7C15)));
        assert_eq!(f(1 << 63), 0.5);
        assert_eq!(f(u64::MAX), f64::from_bits(1.0f64.to_bits() - 1));
        assert_eq!(f(1), f64::from_bits((1023 - 64) << 52)); // 2^-64
        assert_eq!(f(0), 0.0);
    }
}
