//! Speed comparison of the `u64` → uniform `f64` conversion techniques
//! implemented by this crate (see the crate documentation for what each
//! technique computes).
//!
//! All techniques draw from the same Weyl-sequence generator used by the
//! benchmark harness of Campbell's `binary64fast.c`, so the per-word
//! entropy cost is identical; differences reflect the conversion and the
//! number of words each technique consumes (one for `standard`,
//! `pekkizen` and — almost always — `perfect`; two for `campbell::fast`;
//! three for the const-time variants). A `weyl_baseline` entry measures
//! the bare generator.
//!
//! Two groups are measured: `per_call` times one conversion, and
//! `fill_1024` fills an array of 1024 doubles per iteration (throughput
//! is reported per element), which leaves the optimizer free to overlap
//! consecutive conversions the way a bulk-generation loop would.

use criterion::{
    BenchmarkGroup, Criterion, Throughput, criterion_group, criterion_main,
    measurement::WallTime,
};
use rand_float_rs::{campbell, pekkizen, perfect, sources::Weyl, standard};

const SEED: u64 = 0x0123_4567_89AB_CDEF;
const FILL: usize = 1024;

fn bench_per_call(g: &mut BenchmarkGroup<WallTime>, name: &str, mut f: impl FnMut(&mut Weyl) -> f64) {
    g.bench_function(name, move |b| {
        let mut rng = Weyl(SEED);
        b.iter(|| f(&mut rng))
    });
}

fn bench_fill(g: &mut BenchmarkGroup<WallTime>, name: &str, mut f: impl FnMut(&mut Weyl) -> f64) {
    g.bench_function(name, move |b| {
        let mut rng = Weyl(SEED);
        let mut buf = [0.0f64; FILL];
        b.iter(|| {
            for x in buf.iter_mut() {
                *x = f(&mut rng);
            }
            std::hint::black_box(&buf);
        })
    });
}

/// Runs one group with every technique (and the bare-source baseline,
/// returning the word reinterpreted as an `f64`, which costs nothing).
///
/// A macro rather than a function taking a callback: each technique closure
/// must reach the registrar as its own monomorphized type, since routing it
/// through a `fn` pointer would force an indirect, non-inlinable call per
/// conversion and swamp the measurement.
macro_rules! bench_all {
    ($g:expr, $one:ident) => {
        $one($g, "perfect_down", |r| perfect::f64_down(|| r.next_u64()));
        $one($g, "campbell_fast", |r| campbell::fast(|| r.next_u64()));
        $one($g, "campbell_consttime_if", |r| {
            campbell::consttime_if(|| r.next_u64())
        });
        $one($g, "campbell_consttime_smear", |r| {
            campbell::consttime_smear(|| r.next_u64())
        });
        $one($g, "standard_53bits", |r| standard::f64_53bits(|| r.next_u64()));
        $one($g, "pekkizen_64", |r| pekkizen::f64_64(|| r.next_u64()));
        $one($g, "weyl_baseline", |r| f64::from_bits(r.next_u64()));
    };
}

fn bench(c: &mut Criterion) {
    let mut g = c.benchmark_group("per_call");
    g.warm_up_time(std::time::Duration::from_secs(1));
    g.measurement_time(std::time::Duration::from_secs(3));
    bench_all!(&mut g, bench_per_call);
    g.finish();

    let mut g = c.benchmark_group("fill_1024");
    g.warm_up_time(std::time::Duration::from_secs(1));
    g.measurement_time(std::time::Duration::from_secs(3));
    g.throughput(Throughput::Elements(FILL as u64));
    bench_all!(&mut g, bench_fill);
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
