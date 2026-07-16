//! Speed comparison of the `u64` → uniform `f64` conversion techniques
//! implemented by this crate (see the crate documentation for what each
//! technique computes).
//!
//! Two groups are measured: `per_call` times one conversion, and
//! `fill_1024` fills an array of 1024 doubles per iteration (throughput
//! is reported per element), which leaves the optimizer free to overlap
//! consecutive conversions the way a bulk-generation loop would. `sum_1024`
//! generates and sums 1024 doubles, forcing the generated floats into
//! floating point registers in the way numeric code would.

use criterion::{
    BenchmarkGroup, Criterion, Throughput, criterion_group, criterion_main, measurement::WallTime,
};
use rand_float::{badizadegan, campbell, division, pekkizen, sources::Weyl};

const SEED: u64 = 0x0123_4567_89AB_CDEF;
const FILL: usize = 1024;

fn bench_per_call(
    g: &mut BenchmarkGroup<WallTime>,
    name: &str,
    mut f: impl FnMut(&mut Weyl) -> f64,
) {
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
            // The source state must be a local of this closure: `Bencher::iter`
            // is not inlined and receives the captured `&mut` pointers as plain
            // struct fields without `noalias`, so a state living outside the
            // closure cannot be kept in a register across the buffer writes and
            // every conversion pays a load/store roundtrip on the state.
            let mut local = Weyl(rng.0);
            for x in buf.iter_mut() {
                *x = f(&mut local);
            }
            rng.0 = local.0;
            std::hint::black_box(&buf);
        })
    });
}

fn bench_sum(g: &mut BenchmarkGroup<WallTime>, name: &str, mut f: impl FnMut(&mut Weyl) -> f64) {
    g.bench_function(name, move |b| {
        let mut rng = Weyl(SEED);
        b.iter(|| {
            // f64 adding has a multiple-cycle latency on many CPUs, which is
            // longer than the latency of the Weyl RNG. We unroll the loop 4x
            // to allow the CPU to overlap 4 of the generations.
            let mut local = Weyl(rng.0);
            let mut sum0 = 0.0;
            let mut sum1 = 0.0;
            let mut sum2 = 0.0;
            let mut sum3 = 0.0;
            for _ in 0..FILL / 4 {
                sum0 += f(&mut local);
                sum1 += f(&mut local);
                sum2 += f(&mut local);
                sum3 += f(&mut local);
            }
            rng.0 = local.0;
            std::hint::black_box((sum0, sum1, sum2, sum3));
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
        $one($g, "weyl_baseline", |r| f64::from_bits(r.next_u64()));
        $one($g, "division_53bits", |r| {
            division::f64_53bits(|| r.next_u64())
        });
        $one($g, "pekkizen_64", |r| pekkizen::f64_64(|| r.next_u64()));
        $one($g, "pekkizen_117", |r| pekkizen::f64_117(|| r.next_u64()));
        $one($g, "pekkizen_full", |r| pekkizen::f64_full(|| r.next_u64()));
        $one($g, "badizadegan_down", |r| {
            badizadegan::f64_down(|| r.next_u64())
        });
        $one($g, "campbell_fast", |r| campbell::fast(|| r.next_u64()));
        $one($g, "campbell_real", |r| campbell::real(|| r.next_u64()));
        $one($g, "campbell_consttime_cmove", |r| {
            campbell::consttime_cmove(|| r.next_u64())
        });
        $one($g, "campbell_consttime", |r| {
            campbell::consttime(|| r.next_u64())
        });
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

    let mut g = c.benchmark_group("sum_1024");
    g.warm_up_time(std::time::Duration::from_secs(1));
    g.measurement_time(std::time::Duration::from_secs(3));
    g.throughput(Throughput::Elements(FILL as u64));
    bench_all!(&mut g, bench_sum);
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
