//! Statistically impossible collisions in the Gaussian tail.
//!
//! This example draws a large number of Gaussian deviates by inversion and
//! counts the duplicate values in the tail. Such duplicates are statistically
//! impossible in theory, and also in practice when the whole range of
//! floating-point numbers is available, but they are generated easily
//! if one uses division instead of the other techniques described in
//! this crate.
//!
//! Run with `cargo run --release --example gaussian_tail`; an optional argument
//! overrides the number of draws per converter (e.g. `1e10` for a quick,
//! seconds-long run which, however, is too short to accumulate duplicates).

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use rand_float::division::f64_53bits;
use rand_float::uniform::unif_01;

/// Any seed exhibits the phenomenon; change at will.
const SEED: u64 = 42;

/// Draws per converter; override with the first command-line argument.
const DRAWS: u64 = 1_200_000_000_000;

/// Draws per work unit pulled by a thread.
const BLOCK: u64 = 1_000_000_000;

/// Tail threshold 2⁻²²: uniforms below it map to deviates < −5.03σ.
const THRESHOLD: f64 = f64::from_bits((1023 - 22) << 52);

/// The SplitMix64 finalizer (Steele, Lea and Flood, OOPSLA 2014), used
/// only to scramble block indices into stream parameters.
const fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// [MWC192], a multiply-with-carry generator.
///
/// [MWC192]: https://prng.di.unimi.it/MWC192.c
struct Mwc192 {
    x: u64,
    y: u64,
    c: u64,
}

impl Mwc192 {
    const MWC_A2: u64 = 0xFFA04E67B3C95D86;

    #[inline(always)]
    fn next_u64(&mut self) -> u64 {
        let result = self.y;
        let t = Self::MWC_A2 as u128 * self.x as u128 + self.c as u128;
        self.x = self.y;
        self.y = t as u64;
        self.c = (t >> 64) as u64;
        result
    }
}

/// [`f64_53bits`] with the customary guard against u = 0 (which would
/// send the inverse CDF to −∞): redraw until nonzero.
fn f64_53bits_guarded(mut bits: impl FnMut() -> u64) -> f64 {
    loop {
        let u = f64_53bits(&mut bits);
        if u != 0.0 {
            return u;
        }
    }
}

/// Lower-tail inverse normal CDF Φ⁻¹(p) for p < 0.02425: Acklam’s
/// rational approximation (relative error below 1.2·10⁻⁹), tail branch.
fn probit_tail(p: f64) -> f64 {
    debug_assert!(p > 0.0 && p < 0.02425);
    let q = (-2.0 * p.ln()).sqrt();
    (((((-7.784894002430293e-3 * q - 3.223964580411365e-1) * q - 2.400758277161838) * q
        - 2.549732539343734)
        * q
        + 4.374664141464968)
        * q
        + 2.938163982698783)
        / ((((7.784695709041462e-3 * q + 3.224671290700398e-1) * q + 2.445134137142996) * q
            + 3.754408661907416)
            * q
            + 1.0)
}

/// Runs `draws` conversions on all cores and returns the sorted bit
/// patterns of the uniforms that fell below [`THRESHOLD`]. Each
/// [`BLOCK`]-sized work unit runs its own [`Mwc192`] stream, seeded by
/// scrambling the block index.
fn tail_draws(draws: u64, convert: impl Fn(&mut Mwc192) -> f64 + Sync) -> Vec<u64> {
    let blocks = draws.div_ceil(BLOCK);
    let next = AtomicU64::new(0);
    let hits = Mutex::new(Vec::new());
    let threads = std::thread::available_parallelism().map_or(4, usize::from);
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                let mut local = Vec::new();
                loop {
                    let b = next.fetch_add(1, Ordering::Relaxed);
                    if b >= blocks {
                        break;
                    }
                    let mut src = Mwc192 {
                        x: mix(SEED ^ (2 * b)),
                        y: mix(SEED ^ (2 * b + 1)),
                        c: 1,
                    };
                    for _ in 0..BLOCK.min(draws - b * BLOCK) {
                        let u = convert(&mut src);
                        if u < THRESHOLD {
                            local.push(u.to_bits());
                        }
                    }
                    if (b + 1).is_multiple_of(100) {
                        eprintln!("  ...block {}/{blocks}", b + 1);
                    }
                }
                hits.lock().unwrap().append(&mut local);
            });
        }
    });
    let mut hits = hits.into_inner().unwrap();
    hits.sort_unstable();
    hits
}

/// Prints the statistics of a tail sample and returns the number of
/// draws that are bit-identical to a previous one.
fn report(hits: &[u64], seconds: f64) -> usize {
    let duplicates = hits.len() - hits.chunk_by(|a, b| a == b).count();
    println!(
        "  {} deviates beyond {:.3}σ in {seconds:.0} s; {duplicates} duplicates",
        hits.len(),
        probit_tail(THRESHOLD),
    );
    let mut shown = 0;
    for run in hits.chunk_by(|a, b| a == b) {
        if run.len() > 1 {
            if shown == 3 {
                println!("    ...");
                break;
            }
            let u = f64::from_bits(run[0]);
            println!("    {:+.4}σ drawn {} times", probit_tail(u), run.len());
            shown += 1;
        }
    }
    duplicates
}

fn main() {
    let draws = std::env::args()
        .nth(1)
        .map_or(DRAWS, |s| s.parse::<f64>().expect("draws") as u64);
    let threads = std::thread::available_parallelism().map_or(4, usize::from);
    let expected_hits = draws as f64 * THRESHOLD;

    let expected_dups = expected_hits * expected_hits / 2.0 / (2f64.powi(31) - 1.0);
    println!("{draws:e} draws per converter on {threads} threads, seed {SEED}",);
    println!();

    println!("x/2^53 + zero guard:");
    let start = Instant::now();
    let hits = tail_draws(draws, |src| f64_53bits_guarded(|| src.next_u64()));
    let dups_53 = report(&hits, start.elapsed().as_secs_f64());

    println!("unif_01:");
    let start = Instant::now();
    let hits = tail_draws(draws, |src| unif_01(|| src.next_u64()));
    let dups_full = report(&hits, start.elapsed().as_secs_f64());

    assert_eq!(dups_full, 0);
    if expected_dups >= 10.0 {
        assert!(dups_53 as f64 >= expected_dups / 4.0);
    }
}
