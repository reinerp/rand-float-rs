//! Bit-identical extreme deviates: 53-bit scaling quantizes the
//! Gaussian tail.
//!
//! Section 3.1 of the survey [“Gaussian Random Number Generators”] by
//! Thomas, Luk, Leong and Villasenor (ACM Comput. Surv. 39(4), 2007)
//! shows (Fig. 7) that scaling an integer by 2⁻ʷ produces floats that
//! inherit the coarse, equispaced resolution of fixed point near zero —
//! precisely where the transformations turning uniforms into normal
//! deviates are singular. Under CDF inversion (their section 2.1) the
//! deepest uniforms become the most extreme deviates, so the defect
//! surfaces at the tail: below −5.03σ (u < 2⁻²²) the ubiquitous 53-bit
//! scaling ([`standard::f64_53bits`]) can produce only 2³¹ distinct
//! deviates, all on a lattice of pitch 2⁻⁵³.
//!
//! This is observable in an honest simulation, with any seed. The
//! survey estimates 10⁸ deviates per second per machine, 10¹⁷ per
//! cluster-sized simulation; the 1.2·10¹² draws per converter run here
//! (a few minutes on a parallel machine) collect about 286,000 deviates
//! beyond −5.03σ. On the 2³¹-point lattice the birthday effect makes
//! about 19 of them bit-for-bit copies of another, supposedly
//! independent, extreme event — the kind of anomaly any
//! peaks-over-threshold analysis of the simulation output would
//! surface — and the greatest common divisor of the whole tail sample
//! is exactly 2⁻⁵³. [`uniform::unif_01`] reaches every representable
//! double with the probability of the reals rounding down to it (the
//! property section 3.1 credits Matlab’s `rand` with): the same run
//! yields zero duplicates (about 3·10⁻⁶ expected) and no lattice.
//!
//! The duplicates are the early symptom of the same quantization whose
//! extreme form is tail truncation: 53-bit scaling cannot produce any
//! deviate below Φ⁻¹(2⁻⁵³) ≈ −8.21σ (the survey’s target is 10σ), while
//! complete coverage reaches Φ⁻¹(2⁻¹⁰⁷⁴) ≈ −38.5σ. Truncation only
//! becomes visible beyond ~10¹⁶ draws; the lattice is visible today.
//!
//! Run with `cargo run --release --example gaussian_tail`; an optional
//! argument overrides the number of draws per converter (e.g. `1e10`
//! for a quick, seconds-long run — too short to accumulate duplicates).
//!
//! [“Gaussian Random Number Generators”]: https://doi.org/10.1145/1287620.1287622
//! [`standard::f64_53bits`]: rand_float::standard::f64_53bits
//! [`uniform::unif_01`]: rand_float::uniform::unif_01

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use rand_float::standard::f64_53bits;
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

/// A Weyl generator, as [`rand_float::sources::Weyl`] but with a
/// per-stream increment: the cheapest possible source, one add per
/// word. The increment must vary across blocks: within a stream the
/// word following a tail hit w is always w + increment, so with a
/// single shared increment the sub-2⁻⁵³ bits [`f64_full`] fills in
/// would be the same constant for every hit, and the full conversion
/// would inherit the duplicates instead of removing them.
///
/// [`f64_full`]: rand_float::pekkizen::f64_full
struct Weyl {
    state: u64,
    /// Must be odd.
    incr: u64,
}

impl Weyl {
    #[inline(always)]
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(self.incr);
        self.state
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
/// [`BLOCK`]-sized work unit runs its own [`Weyl`] stream, with state
/// and increment scrambled from the block index, so distinct blocks are
/// statistically independent and the sample does not depend on how
/// blocks are scheduled onto threads.
fn tail_draws(draws: u64, convert: impl Fn(&mut Weyl) -> f64 + Sync) -> Vec<u64> {
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
                    let mut src = Weyl {
                        state: mix(SEED ^ (2 * b)),
                        incr: mix(SEED ^ (2 * b + 1)) | 1,
                    };
                    for _ in 0..BLOCK.min(draws - b * BLOCK) {
                        let u = convert(&mut src);
                        if u < THRESHOLD {
                            local.push(u.to_bits());
                        }
                    }
                    if (b + 1) % 100 == 0 {
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
        "  {} deviates beyond {:.3}σ in {seconds:.0} s; {duplicates} bit-identical duplicates",
        hits.len(),
        probit_tail(THRESHOLD),
    );
    let mut shown = 0;
    for run in hits.chunk_by(|a, b| a == b) {
        if run.len() > 1 && shown < 3 {
            let u = f64::from_bits(run[0]);
            println!(
                "    {:+.4}σ drawn {} times (u = {u:e})",
                probit_tail(u),
                run.len()
            );
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
