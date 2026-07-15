//! Prints the bit patterns of a deterministic stream of round-down `f64`s
//! and `f32`s, for cross-validation against the reference Go implementation
//! of fp-rand driven by the same SplitMix64 sequence.

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

fn main() {
    let mut state = 0x123456789ABCDEFu64;
    for _ in 0..100000 {
        let x = perfect_fp_rs::f64_down(|| splitmix64(&mut state));
        println!("{:016x}", x.to_bits());
    }
    let mut state = 0xFEDCBA9876543210u64;
    for _ in 0..100000 {
        let x = perfect_fp_rs::f32_down(|| splitmix64(&mut state));
        println!("{:08x}", x.to_bits());
    }
}
