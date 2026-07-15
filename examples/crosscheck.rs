//! Prints the bit patterns of a deterministic stream of round-down `f64`s
//! and `f32`s produced by the [`rand_float_rs::perfect`] module, for
//! cross-validation against the reference Go implementation of fp-rand
//! driven by the same SplitMix64 sequence.

use rand_float_rs::{perfect, sources::SplitMix64};

fn main() {
    let mut src = SplitMix64(0x123456789ABCDEF);
    for _ in 0..100000 {
        let x = perfect::f64_down(|| src.next_u64());
        println!("{:016x}", x.to_bits());
    }
    let mut src = SplitMix64(0xFEDCBA9876543210);
    for _ in 0..100000 {
        let x = perfect::f32_down(|| src.next_u64());
        println!("{:08x}", x.to_bits());
    }
}
