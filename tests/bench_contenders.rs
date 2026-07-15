//! Runs the sanity tests of the benchmark contenders in
//! `benches/uniform01.rs`: Criterion benches are built with
//! `harness = false`, so `#[test]` functions there would otherwise never
//! run. Compiling the bench file as a module of this integration test puts
//! them under the regular test harness.

#[path = "../benches/uniform01.rs"]
#[allow(dead_code)]
mod uniform01;
