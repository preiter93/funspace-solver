use criterion::Criterion;
use criterion::{criterion_group, criterion_main};
use funspace_solver::hepta::Hepta2101234;
use rand::{thread_rng, Rng};

const SIZES: [usize; 3] = [128, 264, 512];

pub fn bench_hepta_2101234(c: &mut Criterion) {
    let mut group = c.benchmark_group("Hepta2101234");
    group.significance_level(0.1).sample_size(10);
    let mut rng = thread_rng();
    for n in SIZES {
        // Initialize
        let l2: Vec<f64> = (0..n - 2).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let l1: Vec<f64> = (0..n - 1).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let u1: Vec<f64> = (0..n - 1).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let u2: Vec<f64> = (0..n - 2).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let u3: Vec<f64> = (0..n - 3).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let u4: Vec<f64> = (0..n - 4).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let d0: Vec<f64> = (0..n).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let dma = Hepta2101234::new(l2, l1, d0, u1, u2, u3, u4);
        let b: Vec<f64> = (0..n).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let name = format!("Size: {} (Checked)  ", n);
        group.bench_function(&name, |bench| {
            bench.iter(|| {
                let _ = dma.solve(&b);
            })
        });
        let name = format!("Size: {} (Unchecked)", n);
        group.bench_function(&name, |bench| {
            bench.iter(|| {
                let _ = dma.solve_unchecked(&b);
            })
        });
    }
    group.finish()
}

criterion_group!(benches, bench_hepta_2101234);
criterion_main!(benches);
