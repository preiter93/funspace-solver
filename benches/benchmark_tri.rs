use criterion::Criterion;
use criterion::{criterion_group, criterion_main};
use funspace_solver::tri::Tri202;
use rand::{thread_rng, Rng};

const SIZES: [usize; 3] = [128, 264, 512];

pub fn bench_tri_202(c: &mut Criterion) {
    let mut group = c.benchmark_group("Tri202");
    group.significance_level(0.1).sample_size(10);
    let mut rng = thread_rng();
    for n in SIZES {
        // Initialize
        let l2: Vec<f64> = (0..n - 2).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let u2: Vec<f64> = (0..n - 2).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let d0: Vec<f64> = (0..n).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let dma = Tri202::new(l2.clone(), d0.clone(), u2.clone());
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

criterion_group!(benches, bench_tri_202);
criterion_main!(benches);
