use criterion::{criterion_group, criterion_main, Criterion};
use index_simulation_tool::{DistanceMetric, HighDimVector, Index, NaiveIndex};

fn benchmark_index<I: Index>(c: &mut Criterion) {
    let mut group = c.benchmark_group("Indexing");

    let mut index = I::new(DistanceMetric::Euclidean);
    let vectors = (0..1000)
        .map(|i| HighDimVector::new(vec![i as f64, i as f64, i as f64]))
        .collect::<Vec<_>>();
    for vector in vectors {
        index.add_vector(vector);
    }
    let query = HighDimVector::new(vec![500.0, 500.0, 500.0]);

    group.bench_function("find_nearest", |b| {
        b.iter(|| {
            let nearest = index.find_nearest(&query);
            assert!(nearest.is_some())
        });
    });

    group.finish();
}

criterion_group!(benches, benchmark_index<NaiveIndex>);
criterion_main!(benches);
