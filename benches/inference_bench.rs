//! 推理引擎基准测试
//!
//! 使用 criterion 对核心推理操作进行性能测试。

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use rust_ai_inference_engine::engine::graph::GraphBuilder;
use rust_ai_inference_engine::engine::inference::InferenceEngine;
use rust_ai_inference_engine::engine::tensor::TensorF32;

/// 准备测试用张量数据
fn prepare_tensor_data(size: usize) -> Vec<f32> {
    (0..size).map(|i| (i % 1000) as f32 / 1000.0).collect()
}

/// 基准测试：张量创建
fn bench_tensor_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_creation");
    for size in [64, 256, 1024, 4096] {
        let data = prepare_tensor_data(size);
        group.bench_with_input(
            BenchmarkId::new("from_vec", size),
            &(data, size),
            |b, (data, size)| {
                b.iter(|| {
                    TensorF32::from_vec(data.clone(), &[*size]).unwrap()
                });
            },
        );
    }
    group.finish();
}

/// 基准测试：张量激活函数
fn bench_tensor_activations(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_activations");

    for size in [256, 1024, 4096] {
        let data = prepare_tensor_data(size);
        let tensor = TensorF32::from_vec(data, &[size]).unwrap();

        group.bench_with_input(
            BenchmarkId::new("relu", size),
            &tensor,
            |b, tensor| {
                b.iter(|| black_box(tensor.relu()));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("gelu", size),
            &tensor,
            |b, tensor| {
                b.iter(|| black_box(tensor.gelu()));
            },
        );
    }
    group.finish();
}

/// 基准测试：矩阵乘法
fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul");

    for n in [16, 32, 64, 128] {
        let a_data = prepare_tensor_data(n * n);
        let b_data = prepare_tensor_data(n * n);
        let a = TensorF32::from_vec(a_data, &[n, n]).unwrap();
        let b = TensorF32::from_vec(b_data, &[n, n]).unwrap();

        group.bench_with_input(
            BenchmarkId::new("matmul", n),
            &(a, b),
            |b, (a, b)| {
                b.iter(|| black_box(a.matmul(b).unwrap()));
            },
        );
    }
    group.finish();
}

/// 基准测试：完整推理图执行
fn bench_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("inference");

    for (name, hidden_size, layers) in [
        ("small_2l", 32, 2),
        ("medium_4l", 64, 4),
        ("large_8l", 128, 8),
    ] {
        // 构建计算图
        let mut builder = GraphBuilder::new();
        let input_id = builder
            .input("input", "f32", vec![1, hidden_size])
            .unwrap();
        let mut current = input_id;

        for i in 0..layers {
            let dense_id = builder
                .dense(
                    format!("fc_{}", i),
                    hidden_size,
                    hidden_size,
                    &current,
                    true,
                )
                .unwrap();
            let relu_id = builder.relu(format!("relu_{}", i), &dense_id).unwrap();
            current = relu_id;
        }

        let _output_id = builder.output("output", &current).unwrap();
        let graph = builder.build().unwrap();

        // 准备引擎
        let mut engine = InferenceEngine::new_cpu();
        engine.load_graph(graph);

        let input_data: Vec<f32> = (0..hidden_size).map(|i| i as f32 / hidden_size as f32).collect();
        let input_tensor = TensorF32::from_vec(input_data, &[1, hidden_size]).unwrap();
        engine.set_input("input", input_tensor).unwrap();

        // 只执行一次 run 来 warm up
        let _ = engine.run();
        let input_data2: Vec<f32> = (0..hidden_size).map(|i| i as f32 / hidden_size as f32).collect();
        let input_tensor2 = TensorF32::from_vec(input_data2, &[1, hidden_size]).unwrap();

        group.bench_with_input(
            BenchmarkId::new(name, hidden_size * layers),
            &input_tensor2,
            |b, input_tensor| {
                b.iter(|| {
                    engine.set_input("input", input_tensor.clone()).unwrap();
                    engine.run().unwrap();
                });
            },
        );
    }
    group.finish();
}

/// 基准测试：Softmax
fn bench_softmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("softmax");

    for size in [128, 512, 2048] {
        let data = prepare_tensor_data(size);
        let tensor = TensorF32::from_vec(data, &[size]).unwrap();

        group.bench_with_input(
            BenchmarkId::new("softmax_1d", size),
            &tensor,
            |b, tensor| {
                b.iter(|| black_box(tensor.softmax().unwrap()));
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_tensor_creation,
    bench_tensor_activations,
    bench_matmul,
    bench_softmax,
    bench_inference,
);

criterion_main!(benches);