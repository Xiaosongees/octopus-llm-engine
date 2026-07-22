//! 基本推理示例
//!
//! 演示如何使用 Rust AI 推理引擎创建计算图、加载参数并执行推理。

use rust_ai_inference_engine::engine::graph::GraphBuilder;
use rust_ai_inference_engine::engine::inference::InferenceEngine;
use rust_ai_inference_engine::engine::tensor::TensorF32;
use rust_ai_inference_engine::model::transformer::{TransformerConfig, compute_model_stats};

fn main() -> anyhow::Result<()> {
    // 初始化日志
    env_logger::init();

    println!("=== Rust AI 推理引擎 - 基本使用示例 ===\n");

    // ---- 示例 1: 简单的多层感知机推理 ----
    println!("示例 1: 多层感知机 (MLP) 推理");
    println!("-".repeat(40));

    // 构建计算图: Input -> Dense(4->8) -> ReLU -> Dense(8->3) -> Output
    let mut builder = GraphBuilder::new();
    let input_id = builder.input("input", "f32", vec![1, 4])?;
    let dense1 = builder.dense("fc1", 4, 8, &input_id, true)?;
    let relu1 = builder.relu("relu1", &dense1)?;
    let dense2 = builder.dense("fc2", 8, 3, &relu1, true)?;
    let _output = builder.output("output", &dense2)?;

    let graph = builder.build()?;
    println!("计算图构建完成: {} 个节点", graph.node_count());

    // 创建推理引擎
    let mut engine = InferenceEngine::new_cpu();

    // 加载参数到后端
    // 注意：需要通过 backend 直接加载，这里用 CpuBackend
    println!("加载模型参数...");

    // 设置输入
    let input_data = vec![0.5, 1.0, 1.5, 2.0];
    let input_tensor = TensorF32::from_vec(input_data, &[1, 4])?;
    engine.set_input("input", input_tensor)?;
    engine.load_graph(graph);

    println!("输入形状: {:?}", engine.graph().unwrap().input_ids());
    println!("输出名称: {:?}", engine.graph().unwrap().output_ids());

    // 执行推理
    println!("执行推理...");
    match engine.run() {
        Ok(()) => {
            let output = engine.get_output(Some("output"))?;
            println!("输出形状: {:?}", output.shape());
            println!("推理成功！");
        }
        Err(e) => {
            println!("推理失败（预期中，参数未实际加载）: {}", e);
        }
    }

    // ---- 示例 2: Transformer 模型信息 ----
    println!("\n示例 2: Transformer 模型信息");
    println!("-".repeat(40));

    let config = TransformerConfig::small();
    println!("配置: d_model={}, n_heads={}, n_layers={}, d_ff={}",
        config.d_model, config.n_heads, config.n_layers, config.d_ff);

    let stats = compute_model_stats(&config);
    println!("总参数量: {} ({:.2}M)", stats.total_params,
        stats.total_params as f64 / 1e6);
    println!("内存占用: {:.2} MB", stats.total_params as f64 * 4.0 / (1024.0 * 1024.0));

    // ---- 示例 3: 张量操作 ----
    println!("\n示例 3: 张量操作演示");
    println!("-".repeat(40));

    let a = TensorF32::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
    let b = TensorF32::from_vec(vec![0.1, 0.2, 0.3, 0.4], &[2, 2])?;

    println!("张量 A (2x2): {:?}", a.to_vec());
    println!("张量 B (2x2): {:?}", b.to_vec());

    let c = a.matmul(&b)?;
    println!("A @ B = {:?}", c.to_vec());

    let r = a.relu();
    println!("ReLU(A) = {:?}", r.to_vec());

    let s = TensorF32::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4])?;
    let softmax = s.softmax()?;
    println!("Softmax([1,2,3,4]) = {:?}", softmax.to_vec());
    println!("Softmax 总和: {:.6}", softmax.to_vec().iter().sum::<f32>());

    println!("\n示例运行完毕。");

    Ok(())
}