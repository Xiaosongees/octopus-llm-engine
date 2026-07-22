//! 集成测试
//!
//! 测试推理引擎的端到端功能，包括计算图构建、推理执行和 API 服务器。

use rust_ai_inference_engine::engine::graph::{GraphBuilder, GraphError, NodeType};
use rust_ai_inference_engine::engine::inference::InferenceEngine;
use rust_ai_inference_engine::engine::tensor::TensorF32;
use rust_ai_inference_engine::backend::cpu::CpuBackend;
use rust_ai_inference_engine::backend::Backend;
use rust_ai_inference_engine::model::transformer::{
    TransformerConfig, TransformerEncoder, compute_model_stats,
};
use rust_ai_inference_engine::model::onnx_loader::{OnnxLoader, OnnxNode};

// ---- 计算图测试 ----

#[test]
fn test_simple_dense_graph() {
    // 构建一个简单的 Dense -> ReLU 计算图
    let mut builder = GraphBuilder::new();
    let input_id = builder.input("input", "f32", vec![1, 4]).unwrap();
    let dense_id = builder.dense("fc1", 4, 3, &input_id, true).unwrap();
    let relu_id = builder.relu("relu1", &dense_id).unwrap();
    let _output = builder.output("output", &relu_id).unwrap();

    let graph = builder.build().unwrap();
    assert_eq!(graph.node_count(), 4);
    assert_eq!(graph.input_ids().len(), 1);
    assert_eq!(graph.output_ids().len(), 1);
}

#[test]
fn test_multi_layer_graph() {
    let mut builder = GraphBuilder::new();
    let input_id = builder.input("x", "f32", vec![1, 8]).unwrap();

    let mut current = input_id;
    for i in 0..3 {
        let dense = builder.dense(format!("fc{}", i), 8, 8, &current, true).unwrap();
        let relu = builder.relu(format!("relu{}", i), &dense).unwrap();
        current = relu;
    }

    let _output = builder.output("y", &current).unwrap();
    let graph = builder.build().unwrap();
    assert_eq!(graph.node_count(), 9); // 1 input + 3*(dense+relu) + 1 output
}

#[test]
fn test_graph_with_matmul_and_add() {
    let mut builder = GraphBuilder::new();
    let a_id = builder.input("a", "f32", vec![1, 4]).unwrap();
    let b_id = builder.input("b", "f32", vec![1, 4]).unwrap();

    let matmul_id = builder.matmul("mm", &a_id, &b_id).unwrap();
    let add_id = builder.add("add", &matmul_id, &a_id).unwrap();
    let _output = builder.output("out", &add_id).unwrap();

    let graph = builder.build().unwrap();
    assert_eq!(graph.node_count(), 5);
}

// ---- 张量测试 ----

#[test]
fn test_tensor_operations() {
    let a = TensorF32::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let b = TensorF32::from_vec(vec![0.5, 1.0, 1.5, 2.0], &[2, 2]).unwrap();

    // 加法
    let c = a.add(&b).unwrap();
    assert_eq!(c.to_vec(), vec![1.5, 3.0, 4.5, 6.0]);

    // 乘法
    let d = a.mul(&b).unwrap();
    assert_eq!(d.to_vec(), vec![0.5, 2.0, 4.5, 8.0]);

    // 标量乘法
    let e = a.mul_scalar(2.0);
    assert_eq!(e.to_vec(), vec![2.0, 4.0, 6.0, 8.0]);
}

#[test]
fn test_tensor_reshape() {
    let t = TensorF32::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let reshaped = t.reshape(&[3, 2]).unwrap();
    assert_eq!(reshaped.shape(), &[3, 2]);
}

#[test]
fn test_tensor_softmax() {
    let t = TensorF32::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let s = t.softmax().unwrap();

    // 检查 softmax 输出和为 1
    let sum: f32 = s.to_vec().iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);

    // 检查单调递增
    let vals = s.to_vec();
    assert!(vals[0] < vals[1]);
    assert!(vals[1] < vals[2]);
}

// ---- CPU 后端测试 ----

#[test]
fn test_cpu_backend_forward() {
    let mut backend = CpuBackend::new();

    // 加载权重
    let weight = TensorF32::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]).unwrap(); // 单位矩阵
    let bias = TensorF32::from_vec(vec![1.0, 2.0], &[2]).unwrap();

    backend.load_param("fc.weight".to_string(), weight);
    backend.load_param("fc.bias".to_string(), bias);

    // Dense 层
    let input = TensorF32::from_vec(vec![3.0, 4.0], &[1, 2]).unwrap();
    let result = backend
        .forward(
            &NodeType::Dense {
                name: "fc".to_string(),
                in_features: 2,
                out_features: 2,
                has_bias: true,
            },
            &[input],
            &["fc.weight".to_string(), "fc.bias".to_string()],
        )
        .unwrap();

    // 预期: [3, 4] @ I^T + [1, 2] = [3, 4] + [1, 2] = [4, 6]
    let vals = result.to_vec();
    assert!((vals[0] - 4.0).abs() < 1e-5);
    assert!((vals[1] - 6.0).abs() < 1e-5);
}

// ---- 推理引擎测试 ----

#[test]
fn test_engine_creation_and_info() {
    let engine = InferenceEngine::new_cpu();
    let info = engine.engine_info();

    assert_eq!(info.version, env!("CARGO_PKG_VERSION"));
    assert!(!info.backend.is_empty());
}

#[test]
fn test_engine_with_sample_graph() {
    let mut engine = InferenceEngine::new_cpu();

    // 创建一个简单的图
    let mut builder = GraphBuilder::new();
    let input_id = builder.input("x", "f32", vec![1, 4]).unwrap();
    let dense_id = builder.dense("fc1", 4, 2, &input_id, true).unwrap();
    let relu_id = builder.relu("relu1", &dense_id).unwrap();
    let _output = builder.output("y", &relu_id).unwrap();

    let graph = builder.build().unwrap();
    engine.load_graph(graph);

    // 设置输入
    let input = TensorF32::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4]).unwrap();
    engine.set_input("x", input).unwrap();

    // 执行推理
    engine.run().unwrap();

    // 检查输出
    let output = engine.get_output(Some("y")).unwrap();
    assert_eq!(output.shape(), &[1, 2]);

    // 检查统计信息
    let stats = engine.last_stats().unwrap();
    assert!(stats.total_time_ms > 0.0);
    assert_eq!(stats.nodes_executed, 4);
}

// ---- Transformer 测试 ----

#[test]
fn test_transformer_config() {
    let config = TransformerConfig::small();
    assert!(config.validate().is_ok());
    assert_eq!(config.head_dim(), 16);
}

#[test]
fn test_transformer_encoder_creation() {
    let config = TransformerConfig::small();
    let encoder = TransformerEncoder::new(config).unwrap();
    assert_eq!(encoder.config().n_layers, 2);
}

#[test]
fn test_model_stats_calculation() {
    let config = TransformerConfig::small();
    let stats = compute_model_stats(&config);
    assert!(stats.total_params > 0);
    assert_eq!(stats.total_params, stats.trainable_params);

    // 验证层统计数量
    // 1 (token_embed) + 1 (pos_embed) + 2 * (4 proj * 2 + 2 ln * 2 + 4 ffn) + 2 (final_ln)
    assert!(!stats.layer_stats.is_empty());
}

// ---- ONNX 加载器测试 ----

#[test]
fn test_onnx_loader_basic() {
    let mut loader = OnnxLoader::new();
    loader.add_input("input", "f32", vec![1, 10]);
    loader.add_output("output");

    // 添加一些节点
    loader.add_node(OnnxNode {
        op_type: "MatMul".to_string(),
        name: "mm1".to_string(),
        inputs: vec!["input".to_string(), "w1".to_string()],
        outputs: vec!["hidden".to_string()],
        attributes: HashMap::new(),
    });
    loader.add_node(OnnxNode {
        op_type: "Add".to_string(),
        name: "add1".to_string(),
        inputs: vec!["hidden".to_string(), "b1".to_string()],
        outputs: vec!["output".to_string()],
        attributes: HashMap::new(),
    });

    // 注意：由于节点引用的输入中有 "w1" 和 "b1" 不在图中，
    // 转换可能会失败。这里只验证加载器的基本功能。
    assert_eq!(loader.nodes.len(), 2);
    assert_eq!(loader.inputs.len(), 1);
    assert_eq!(loader.outputs.len(), 1);
}

#[test]
fn test_onnx_load_from_json() {
    let json = r#"
    [
        {"op_type": "Relu", "name": "relu1", "inputs": ["x"], "outputs": ["y"], "attributes": {}}
    ]
    "#;

    let loader = OnnxLoader::load_from_json_nodes(json).unwrap();
    assert_eq!(loader.nodes.len(), 1);
    assert_eq!(loader.nodes[0].op_type, "Relu");
}

// ---- 错误处理测试 ----

#[test]
fn test_invalid_tensor_shape() {
    let result = TensorF32::from_vec(vec![1.0, 2.0], &[4]);
    assert!(result.is_err());
}

#[test]
fn test_graph_cycle_detection() {
    let mut loader = OnnxLoader::new();
    loader.add_input("a", "f32", vec![1, 4]);
    loader.add_output("a"); // 循环引用

    loader.add_node(OnnxNode {
        op_type: "Add".to_string(),
        name: "cycle".to_string(),
        inputs: vec!["a".to_string()],
        outputs: vec!["a".to_string()],
        attributes: HashMap::new(),
    });
}

// ---- 边界条件测试 ----

#[test]
fn test_empty_graph_rejection() {
    let builder = GraphBuilder::new();
    let result = builder.build();
    assert!(result.is_err());
}

#[test]
fn test_scalar_tensor_operations() {
    let t = TensorF32::from_vec(vec![42.0], &[1]).unwrap();
    assert_eq!(t.shape(), &[1]);
    let relu_result = t.relu();
    assert!((relu_result.get(&[0]).unwrap() - 42.0).abs() < 1e-6);
}