//! # Rust AI 推理引擎
//!
//! 一个高性能的 AI 推理引擎，使用 Rust 编写。
//!
//! ## 功能特性
//!
//! - 基于计算图的推理执行
//! - 多后端支持（CPU，CUDA 桩）
//! - Transformer 模型组件
//! - ONNX 模型加载（简化版）
//! - HTTP REST API 服务
//! - 并行算子执行
//!
//! ## 模块结构
//!
//! - `engine`: 核心张量、计算图和推理引擎
//! - `backend`: 计算后端（CPU、CUDA）
//! - `model`: 模型组件（Transformer、ONNX 加载）
//! - `api`: HTTP REST API 服务

// 核心引擎模块
pub mod engine;
// 计算后端模块
pub mod backend;
// 模型模块
pub mod model;
// HTTP API 模块
pub mod api;

// 重新导出常用类型
pub use engine::{
    Tensor, TensorError, TensorF32, TensorF64,
    Graph, GraphBuilder, Node, NodeType, GraphError,
    InferenceEngine, InferenceError, InferenceStats, EngineInfo,
};
pub use backend::{Backend, BackendType};
pub use model::{
    TransformerConfig, TransformerEncoder,
    OnnxLoader, OnnxLoadError,
};