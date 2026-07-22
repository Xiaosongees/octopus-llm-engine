//! 模型模块
//!
//! 包含 Transformer 模型组件和 ONNX 模型加载器。

pub mod transformer;
pub mod onnx_loader;

// 重新导出核心类型
pub use transformer::{
    TransformerConfig, TransformerEncoder, TransformerBlock,
    Attention, FeedForward, Embedding,
    ModelStats, LayerParamStats, compute_model_stats,
};
pub use onnx_loader::{OnnxLoader, OnnxLoadError, OnnxNode, OnnxModelInfo, OnnxAttribute};