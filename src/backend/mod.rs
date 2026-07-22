//! 后端模块
//!
//! 定义计算后端的统一接口，提供 CPU 和 CUDA（桩）实现。

pub mod cpu;
pub mod cuda;

use crate::engine::graph::NodeType;
use crate::engine::tensor::{TensorError, TensorF32};

/// 计算后端 trait
///
/// 定义了推理引擎与硬件之间的统一接口。
/// 不同的后端（CPU、CUDA 等）实现此 trait 以支持不同的计算设备。
pub trait Backend: Send + Sync {
    /// 获取后端名称
    fn name(&self) -> &str;

    /// 获取设备信息字符串
    fn device_info(&self) -> String;

    /// 执行算子前向传播
    ///
    /// # 参数
    ///
    /// - `node_type`: 算子类型
    /// - `inputs`: 输入张量列表
    /// - `param_names`: 参数名称列表（用于查找权重等）
    ///
    /// # 返回
    ///
    /// 计算结果张量
    fn forward(
        &self,
        node_type: &NodeType,
        inputs: &[TensorF32],
        param_names: &[String],
    ) -> Result<TensorF32, TensorError>;
}

/// 后端类型枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BackendType {
    /// CPU 后端（默认）
    #[default]
    Cpu,
    /// CUDA GPU 后端
    Cuda,
}

impl std::fmt::Display for BackendType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendType::Cpu => write!(f, "cpu"),
            BackendType::Cuda => write!(f, "cuda"),
        }
    }
}

impl std::str::FromStr for BackendType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "cpu" => Ok(BackendType::Cpu),
            "cuda" | "gpu" => Ok(BackendType::Cuda),
            _ => Err(format!(
                "不支持的后端类型: '{}'。可选: cpu, cuda",
                s
            )),
        }
    }
}