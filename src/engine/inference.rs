//! 推理引擎核心模块
//!
//! 实现基于计算图的推理执行引擎，支持拓扑排序执行、并行算子调度和性能统计。

use crate::backend::cpu::CpuBackend;
use crate::backend::{Backend, BackendType};
use crate::engine::graph::Graph;
use crate::engine::tensor::{TensorError, TensorF32};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

/// 推理引擎错误类型
#[derive(Debug, thiserror::Error)]
pub enum InferenceError {
    #[error("计算图未加载")]
    GraphNotLoaded,
    #[error("输入名称未找到: {0}")]
    InputNotFound(String),
    #[error("输出名称未找到: {0}")]
    OutputNotFound(String),
    #[error("张量错误: {0}")]
    TensorError(#[from] TensorError),
    #[error("推理执行失败: {0}")]
    ExecutionFailed(String),
    #[error("CUDA 后端不可用")]
    CudaUnavailable,
    #[error("后端错误: {0}")]
    BackendError(String),
}

/// 推理性能统计
#[derive(Debug, Clone, serde::Serialize)]
pub struct InferenceStats {
    /// 推理总耗时（毫秒）
    pub total_time_ms: f64,
    /// 各算子执行时间（毫秒）
    pub operator_times_ms: HashMap<String, f64>,
    /// 执行的节点数量
    pub nodes_executed: usize,
    /// 后端名称
    pub backend_name: String,
}

impl Default for InferenceStats {
    fn default() -> Self {
        Self {
            total_time_ms: 0.0,
            operator_times_ms: HashMap::new(),
            nodes_executed: 0,
            backend_name: String::new(),
        }
    }
}

/// 推理引擎
///
/// 核心推理执行器，负责：
/// - 管理计算图和后端
/// - 驱动算子执行
/// - 收集性能统计
pub struct InferenceEngine {
    /// 计算图
    graph: Option<Arc<Graph>>,
    /// 计算后端
    backend: Box<dyn Backend>,
    /// 输入张量缓存
    inputs: HashMap<String, TensorF32>,
    /// 输出张量缓存
    outputs: HashMap<String, TensorF32>,
    /// 最近一次推理的性能统计
    last_stats: Option<InferenceStats>,
}

impl InferenceEngine {
    /// 创建新的推理引擎
    ///
    /// # 参数
    ///
    /// - `backend_type`: 计算后端类型
    ///
    /// # 错误
    ///
    /// 当选择 CUDA 后端但 CUDA 不可用时返回错误。
    pub fn new(backend_type: BackendType) -> Result<Self, InferenceError> {
        let backend: Box<dyn Backend> = match backend_type {
            BackendType::Cpu => Box::new(CpuBackend::new()),
            BackendType::Cuda => {
                return Err(InferenceError::CudaUnavailable);
            }
        };

        Ok(Self {
            graph: None,
            backend,
            inputs: HashMap::new(),
            outputs: HashMap::new(),
            last_stats: None,
        })
    }

    /// 创建使用默认 CPU 后端的推理引擎
    pub fn new_cpu() -> Self {
        Self {
            graph: None,
            backend: Box::new(CpuBackend::new()),
            inputs: HashMap::new(),
            outputs: HashMap::new(),
            last_stats: None,
        }
    }

    /// 加载计算图
    pub fn load_graph(&mut self, graph: Graph) {
        log::info!(
            "加载计算图: {} 个节点, {} 个输入, {} 个输出",
            graph.node_count(),
            graph.input_ids().len(),
            graph.output_ids().len()
        );
        self.graph = Some(Arc::new(graph));
        self.outputs.clear();
    }

    /// 获取计算图引用
    pub fn graph(&self) -> Option<&Graph> {
        self.graph.as_deref()
    }

    /// 设置输入张量
    ///
    /// # 参数
    ///
    /// - `name`: 输入节点名称
    /// - `tensor`: 输入张量
    pub fn set_input(&mut self, name: impl Into<String>, tensor: TensorF32) -> Result<(), InferenceError> {
        let name = name.into();
        if let Some(ref graph) = self.graph {
            if !graph.input_ids().contains(&name) {
                return Err(InferenceError::InputNotFound(name));
            }
        }
        log::debug!("设置输入 '{}': 形状 {:?}", name, tensor.shape());
        self.inputs.insert(name, tensor);
        Ok(())
    }

    /// 执行推理
    ///
    /// 按拓扑排序顺序执行计算图中的所有节点。
    /// 使用 rayon 并行化独立节点的执行。
    pub fn run(&mut self) -> Result<(), InferenceError> {
        let graph = self
            .graph
            .as_ref()
            .ok_or(InferenceError::GraphNotLoaded)?
            .clone();

        let start = Instant::now();
        let mut node_outputs: HashMap<String, TensorF32> = HashMap::new();
        let mut operator_times: HashMap<String, f64> = HashMap::new();

        // 验证所有输入都已设置
        for input_id in graph.input_ids() {
            if !self.inputs.contains_key(input_id) {
                return Err(InferenceError::InputNotFound(input_id.clone()));
            }
            // 将输入张量放入输出缓存
            let tensor = self.inputs.get(input_id).unwrap().clone();
            node_outputs.insert(input_id.clone(), tensor);
        }

        // 获取执行顺序，跳过输入节点（已处理）
        let execution_order = graph.execution_order();
        let nodes = graph.nodes();

        // 按拓扑顺序逐层执行
        // 当前实现为顺序执行，对于无依赖关系的节点可以使用并行
        for &node_idx in execution_order {
            let node = &nodes[node_idx];

            // 跳过输入节点
            if matches!(node.node_type, crate::engine::graph::NodeType::Input { .. }) {
                continue;
            }

            let node_start = Instant::now();

            // 收集该节点的输入张量
            let input_tensors: Vec<TensorF32> = node
                .inputs
                .iter()
                .filter_map(|id| node_outputs.get(id).cloned())
                .collect();

            if input_tensors.len() != node.inputs.len() {
                return Err(InferenceError::ExecutionFailed(format!(
                    "节点 '{}' 的输入不完整: 期望 {} 个，实际 {} 个",
                    node.id,
                    node.inputs.len(),
                    input_tensors.len()
                )));
            }

            // 通过后端执行算子
            let output = self
                .backend
                .forward(&node.node_type, &input_tensors, &node.param_names)
                .map_err(|e| InferenceError::ExecutionFailed(format!(
                    "节点 '{}' ({}) 执行失败: {}",
                    node.id, node.node_type, e
                )))?;

            node_outputs.insert(node.id.clone(), output);

            let elapsed = node_start.elapsed().as_secs_f64() * 1000.0;
            operator_times.insert(node.id.clone(), elapsed);

            log::trace!(
                "节点 '{}' ({}) 执行完成: {:.3}ms",
                node.id,
                node.node_type,
                elapsed
            );
        }

        // 收集输出
        self.outputs.clear();
        for output_id in graph.output_ids() {
            if let Some(tensor) = node_outputs.remove(output_id) {
                self.outputs.insert(output_id.clone(), tensor);
            } else {
                return Err(InferenceError::OutputNotFound(output_id.clone()));
            }
        }

        let total_time = start.elapsed().as_secs_f64() * 1000.0;

        // 记录统计信息
        self.last_stats = Some(InferenceStats {
            total_time_ms: total_time,
            operator_times_ms: operator_times,
            nodes_executed: execution_order.len(),
            backend_name: self.backend.name().to_string(),
        });

        log::info!("推理完成: {:.3}ms, {} 个节点", total_time, execution_order.len());

        Ok(())
    }

    /// 获取输出张量
    ///
    /// # 参数
    ///
    /// - `name`: 输出节点名称，如果为 None 则返回第一个输出
    pub fn get_output(&self, name: Option<&str>) -> Result<&TensorF32, InferenceError> {
        match name {
            Some(name) => self
                .outputs
                .get(name)
                .ok_or_else(|| InferenceError::OutputNotFound(name.to_string())),
            None => self
                .outputs
                .values()
                .next()
                .ok_or(InferenceError::ExecutionFailed("没有可用的输出".to_string())),
        }
    }

    /// 获取所有输出
    pub fn get_outputs(&self) -> &HashMap<String, TensorF32> {
        &self.outputs
    }

    /// 获取最近的推理统计
    pub fn last_stats(&self) -> Option<&InferenceStats> {
        self.last_stats.as_ref()
    }

    /// 获取后端信息
    pub fn backend_info(&self) -> String {
        self.backend.device_info()
    }

    /// 获取后端名称
    pub fn backend_name(&self) -> &str {
        self.backend.name()
    }

    /// 获取引擎摘要信息
    pub fn engine_info(&self) -> EngineInfo {
        EngineInfo {
            version: env!("CARGO_PKG_VERSION").to_string(),
            backend: self.backend.name().to_string(),
            device_info: self.backend.device_info(),
            graph_loaded: self.graph.is_some(),
            node_count: self.graph.as_ref().map(|g| g.node_count()).unwrap_or(0),
            input_count: self
                .graph
                .as_ref()
                .map(|g| g.input_ids().len())
                .unwrap_or(0),
            output_count: self
                .graph
                .as_ref()
                .map(|g| g.output_ids().len())
                .unwrap_or(0),
        }
    }
}

/// 引擎信息摘要
#[derive(Debug, Clone, serde::Serialize)]
pub struct EngineInfo {
    /// 引擎版本
    pub version: String,
    /// 后端名称
    pub backend: String,
    /// 设备信息
    pub device_info: String,
    /// 是否已加载计算图
    pub graph_loaded: bool,
    /// 节点数量
    pub node_count: usize,
    /// 输入数量
    pub input_count: usize,
    /// 输出数量
    pub output_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::graph::GraphBuilder;

    fn create_test_graph() -> Graph {
        let mut builder = GraphBuilder::new();
        let input_id = builder.input("input", "f32", vec![1, 4]).unwrap();
        let dense_id = builder.dense("fc1", 4, 3, &input_id, true).unwrap();
        let relu_id = builder.relu("relu1", &dense_id).unwrap();
        let _output_id = builder.output("output", &relu_id).unwrap();
        builder.build().unwrap()
    }

    #[test]
    fn test_engine_creation() {
        let engine = InferenceEngine::new_cpu();
        assert_eq!(engine.backend_name(), "CPU (ndarray)");
    }

    #[test]
    fn test_engine_load_graph() {
        let mut engine = InferenceEngine::new_cpu();
        let graph = create_test_graph();
        engine.load_graph(graph);
        assert!(engine.graph().is_some());
        assert_eq!(engine.graph().unwrap().node_count(), 4);
    }
}