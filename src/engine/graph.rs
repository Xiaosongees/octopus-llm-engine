//! 计算图模块
//!
//! 实现计算图数据结构，支持算子节点定义和拓扑排序。

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// 计算图错误类型
#[derive(Debug, thiserror::Error)]
pub enum GraphError {
    #[error("节点未找到: {0}")]
    NodeNotFound(String),
    #[error("循环依赖检测到")]
    CycleDetected,
    #[error("无效的边连接: 从 {from} 到 {to}")]
    InvalidEdge { from: String, to: String },
    #[error("图构建失败: {0}")]
    BuildFailed(String),
}

/// 算子节点类型
///
/// 每种变体代表一种计算图中的操作节点。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum NodeType {
    /// 输入节点
    Input {
        name: String,
        dtype: String,
        shape: Vec<usize>,
    },
    /// 输出节点
    Output {
        name: String,
    },
    /// 全连接层 (y = x @ W + b)
    Dense {
        name: String,
        in_features: usize,
        out_features: usize,
        has_bias: bool,
    },
    /// 二维卷积层
    Conv2D {
        name: String,
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    },
    /// ReLU 激活
    ReLU,
    /// Softmax 激活
    Softmax {
        axis: i32, // -1 表示最后一个维度
    },
    /// 层归一化
    LayerNorm {
        normalized_shape: Vec<usize>,
        epsilon: f64,
    },
    /// 矩阵乘法
    MatMul,
    /// 逐元素加法
    Add,
    /// 形状重塑
    Reshape {
        target_shape: Vec<i64>, // -1 表示推断
    },
    /// GELU 激活
    GELU,
    /// 嵌入查找
    Embedding {
        vocab_size: usize,
        embed_dim: usize,
    },
    /// 多头注意力
    MultiHeadAttention {
        num_heads: usize,
        head_dim: usize,
    },
    /// DropOut
    Dropout {
        rate: f64,
    },
}

impl std::fmt::Display for NodeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NodeType::Input { name, .. } => write!(f, "Input({})", name),
            NodeType::Output { name } => write!(f, "Output({})", name),
            NodeType::Dense { name, .. } => write!(f, "Dense({})", name),
            NodeType::Conv2D { name, .. } => write!(f, "Conv2D({})", name),
            NodeType::ReLU => write!(f, "ReLU"),
            NodeType::Softmax { .. } => write!(f, "Softmax"),
            NodeType::LayerNorm { .. } => write!(f, "LayerNorm"),
            NodeType::MatMul => write!(f, "MatMul"),
            NodeType::Add => write!(f, "Add"),
            NodeType::Reshape { .. } => write!(f, "Reshape"),
            NodeType::GELU => write!(f, "GELU"),
            NodeType::Embedding { .. } => write!(f, "Embedding"),
            NodeType::MultiHeadAttention { .. } => write!(f, "MultiHeadAttention"),
            NodeType::Dropout { .. } => write!(f, "Dropout"),
        }
    }
}

/// 计算图中的节点
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    /// 节点唯一标识符
    pub id: String,
    /// 节点类型
    pub node_type: NodeType,
    /// 输入边的来源节点 ID 列表
    pub inputs: Vec<String>,
    /// 输出边的目标节点 ID 列表
    pub outputs: Vec<String>,
    /// 节点关联的参数名称（权重、偏置等）
    pub param_names: Vec<String>,
}

impl Node {
    /// 创建新节点
    pub fn new(id: impl Into<String>, node_type: NodeType) -> Self {
        Self {
            id: id.into(),
            node_type,
            inputs: Vec::new(),
            outputs: Vec::new(),
            param_names: Vec::new(),
        }
    }

    /// 添加输入连接
    pub fn with_input(mut self, input_id: impl Into<String>) -> Self {
        self.inputs.push(input_id.into());
        self
    }

    /// 添加参数名称
    pub fn with_param(mut self, name: impl Into<String>) -> Self {
        self.param_names.push(name.into());
        self
    }
}

/// 计算图
///
/// 持有所有节点和拓扑排序信息，用于驱动推理执行。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Graph {
    /// 所有节点
    nodes: Vec<Node>,
    /// 节点 ID 到索引的映射
    node_index: HashMap<String, usize>,
    /// 拓扑排序后的节点执行顺序
    execution_order: Vec<usize>,
    /// 输入节点 ID 列表
    pub(crate) input_ids: Vec<String>,
    /// 输出节点 ID 列表
    pub(crate) output_ids: Vec<String>,
}

impl Graph {
    /// 创建空计算图
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            node_index: HashMap::new(),
            execution_order: Vec::new(),
            input_ids: Vec::new(),
            output_ids: Vec::new(),
        }
    }

    /// 添加节点到图中
    pub fn add_node(&mut self, node: Node) -> Result<(), GraphError> {
        if self.node_index.contains_key(&node.id) {
            return Err(GraphError::BuildFailed(format!(
                "节点 ID '{}' 已存在",
                node.id
            )));
        }
        let idx = self.nodes.len();
        self.node_index.insert(node.id.clone(), idx);

        // 记录输入/输出节点
        match &node.node_type {
            NodeType::Input { .. } => {
                self.input_ids.push(node.id.clone());
            }
            NodeType::Output { .. } => {
                self.output_ids.push(node.id.clone());
            }
            _ => {}
        }

        self.nodes.push(node);
        Ok(())
    }

    /// 获取所有节点
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    /// 根据 ID 获取节点
    pub fn get_node(&self, id: &str) -> Option<&Node> {
        self.node_index.get(id).map(|&idx| &self.nodes[idx])
    }

    /// 获取拓扑排序后的执行顺序
    pub fn execution_order(&self) -> &[usize] {
        &self.execution_order
    }

    /// 获取输入节点 ID 列表
    pub fn input_ids(&self) -> &[String] {
        &self.input_ids
    }

    /// 获取输出节点 ID 列表
    pub fn output_ids(&self) -> &[String] {
        &self.output_ids
    }

    /// 获取节点数量
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// 执行拓扑排序（Kahn 算法）
    ///
    /// 计算节点的执行顺序，确保每个节点的所有输入都已计算完毕后才执行该节点。
    pub fn topological_sort(&mut self) -> Result<(), GraphError> {
        // 计算每个节点的入度
        let mut in_degree = vec![0usize; self.nodes.len()];
        for node in &self.nodes {
            for input_id in &node.inputs {
                if let Some(&idx) = self.node_index.get(input_id) {
                    // 建立从 input 节点到当前节点的边
                    in_degree[self.node_index[&node.id]] += 1;
                    let _ = idx; // 标记为已使用
                }
            }
        }

        // 重新计算入度：当前节点有多少输入节点
        let mut in_degree = vec![0usize; self.nodes.len()];
        for node in &self.nodes {
            in_degree[self.node_index[&node.id]] = node.inputs.len();
        }

        // 将入度为 0 的节点加入队列
        let mut queue: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
        for (i, &deg) in in_degree.iter().enumerate() {
            if deg == 0 {
                queue.push_back(i);
            }
        }

        let mut sorted = Vec::with_capacity(self.nodes.len());

        while let Some(node_idx) = queue.pop_front() {
            sorted.push(node_idx);

            // 遍历所有节点，找到以当前节点为输入的节点
            let current_id = self.nodes[node_idx].id.clone();
            for (i, node) in self.nodes.iter().enumerate() {
                if node.inputs.contains(&current_id) {
                    in_degree[i] -= 1;
                    if in_degree[i] == 0 {
                        queue.push_back(i);
                    }
                }
            }
        }

        if sorted.len() != self.nodes.len() {
            return Err(GraphError::CycleDetected);
        }

        self.execution_order = sorted;
        log::debug!("拓扑排序完成，执行顺序: {:?}", self.execution_order);
        Ok(())
    }

    /// 验证计算图的完整性
    pub fn validate(&self) -> Result<(), GraphError> {
        // 检查所有输入引用都存在
        for node in &self.nodes {
            for input_id in &node.inputs {
                if !self.node_index.contains_key(input_id) {
                    return Err(GraphError::InvalidEdge {
                        from: input_id.clone(),
                        to: node.id.clone(),
                    });
                }
            }
        }

        // 检查是否有输入和输出节点
        if self.input_ids.is_empty() {
            return Err(GraphError::BuildFailed("计算图缺少输入节点".to_string()));
        }
        if self.output_ids.is_empty() {
            return Err(GraphError::BuildFailed("计算图缺少输出节点".to_string()));
        }

        Ok(())
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

/// 计算图构建器
///
/// 提供流式 API 来方便地构建计算图。
pub struct GraphBuilder {
    graph: Graph,
}

impl GraphBuilder {
    /// 创建新的图构建器
    pub fn new() -> Self {
        Self {
            graph: Graph::new(),
        }
    }

    /// 直接添加一个 Node 到图中（用于非标准节点类型）
    ///
    /// 与 `input/output/dense` 等便捷方法不同，此方法接受一个已构造的 Node。
    pub fn graph_add_node(&mut self, node: Node) -> Result<(), GraphError> {
        self.graph.add_node(node)
    }

    /// 添加输入节点，返回节点 ID
    pub fn input(
        &mut self,
        name: impl Into<String>,
        dtype: impl Into<String>,
        shape: Vec<usize>,
    ) -> Result<String, GraphError> {
        let id = name.into();
        let node = Node::new(
            id.clone(),
            NodeType::Input {
                name: id.clone(),
                dtype: dtype.into(),
                shape,
            },
        );
        self.graph.add_node(node)?;
        Ok(id)
    }

    /// 添加输出节点，返回节点 ID
    pub fn output(&mut self, name: impl Into<String>, input: impl Into<String>) -> Result<String, GraphError> {
        let id = name.into();
        let input_id = input.into();
        let node = Node::new(id.clone(), NodeType::Output { name: id.clone() })
            .with_input(input_id);
        self.graph.add_node(node)?;
        Ok(id)
    }

    /// 添加全连接层
    pub fn dense(
        &mut self,
        name: impl Into<String>,
        in_features: usize,
        out_features: usize,
        input: impl Into<String>,
        has_bias: bool,
    ) -> Result<String, GraphError> {
        let id = name.into();
        let weight_name = format!("{}.weight", id);
        let mut node = Node::new(
            id.clone(),
            NodeType::Dense {
                name: id.clone(),
                in_features,
                out_features,
                has_bias,
            },
        )
        .with_input(input)
        .with_param(&weight_name);
        if has_bias {
            node = node.with_param(format!("{}.bias", id));
        }
        self.graph.add_node(node)?;
        Ok(id)
    }

    /// 添加 ReLU 激活
    pub fn relu(&mut self, name: impl Into<String>, input: impl Into<String>) -> Result<String, GraphError> {
        let id = name.into();
        let node = Node::new(id.clone(), NodeType::ReLU).with_input(input);
        self.graph.add_node(node)?;
        Ok(id)
    }

    /// 添加 Softmax
    pub fn softmax(&mut self, name: impl Into<String>, input: impl Into<String>, axis: i32) -> Result<String, GraphError> {
        let id = name.into();
        let node = Node::new(id.clone(), NodeType::Softmax { axis }).with_input(input);
        self.graph.add_node(node)?;
        Ok(id)
    }

    /// 添加矩阵乘法
    pub fn matmul(&mut self, name: impl Into<String>, input_a: impl Into<String>, input_b: impl Into<String>) -> Result<String, GraphError> {
        let id = name.into();
        let node = Node::new(id.clone(), NodeType::MatMul)
            .with_input(input_a)
            .with_input(input_b);
        self.graph.add_node(node)?;
        Ok(id)
    }

    /// 添加逐元素加法
    pub fn add(&mut self, name: impl Into<String>, input_a: impl Into<String>, input_b: impl Into<String>) -> Result<String, GraphError> {
        let id = name.into();
        let node = Node::new(id.clone(), NodeType::Add)
            .with_input(input_a)
            .with_input(input_b);
        self.graph.add_node(node)?;
        Ok(id)
    }

    /// 添加层归一化
    pub fn layer_norm(
        &mut self,
        name: impl Into<String>,
        input: impl Into<String>,
        normalized_shape: Vec<usize>,
        epsilon: f64,
    ) -> Result<String, GraphError> {
        let id = name.into();
        let node = Node::new(
            id.clone(),
            NodeType::LayerNorm {
                normalized_shape,
                epsilon,
            },
        )
        .with_input(input)
        .with_param(format!("{}.weight", id))
        .with_param(format!("{}.bias", id));
        self.graph.add_node(node)?;
        Ok(id)
    }

    /// 添加形状重塑
    pub fn reshape(&mut self, name: impl Into<String>, input: impl Into<String>, target_shape: Vec<i64>) -> Result<String, GraphError> {
        let id = name.into();
        let node = Node::new(id.clone(), NodeType::Reshape { target_shape }).with_input(input);
        self.graph.add_node(node)?;
        Ok(id)
    }

    /// 添加 Conv2D
    pub fn conv2d(
        &mut self,
        name: impl Into<String>,
        input: impl Into<String>,
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<String, GraphError> {
        let id = name.into();
        let node = Node::new(
            id.clone(),
            NodeType::Conv2D {
                name: id.clone(),
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
            },
        )
        .with_input(input)
        .with_param(format!("{}.weight", id))
        .with_param(format!("{}.bias", id));
        self.graph.add_node(node)?;
        Ok(id)
    }

    /// 添加 GELU 激活
    pub fn gelu(&mut self, name: impl Into<String>, input: impl Into<String>) -> Result<String, GraphError> {
        let id = name.into();
        let node = Node::new(id.clone(), NodeType::GELU).with_input(input);
        self.graph.add_node(node)?;
        Ok(id)
    }

    /// 构建计算图（执行拓扑排序和验证）
    pub fn build(mut self) -> Result<Graph, GraphError> {
        self.graph.validate()?;
        self.graph.topological_sort()?;
        Ok(self.graph)
    }
}

impl Default for GraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_simple_graph() {
        let mut builder = GraphBuilder::new();
        let input_id = builder.input("input", "f32", vec![1, 10]).unwrap();
        let dense_id = builder.dense("fc1", 10, 5, &input_id, true).unwrap();
        let relu_id = builder.relu("relu1", &dense_id).unwrap();
        let _output_id = builder.output("output", &relu_id).unwrap();

        let graph = builder.build().unwrap();
        assert_eq!(graph.node_count(), 4);
        assert_eq!(graph.execution_order().len(), 4);
    }

    #[test]
    fn test_topological_order() {
        let mut builder = GraphBuilder::new();
        let a = builder.input("a", "f32", vec![1, 4]).unwrap();
        let b = builder.input("b", "f32", vec![1, 4]).unwrap();
        let add_id = builder.add("add1", &a, &b).unwrap();
        let _output = builder.output("out", &add_id).unwrap();

        let graph = builder.build().unwrap();
        // a 和 b 应该在 add 之前
        let order = graph.execution_order();
        let a_idx = order.iter().position(|&i| graph.nodes()[i].id == "a").unwrap();
        let b_idx = order.iter().position(|&i| graph.nodes()[i].id == "b").unwrap();
        let add_idx = order.iter().position(|&i| graph.nodes()[i].id == "add1").unwrap();
        assert!(a_idx < add_idx);
        assert!(b_idx < add_idx);
    }

    #[test]
    fn test_cycle_detection() {
        // 这个测试通过手动构建图来测试循环检测
        let mut graph = Graph::new();
        graph
            .add_node(Node::new("a", NodeType::ReLU).with_input("b"))
            .unwrap();
        graph
            .add_node(Node::new("b", NodeType::ReLU).with_input("a"))
            .unwrap();

        let result = graph.topological_sort();
        assert!(result.is_err());
    }
}