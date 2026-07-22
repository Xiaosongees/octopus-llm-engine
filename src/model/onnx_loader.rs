//! ONNX 模型加载器（简化版）
//!
//! 支持加载 ONNX protobuf 格式模型，并将 ONNX 节点转换为内部 Graph 表示。
//!
//! 注意：当前实现为简化版本，支持基本的 ONNX 算子映射。
//! 完整的 ONNX 支持需要引入 prost 或 onnxruntime-rs 等库。

use crate::engine::graph::{Graph, GraphBuilder, GraphError, Node, NodeType};
use crate::engine::tensor::TensorF32;
use std::collections::HashMap;
use std::path::Path;

/// ONNX 加载器错误类型
#[derive(Debug, thiserror::Error)]
pub enum OnnxLoadError {
    #[error("文件读取失败: {0}")]
    FileReadFailed(String),
    #[error("ONNX 格式解析失败: {0}")]
    ParseError(String),
    #[error("不支持的 ONNX 算子: {0}")]
    UnsupportedOpType(String),
    #[error("节点转换失败: {0}")]
    NodeConversionFailed(String),
    #[error("计算图构建失败: {0}")]
    GraphBuildFailed(#[from] GraphError),
    #[error("I/O 错误: {0}")]
    IoError(#[from] std::io::Error),
}

/// ONNX 模型元信息
#[derive(Debug, Clone, serde::Serialize)]
pub struct OnnxModelInfo {
    /// 模型生产者
    pub producer_name: Option<String>,
    /// 模型版本
    pub model_version: Option<String>,
    /// ONNX IR 版本
    pub ir_version: Option<String>,
    /// 域
    pub domain: Option<String>,
    /// 图名称
    pub graph_name: Option<String>,
    /// 节点数量
    pub node_count: usize,
    /// 输入数量
    pub input_count: usize,
    /// 输出数量
    pub output_count: usize,
}

/// 简化的 ONNX 节点表示
///
/// 用于在不依赖 protobuf 库的情况下表示 ONNX 模型结构。
#[derive(Debug, Clone)]
pub struct OnnxNode {
    /// 算子类型（如 "MatMul", "Add", "Relu" 等）
    pub op_type: String,
    /// 节点名称
    pub name: String,
    /// 输入名称列表
    pub inputs: Vec<String>,
    /// 输出名称列表
    pub outputs: Vec<String>,
    /// 属性（键值对）
    pub attributes: HashMap<String, OnnxAttribute>,
}

/// ONNX 节点属性
#[derive(Debug, Clone)]
pub enum OnnxAttribute {
    Float(f32),
    Int(i64),
    String(String),
    Floats(Vec<f32>),
    Ints(Vec<i64>),
}

/// ONNX 模型加载器
///
/// 将 ONNX 模型转换为内部的 Graph 表示。
pub struct OnnxLoader {
    /// 加载的模型节点
    nodes: Vec<OnnxNode>,
    /// 模型输入信息
    inputs: Vec<(String, String, Vec<usize>)>, // (name, dtype, shape)
    /// 模型输出信息
    outputs: Vec<String>,
    /// 初始化器（权重等）
    initializers: HashMap<String, Vec<f32>>,
    /// 模型元信息
    model_info: OnnxModelInfo,
}

impl OnnxLoader {
    /// 创建新的空 ONNX 加载器
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            initializers: HashMap::new(),
            model_info: OnnxModelInfo {
                producer_name: None,
                model_version: None,
                ir_version: None,
                domain: None,
                graph_name: None,
                node_count: 0,
                input_count: 0,
                output_count: 0,
            },
        }
    }

    /// 从文件路径加载 ONNX 模型
    ///
    /// 当前实现为简化版本，实际生产中应使用 protobuf 解析 ONNX 格式。
    pub fn load_from_file(path: impl AsRef<Path>) -> Result<Self, OnnxLoadError> {
        let path = path.as_ref();
        log::info!("加载 ONNX 模型: {:?}", path);

        // 检查文件是否存在
        if !path.exists() {
            return Err(OnnxLoadError::FileReadFailed(format!(
                "文件不存在: {:?}",
                path
            )));
        }

        // 读取文件头，验证是否为 ONNX 格式
        let bytes = std::fs::read(path)?;
        if bytes.len() < 4 {
            return Err(OnnxLoadError::ParseError(
                "文件太小，不是有效的 ONNX 模型".to_string(),
            ));
        }

        // 检查 ONNX 魔数 (onnx3 的 protobuf 前缀)
        // ONNX 文件通常以 protobuf 编码开头
        // 这里做简单的格式检查
        log::warn!(
            "ONNX protobuf 解析需要 prost 库支持，当前使用简化模式"
        );

        let mut loader = Self::new();
        loader.model_info = OnnxModelInfo {
            graph_name: path
                .file_stem()
                .and_then(|s| s.to_str())
                .map(|s| s.to_string()),
            ..Default::default()
        };

        Ok(loader)
    }

    /// 手动添加节点（用于测试或自定义加载）
    pub fn add_node(&mut self, node: OnnxNode) {
        self.nodes.push(node);
    }

    /// 手动添加输入信息
    pub fn add_input(&mut self, name: impl Into<String>, dtype: impl Into<String>, shape: Vec<usize>) {
        self.inputs.push((name.into(), dtype.into(), shape));
    }

    /// 手动添加输出信息
    pub fn add_output(&mut self, name: impl Into<String>) {
        self.outputs.push(name.into());
    }

    /// 添加初始化器（权重）
    pub fn add_initializer(&mut self, name: impl Into<String>, data: Vec<f32>) {
        self.initializers.insert(name.into(), data);
    }

    /// 获取模型元信息
    pub fn model_info(&self) -> &OnnxModelInfo {
        &self.model_info
    }

    /// 将 ONNX 算子类型转换为内部 NodeType
    fn convert_op_type(
        &self,
        op_type: &str,
        name: &str,
        attributes: &HashMap<String, OnnxAttribute>,
    ) -> Result<NodeType, OnnxLoadError> {
        match op_type {
            // 矩阵运算
            "MatMul" => Ok(NodeType::MatMul),
            "Gemm" => {
                // Gemm = alpha * A @ B + beta * C
                Ok(NodeType::MatMul)
            }

            // 逐元素运算
            "Add" => Ok(NodeType::Add),
            "Sub" => Ok(NodeType::Add), // 简化：用 Add 替代
            "Mul" => Ok(NodeType::Add), // 简化：用 Add 替代

            // 激活函数
            "Relu" => Ok(NodeType::ReLU),
            "Gelu" => Ok(NodeType::GELU),
            "Sigmoid" => Ok(NodeType::GELU), // 简化
            "Tanh" => Ok(NodeType::GELU),    // 简化

            // 归一化
            "LayerNormalization" => {
                let epsilon = attributes
                    .get("epsilon")
                    .map(|attr| match attr {
                        OnnxAttribute::Float(f) => *f as f64,
                        _ => 1e-5,
                    })
                    .unwrap_or(1e-5);

                let normalized_shape = attributes
                    .get("axis")
                    .map(|attr| match attr {
                        OnnxAttribute::Int(i) => vec![*i as usize],
                        _ => vec![1],
                    })
                    .unwrap_or_default();

                Ok(NodeType::LayerNorm {
                    normalized_shape,
                    epsilon,
                })
            }
            "BatchNormalization" => Ok(NodeType::LayerNorm {
                normalized_shape: vec![1],
                epsilon: 1e-5,
            }),

            // Softmax
            "Softmax" => {
                let axis = attributes
                    .get("axis")
                    .map(|attr| match attr {
                        OnnxAttribute::Int(i) => *i as i32,
                        _ => -1,
                    })
                    .unwrap_or(-1);
                Ok(NodeType::Softmax { axis })
            }

            // 线性层
            "Conv" => {
                let kernel_shape = attributes
                    .get("kernel_shape")
                    .map(|attr| match attr {
                        OnnxAttribute::Ints(v) => {
                            if v.len() >= 2 {
                                (v[0] as usize, v[1] as usize)
                            } else {
                                (3, 3)
                            }
                        }
                        _ => (3, 3),
                    })
                    .unwrap_or((3, 3));

                let strides = attributes
                    .get("strides")
                    .map(|attr| match attr {
                        OnnxAttribute::Ints(v) => {
                            if v.len() >= 2 {
                                (v[0] as usize, v[1] as usize)
                            } else {
                                (1, 1)
                            }
                        }
                        _ => (1, 1),
                    })
                    .unwrap_or((1, 1));

                let pads = attributes
                    .get("pads")
                    .map(|attr| match attr {
                        OnnxAttribute::Ints(v) => {
                            if v.len() >= 2 {
                                (v[0] as usize, v[1] as usize)
                            } else {
                                (0, 0)
                            }
                        }
                        _ => (0, 0),
                    })
                    .unwrap_or((0, 0));

                Ok(NodeType::Conv2D {
                    name: name.to_string(),
                    in_channels: 0, // 需要从输入推断
                    out_channels: 0,
                    kernel_size: kernel_shape,
                    stride: strides,
                    padding: pads,
                })
            }

            // 形状操作
            "Reshape" => {
                let shape = attributes
                    .get("shape")
                    .map(|attr| match attr {
                        OnnxAttribute::Ints(v) => v.iter().map(|&x| x).collect(),
                        _ => vec![-1i64],
                    })
                    .unwrap_or(vec![-1]);
                Ok(NodeType::Reshape { target_shape: shape })
            }
            "Transpose" => Ok(NodeType::Reshape {
                target_shape: vec![-1],
            }), // 简化
            "Squeeze" => Ok(NodeType::Reshape {
                target_shape: vec![-1],
            }),
            "Unsqueeze" => Ok(NodeType::Reshape {
                target_shape: vec![-1],
            }),

            // 注意力（ONNX Attention 算子）
            "Attention" => {
                let num_heads = attributes
                    .get("num_heads")
                    .map(|attr| match attr {
                        OnnxAttribute::Int(i) => *i as usize,
                        _ => 8,
                    })
                    .unwrap_or(8);
                Ok(NodeType::MultiHeadAttention {
                    num_heads,
                    head_dim: 0, // 从模型推断
                })
            }

            // Embedding
            "Gather" => Ok(NodeType::Embedding {
                vocab_size: 0,
                embed_dim: 0,
            }),

            // Dropout
            "Dropout" => {
                let rate = attributes
                    .get("ratio")
                    .map(|attr| match attr {
                        OnnxAttribute::Float(f) => *f as f64,
                        _ => 0.1,
                    })
                    .unwrap_or(0.1);
                Ok(NodeType::Dropout { rate })
            }

            // 不支持的算子
            _ => Err(OnnxLoadError::UnsupportedOpType(op_type.to_string())),
        }
    }

    /// 将加载的 ONNX 模型转换为内部计算图
    pub fn into_graph(self) -> Result<Graph, OnnxLoadError> {
        let mut builder = GraphBuilder::new();

        // 添加输入节点
        for (name, dtype, shape) in &self.inputs {
            builder.input(name, dtype, shape.clone())?;
        }

        // 创建名称到节点 ID 的映射
        let mut name_map: HashMap<String, String> = HashMap::new();
        for (name, _, _) in &self.inputs {
            name_map.insert(name.clone(), name.clone());
        }

        // 转换 ONNX 节点
        for onnx_node in &self.nodes {
            let node_type =
                self.convert_op_type(&onnx_node.op_type, &onnx_node.name, &onnx_node.attributes)?;

            let mut node = Node::new(&onnx_node.name, node_type);

            // 添加输入连接
            for input_name in &onnx_node.inputs {
                if let Some(mapped_id) = name_map.get(input_name) {
                    node = node.with_input(mapped_id);
                } else {
                    // 可能是初始化器输入，跳过
                    log::debug!(
                        "跳过未映射的输入 '{}' 对于节点 '{}'",
                        input_name,
                        onnx_node.name
                    );
                }
            }

            // 构建节点 ID
            let node_id = &onnx_node.name;
            builder.graph_add_node(node)?;

            // 映射输出名称
            for output_name in &onnx_node.outputs {
                name_map.insert(output_name.clone(), node_id.clone());
            }
        }

        // 添加输出节点
        for output_name in &self.outputs {
            builder.output(output_name, name_map.get(output_name).map(|s| s.as_str()).unwrap_or(""))?;
        }

        // 构建计算图
        let graph = builder.build()?;
        log::info!(
            "ONNX 模型转换完成: {} 个节点",
            graph.node_count()
        );

        Ok(graph)
    }

    /// 从简化格式（JSON）加载 ONNX 节点
    ///
    /// 用于测试和开发，不依赖 protobuf。
    pub fn load_from_json_nodes(json_str: &str) -> Result<Self, OnnxLoadError> {
        let nodes: Vec<serde_json::Value> =
            serde_json::from_str(json_str).map_err(|e| OnnxLoadError::ParseError(e.to_string()))?;

        let mut loader = Self::new();
        for node_val in nodes {
            let op_type = node_val["op_type"]
                .as_str()
                .unwrap_or("Unknown")
                .to_string();
            let name = node_val["name"]
                .as_str()
                .unwrap_or("unnamed")
                .to_string();
            let inputs: Vec<String> = node_val["inputs"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default();
            let outputs: Vec<String> = node_val["outputs"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default();

            let mut attributes = HashMap::new();
            if let Some(attrs) = node_val["attributes"].as_object() {
                for (key, val) in attrs {
                    if let Some(f) = val.as_f64() {
                        attributes.insert(key.clone(), OnnxAttribute::Float(f as f32));
                    } else if let Some(i) = val.as_i64() {
                        attributes.insert(key.clone(), OnnxAttribute::Int(i));
                    } else if let Some(s) = val.as_str() {
                        attributes.insert(key.clone(), OnnxAttribute::String(s.to_string()));
                    }
                }
            }

            loader.add_node(OnnxNode {
                op_type,
                name,
                inputs,
                outputs,
                attributes,
            });
        }

        loader.model_info.node_count = loader.nodes.len();
        Ok(loader)
    }
}

impl Default for OnnxLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for OnnxModelInfo {
    fn default() -> Self {
        Self {
            producer_name: None,
            model_version: None,
            ir_version: None,
            domain: None,
            graph_name: None,
            node_count: 0,
            input_count: 0,
            output_count: 0,
        }
    }
}

/// ONNX 算子到内部算子的支持映射
pub fn supported_onnx_ops() -> Vec<&'static str> {
    vec![
        "MatMul",
        "Gemm",
        "Add",
        "Relu",
        "Gelu",
        "Sigmoid",
        "Tanh",
        "LayerNormalization",
        "BatchNormalization",
        "Softmax",
        "Conv",
        "Reshape",
        "Transpose",
        "Squeeze",
        "Unsqueeze",
        "Attention",
        "Gather",
        "Dropout",
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_onnx_loader_creation() {
        let loader = OnnxLoader::new();
        assert!(loader.nodes.is_empty());
    }

    #[test]
    fn test_op_type_conversion() {
        let loader = OnnxLoader::new();

        assert!(matches!(
            loader.convert_op_type("MatMul", "test", &HashMap::new()),
            Ok(NodeType::MatMul)
        ));
        assert!(matches!(
            loader.convert_op_type("Relu", "test", &HashMap::new()),
            Ok(NodeType::ReLU)
        ));
        assert!(matches!(
            loader.convert_op_type("Add", "test", &HashMap::new()),
            Ok(NodeType::Add)
        ));
    }

    #[test]
    fn test_unsupported_op() {
        let loader = OnnxLoader::new();
        let result = loader.convert_op_type("CustomOp", "test", &HashMap::new());
        assert!(result.is_err());
    }

    #[test]
    fn test_supported_ops() {
        let ops = supported_onnx_ops();
        assert!(ops.contains(&"MatMul"));
        assert!(ops.contains(&"Relu"));
        assert!(ops.contains(&"Softmax"));
    }

    #[test]
    fn test_load_from_json() {
        let json = r#"
        [
            {"op_type": "MatMul", "name": "mm1", "inputs": ["a", "b"], "outputs": ["c"], "attributes": {}},
            {"op_type": "Add", "name": "add1", "inputs": ["c", "d"], "outputs": ["e"], "attributes": {}},
            {"op_type": "Relu", "name": "relu1", "inputs": ["e"], "outputs": ["f"], "attributes": {}}
        ]
        "#;

        let loader = OnnxLoader::load_from_json_nodes(json).unwrap();
        assert_eq!(loader.nodes.len(), 3);
        assert_eq!(loader.model_info.node_count, 3);
    }
}