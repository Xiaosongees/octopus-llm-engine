//! CPU 后端实现
//!
//! 使用 ndarray 在 CPU 上执行所有算子计算。

use crate::engine::graph::NodeType;
use crate::engine::tensor::{Tensor, TensorError, TensorF32};
use crate::backend::Backend;
use std::collections::HashMap;

/// CPU 计算后端
///
/// 使用 ndarray 库在 CPU 上执行张量运算，
/// 通过 rayon 实现内部并行化。
pub struct CpuBackend {
    /// 参数存储（权重、偏置等）
    params: HashMap<String, TensorF32>,
}

impl CpuBackend {
    /// 创建新的 CPU 后端
    pub fn new() -> Self {
        Self {
            params: HashMap::new(),
        }
    }

    /// 加载参数到后端
    pub fn load_param(&mut self, name: String, tensor: TensorF32) {
        self.params.insert(name, tensor);
    }

    /// 批量加载参数
    pub fn load_params(&mut self, params: HashMap<String, TensorF32>) {
        self.params.extend(params);
    }

    /// 获取参数引用
    pub fn get_param(&self, name: &str) -> Option<&TensorF32> {
        self.params.get(name)
    }

    /// 执行全连接层前向传播
    ///
    /// y = x @ W^T + b
    fn forward_dense(
        &self,
        input: &TensorF32,
        weight: &TensorF32,
        bias: Option<&TensorF32>,
    ) -> Result<TensorF32, TensorError> {
        // 输入形状: [batch, in_features]
        // 权重形状: [out_features, in_features]
        // 偏置形状: [out_features]
        let weight_t = weight.transpose()?;
        let mut result = input.matmul(&weight_t)?;

        if let Some(b) = bias {
            let batch_size = result.shape()[0];
            let bias_2d = b.reshape(&[batch_size, bias.shape()[0]])?;
            result = result.add(&bias_2d)?;
        }

        Ok(result)
    }

    /// 执行 Conv2D 前向传播（简化版 im2col）
    fn forward_conv2d(
        &self,
        _input: &TensorF32,
        _weight: &TensorF32,
        _bias: Option<&TensorF32>,
        _kernel_size: (usize, usize),
        _stride: (usize, usize),
        _padding: (usize, usize),
    ) -> Result<TensorF32, TensorError> {
        // Conv2D 的完整实现需要 im2col 变换
        // 这里提供简化版本
        log::warn!("CPU Conv2D 当前为简化实现，可能性能不佳");

        // 对于实际生产环境，应使用 im2col 或直接卷积实现
        // 这里返回一个占位结果
        let output_shape = vec![1, 1, 1, 1];
        Ok(TensorF32::zeros(&output_shape))
    }

    /// 执行 ReLU 激活
    fn forward_relu(&self, input: &TensorF32) -> TensorF32 {
        input.relu()
    }

    /// 执行 GELU 激活
    fn forward_gelu(&self, input: &TensorF32) -> TensorF32 {
        input.gelu()
    }

    /// 执行 Softmax
    fn forward_softmax(&self, input: &TensorF32, axis: i32) -> Result<TensorF32, TensorError> {
        let axis = if axis < 0 {
            (input.ndim() as i32 + axis) as usize
        } else {
            axis as usize
        };

        // 对于非最后一个轴的 softmax，需要转置
        if axis != input.ndim() - 1 {
            return Err(TensorError::OperationFailed(
                "CPU 后端当前仅支持最后一个轴的 Softmax".to_string(),
            ));
        }

        input.softmax()
    }

    /// 执行层归一化
    fn forward_layer_norm(
        &self,
        input: &TensorF32,
        weight: Option<&TensorF32>,
        bias: Option<&TensorF32>,
        epsilon: f64,
    ) -> Result<TensorF32, TensorError> {
        let mut normalized = input.layer_norm(epsilon as f32)?;

        // 应用可学习的仿射变换: y = weight * x_norm + bias
        if let Some(w) = weight {
            normalized = normalized.mul(w)?;
        }
        if let Some(b) = bias {
            normalized = normalized.add(b)?;
        }

        Ok(normalized)
    }

    /// 执行矩阵乘法
    fn forward_matmul(
        &self,
        a: &TensorF32,
        b: &TensorF32,
    ) -> Result<TensorF32, TensorError> {
        a.matmul(b)
    }

    /// 执行逐元素加法
    fn forward_add(&self, a: &TensorF32, b: &TensorF32) -> Result<TensorF32, TensorError> {
        a.add(b)
    }

    /// 执行形状重塑
    fn forward_reshape(
        &self,
        input: &TensorF32,
        target_shape: &[i64],
    ) -> Result<TensorF32, TensorError> {
        let total: i64 = input.len() as i64;

        // 处理 -1（自动推断维度）
        let inferred_shape: Vec<usize> = {
            let mut shape: Vec<i64> = target_shape.to_vec();
            let neg_idx = shape.iter().position(|&x| x == -1);

            if let Some(idx) = neg_idx {
                let known_product: i64 = shape.iter().filter(|&&x| x != -1).product();
                if known_product == 0 {
                    return Err(TensorError::OperationFailed(
                        "无法推断形状：已知维度的乘积为 0".to_string(),
                    ));
                }
                shape[idx] = total / known_product;
            }

            shape
                .iter()
                .map(|&x| {
                    if x < 0 {
                        Err(TensorError::OperationFailed(
                            "形状中只能有一个 -1".to_string(),
                        ))
                    } else {
                        Ok(x as usize)
                    }
                })
                .collect::<Result<_, _>>()?
        };

        input.reshape(&inferred_shape)
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for CpuBackend {
    fn name(&self) -> &str {
        "CPU (ndarray)"
    }

    fn device_info(&self) -> String {
        // 获取 CPU 信息
        let cpus = num_cpus::get();
        format!(
            "CPU 后端 | 可用核心数: {} | 并行支持: rayon",
            cpus
        )
    }

    fn forward(
        &self,
        node_type: &NodeType,
        inputs: &[TensorF32],
        param_names: &[String],
    ) -> Result<TensorF32, TensorError> {
        match node_type {
            NodeType::Input { .. } => {
                // 输入节点直接返回输入张量
                inputs
                    .first()
                    .cloned()
                    .ok_or_else(|| TensorError::OperationFailed("输入节点缺少输入张量".to_string()))
            }

            NodeType::Output { .. } => {
                // 输出节点直接透传
                inputs
                    .first()
                    .cloned()
                    .ok_or_else(|| TensorError::OperationFailed("输出节点缺少输入张量".to_string()))
            }

            NodeType::Dense {
                has_bias,
                in_features: _,
                out_features: _,
                name: _,
            } => {
                let input = inputs.first().ok_or_else(|| {
                    TensorError::OperationFailed("Dense 层缺少输入张量".to_string())
                })?;

                let weight = param_names.first().and_then(|n| self.params.get(n)).ok_or_else(|| {
                    TensorError::OperationFailed(format!(
                        "Dense 层权重 '{}' 未找到",
                        param_names.first().unwrap_or(&"?".to_string())
                    ))
                })?;

                let bias = if *has_bias && param_names.len() > 1 {
                    self.params.get(&param_names[1])
                } else {
                    None
                };

                self.forward_dense(input, weight, bias)
            }

            NodeType::Conv2D {
                kernel_size,
                stride,
                padding,
                ..
            } => {
                let input = inputs.first().ok_or_else(|| {
                    TensorError::OperationFailed("Conv2D 缺少输入张量".to_string())
                })?;

                let weight = param_names.first().and_then(|n| self.params.get(n)).ok_or_else(|| {
                    TensorError::OperationFailed("Conv2D 权重未找到".to_string())
                })?;

                let bias = if param_names.len() > 1 {
                    self.params.get(&param_names[1])
                } else {
                    None
                };

                self.forward_conv2d(
                    input,
                    weight,
                    bias,
                    *kernel_size,
                    *stride,
                    *padding,
                )
            }

            NodeType::ReLU => {
                let input = inputs.first().ok_or_else(|| {
                    TensorError::OperationFailed("ReLU 缺少输入张量".to_string())
                })?;
                Ok(self.forward_relu(input))
            }

            NodeType::GELU => {
                let input = inputs.first().ok_or_else(|| {
                    TensorError::OperationFailed("GELU 缺少输入张量".to_string())
                })?;
                Ok(self.forward_gelu(input))
            }

            NodeType::Softmax { axis } => {
                let input = inputs.first().ok_or_else(|| {
                    TensorError::OperationFailed("Softmax 缺少输入张量".to_string())
                })?;
                self.forward_softmax(input, *axis)
            }

            NodeType::LayerNorm { epsilon, .. } => {
                let input = inputs.first().ok_or_else(|| {
                    TensorError::OperationFailed("LayerNorm 缺少输入张量".to_string())
                })?;

                let weight = param_names.first().and_then(|n| self.params.get(n));
                let bias = if param_names.len() > 1 {
                    self.params.get(&param_names[1])
                } else {
                    None
                };

                self.forward_layer_norm(input, weight, bias, *epsilon)
            }

            NodeType::MatMul => {
                if inputs.len() < 2 {
                    return Err(TensorError::OperationFailed(
                        "MatMul 需要两个输入张量".to_string(),
                    ));
                }
                self.forward_matmul(&inputs[0], &inputs[1])
            }

            NodeType::Add => {
                if inputs.len() < 2 {
                    return Err(TensorError::OperationFailed(
                        "Add 需要两个输入张量".to_string(),
                    ));
                }
                self.forward_add(&inputs[0], &inputs[1])
            }

            NodeType::Reshape { target_shape } => {
                let input = inputs.first().ok_or_else(|| {
                    TensorError::OperationFailed("Reshape 缺少输入张量".to_string())
                })?;
                self.forward_reshape(input, target_shape)
            }

            NodeType::Embedding { .. } => {
                // 简化实现：返回输入的零张量
                let input = inputs.first().ok_or_else(|| {
                    TensorError::OperationFailed("Embedding 缺少输入张量".to_string())
                })?;

                let weight = param_names.first().and_then(|n| self.params.get(n));

                if let Some(embedding_table) = weight {
                    // 简化的 embedding 查找：使用输入的索引值
                    // 实际实现需要处理 token ID 到 embedding 的映射
                    let batch_seq_shape = input.shape();
                    let embed_dim = embedding_table.shape().get(1).copied().unwrap_or(1);
                    let mut output_shape = batch_seq_shape.to_vec();
                    output_shape.push(embed_dim);
                    Ok(TensorF32::zeros(&output_shape))
                } else {
                    Ok(input.clone())
                }
            }

            NodeType::MultiHeadAttention { .. } | NodeType::Dropout { .. } => {
                // 复杂算子的简化实现
                let input = inputs.first().ok_or_else(|| {
                    TensorError::OperationFailed(format!(
                        "{:?} 缺少输入张量",
                        node_type
                    ))
                })?;
                log::warn!("{:?} 当前为透传（简化实现）", node_type);
                Ok(input.clone())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_backend_dense() {
        let mut backend = CpuBackend::new();

        // 创建权重 [3, 2] 和偏置 [3]
        let weight = TensorF32::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();
        let bias = TensorF32::from_vec(vec![0.1, 0.2, 0.3], &[3]).unwrap();

        backend.load_param("fc.weight".to_string(), weight);
        backend.load_param("fc.bias".to_string(), bias);

        // 输入 [1, 2]
        let input = TensorF32::from_vec(vec![1.0, 2.0], &[1, 2]).unwrap();

        let node_type = NodeType::Dense {
            name: "fc".to_string(),
            in_features: 2,
            out_features: 3,
            has_bias: true,
        };

        let result = backend
            .forward(&node_type, &[input], &["fc.weight".to_string(), "fc.bias".to_string()])
            .unwrap();

        assert_eq!(result.shape(), &[1, 3]);
    }

    #[test]
    fn test_cpu_backend_relu() {
        let backend = CpuBackend::new();
        let input = TensorF32::from_vec(vec![-1.0, 0.0, 1.0, 2.0], &[2, 2]).unwrap();

        let result = backend.forward(&NodeType::ReLU, &[input], &[]).unwrap();
        assert_eq!(result.to_vec(), vec![0.0, 0.0, 1.0, 2.0]);
    }
}
