//! Transformer 模型组件
//!
//! 实现 Transformer 架构的核心组件：
//! - 多头自注意力机制（Multi-Head Self-Attention）
//! - 前馈神经网络（Feed-Forward Network）
//! - Transformer Block
//! - 嵌入层（Embedding）

use crate::engine::graph::{GraphBuilder, GraphError, NodeType};
use crate::engine::tensor::{Tensor, TensorError, TensorF32};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Transformer 配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerConfig {
    /// 模型隐藏维度
    pub d_model: usize,
    /// 注意力头数
    pub n_heads: usize,
    /// Transformer 层数
    pub n_layers: usize,
    /// 前馈网络中间维度
    pub d_ff: usize,
    /// 词表大小
    pub vocab_size: usize,
    /// 最大序列长度
    pub max_seq_len: usize,
    /// Dropout 比率
    #[serde(default = "default_dropout")]
    pub dropout_rate: f64,
    /// Layer Norm epsilon
    #[serde(default = "default_epsilon")]
    pub layer_norm_epsilon: f64,
    /// 是否使用 GELU 激活（否则使用 ReLU）
    #[serde(default)]
    pub use_gelu: bool,
}

fn default_dropout() -> f64 {
    0.1
}

fn default_epsilon() -> f64 {
    1e-6
}

impl Default for TransformerConfig {
    fn default() -> Self {
        Self {
            d_model: 512,
            n_heads: 8,
            n_layers: 6,
            d_ff: 2048,
            vocab_size: 30000,
            max_seq_len: 512,
            dropout_rate: 0.1,
            layer_norm_epsilon: 1e-6,
            use_gelu: true,
        }
    }
}

impl TransformerConfig {
    /// 创建一个小型测试配置
    pub fn small() -> Self {
        Self {
            d_model: 64,
            n_heads: 4,
            n_layers: 2,
            d_ff: 128,
            vocab_size: 1000,
            max_seq_len: 32,
            dropout_rate: 0.0,
            layer_norm_epsilon: 1e-5,
            use_gelu: true,
        }
    }

    /// 每个注意力头的维度
    pub fn head_dim(&self) -> usize {
        assert_eq!(
            self.d_model % self.n_heads,
            0,
            "d_model 必须能被 n_heads 整除"
        );
        self.d_model / self.n_heads
    }

    /// 验证配置
    pub fn validate(&self) -> Result<(), String> {
        if self.d_model == 0 {
            return Err("d_model 不能为 0".to_string());
        }
        if self.n_heads == 0 {
            return Err("n_heads 不能为 0".to_string());
        }
        if self.d_model % self.n_heads != 0 {
            return Err(format!(
                "d_model ({}) 必须能被 n_heads ({}) 整除",
                self.d_model, self.n_heads
            ));
        }
        if self.n_layers == 0 {
            return Err("n_layers 不能为 0".to_string());
        }
        if self.d_ff == 0 {
            return Err("d_ff 不能为 0".to_string());
        }
        if self.vocab_size == 0 {
            return Err("vocab_size 不能为 0".to_string());
        }
        if self.max_seq_len == 0 {
            return Err("max_seq_len 不能为 0".to_string());
        }
        Ok(())
    }
}

/// 多头自注意力层
///
/// 实现 Scaled Dot-Product Attention 的多头版本：
/// Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
pub struct Attention {
    /// 注意力头数
    num_heads: usize,
    /// 每个头的维度
    head_dim: usize,
    /// Q/K/V 投影权重维度: (d_model, d_model)
    d_model: usize,
}

impl Attention {
    /// 创建新的注意力层
    pub fn new(d_model: usize, num_heads: usize) -> Result<Self, TensorError> {
        if d_model % num_heads != 0 {
            return Err(TensorError::OperationFailed(format!(
                "d_model ({}) 必须能被 num_heads ({}) 整除",
                d_model, num_heads
            )));
        }
        Ok(Self {
            num_heads,
            head_dim: d_model / num_heads,
            d_model,
        })
    }

    /// 注意力头数
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// 头维度
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// 执行缩放点积注意力
    ///
    /// # 参数
    ///
    /// - `query`: 查询张量 [batch, num_heads, seq_len, head_dim]
    /// - `key`: 键张量 [batch, num_heads, seq_len, head_dim]
    /// - `value`: 值张量 [batch, num_heads, seq_len, head_dim]
    /// - `mask`: 可选的注意力掩码
    ///
    /// # 返回
    ///
    /// 注意力输出和注意力权重
    pub fn scaled_dot_product_attention(
        query: &TensorF32,
        key: &TensorF32,
        value: &TensorF32,
        _mask: Option<&TensorF32>,
    ) -> Result<(TensorF32, TensorF32), TensorError> {
        // query, key, value: [batch, heads, seq_len, head_dim]
        let head_dim = query.shape().last().copied().unwrap_or(1) as f32;
        let scale = 1.0 / head_dim.sqrt();

        // Q @ K^T: [batch, heads, seq_len, seq_len]
        let key_t = key.transpose()?;
        let scores = query.matmul(&key_t)?;
        let scores = scores.mul_scalar(scale);

        // Softmax
        let attention_weights = scores.softmax()?;

        // attention_weights @ V: [batch, heads, seq_len, head_dim]
        let output = attention_weights.matmul(value)?;

        Ok((output, attention_weights))
    }

    /// 将注意力输出构建为计算图节点
    pub fn build_graph(
        builder: &mut GraphBuilder,
        layer_name: &str,
        input_id: &str,
        config: &TransformerConfig,
    ) -> Result<Vec<String>, GraphError> {
        let mut ids = Vec::new();

        // Q 投影
        let q_proj = builder.dense(
            format!("{}.q_proj", layer_name),
            config.d_model,
            config.d_model,
            input_id,
            true,
        )?;
        ids.push(q_proj.clone());

        // K 投影
        let k_proj = builder.dense(
            format!("{}.k_proj", layer_name),
            config.d_model,
            config.d_model,
            input_id,
            true,
        )?;
        ids.push(k_proj.clone());

        // V 投影
        let v_proj = builder.dense(
            format!("{}.v_proj", layer_name),
            config.d_model,
            config.d_model,
            input_id,
            true,
        )?;
        ids.push(v_proj.clone());

        // 多头注意力算子（在计算图中表示为一个节点）
        let attn_output = {
            let node = crate::engine::graph::Node::new(
                format!("{}.attention", layer_name),
                NodeType::MultiHeadAttention {
                    num_heads: config.n_heads,
                    head_dim: config.head_dim(),
                },
            )
            .with_input(&q_proj)
            .with_input(&k_proj)
            .with_input(&v_proj)
            .with_param(format!("{}.q_proj.weight", layer_name))
            .with_param(format!("{}.q_proj.bias", layer_name))
            .with_param(format!("{}.k_proj.weight", layer_name))
            .with_param(format!("{}.k_proj.bias", layer_name))
            .with_param(format!("{}.v_proj.weight", layer_name))
            .with_param(format!("{}.v_proj.bias", layer_name));
            builder.graph_add_node(node)?
            format!("{}.attention", layer_name)
        };
        ids.push(attn_output.clone());

        // 输出投影
        let out_proj = builder.dense(
            format!("{}.out_proj", layer_name),
            config.d_model,
            config.d_model,
            &attn_output,
            true,
        )?;
        ids.push(out_proj);

        Ok(ids)
    }
}

/// 前馈神经网络层
///
/// FFN(x) = activation(x @ W1 + b1) @ W2 + b2
pub struct FeedForward {
    /// 输入维度
    d_model: usize,
    /// 中间维度
    d_ff: usize,
    /// 是否使用 GELU
    use_gelu: bool,
}

impl FeedForward {
    /// 创建新的前馈层
    pub fn new(d_model: usize, d_ff: usize, use_gelu: bool) -> Self {
        Self {
            d_model,
            d_ff,
            use_gelu,
        }
    }
}

/// Transformer Block
///
/// 包含一个注意力层和一个前馈层，各自带有残差连接和层归一化。
pub struct TransformerBlock {
    /// 注意力层
    attention: Attention,
    /// 前馈层
    feed_forward: FeedForward,
    /// 层归一化 epsilon
    layer_norm_epsilon: f64,
}

impl TransformerBlock {
    /// 创建新的 Transformer Block
    pub fn new(config: &TransformerConfig) -> Result<Self, TensorError> {
        let attention = Attention::new(config.d_model, config.n_heads)?;
        let feed_forward = FeedForward::new(config.d_model, config.d_ff, config.use_gelu);

        Ok(Self {
            attention,
            feed_forward,
            layer_norm_epsilon: config.layer_norm_epsilon,
        })
    }

    /// 构建单个 Transformer Block 的计算图
    ///
    /// 结构: x -> LayerNorm -> Attention -> Add -> LayerNorm -> FFN -> Add -> output
    pub fn build_graph(
        builder: &mut GraphBuilder,
        layer_name: &str,
        input_id: &str,
        config: &TransformerConfig,
    ) -> Result<String, GraphError> {
        // Pre-LayerNorm 1
        let norm1 = builder.layer_norm(
            format!("{}.norm1", layer_name),
            input_id,
            vec![config.d_model],
            config.layer_norm_epsilon,
        )?;

        // Multi-Head Attention
        let attn_output = Attention::build_graph(builder, layer_name, &norm1, config)?;
        let attn_out_id = attn_output.last().unwrap().clone();

        // 残差连接 1
        let residual1 = builder.add(
            format!("{}.residual1", layer_name),
            input_id,
            &attn_out_id,
        )?;

        // Pre-LayerNorm 2
        let norm2 = builder.layer_norm(
            format!("{}.norm2", layer_name),
            &residual1,
            vec![config.d_model],
            config.layer_norm_epsilon,
        )?;

        // Feed-Forward Network
        let ffn = {
            // 第一层
            let fc1 = builder.dense(
                format!("{}.ffn.fc1", layer_name),
                config.d_model,
                config.d_ff,
                &norm2,
                true,
            )?;

            // 激活
            let act = if config.use_gelu {
                builder.gelu(format!("{}.ffn.act", layer_name), &fc1)?
            } else {
                builder.relu(format!("{}.ffn.act", layer_name), &fc1)?
            };

            // 第二层
            let fc2 = builder.dense(
                format!("{}.ffn.fc2", layer_name),
                config.d_ff,
                config.d_model,
                &act,
                true,
            )?;

            fc2
        };

        // 残差连接 2
        let residual2 = builder.add(
            format!("{}.residual2", layer_name),
            &residual1,
            &ffn,
        )?;

        Ok(residual2)
    }
}

/// 嵌入层
///
/// 将 token ID 映射为密集向量表示。
pub struct Embedding {
    /// 词表大小
    vocab_size: usize,
    /// 嵌入维度
    embed_dim: usize,
}

impl Embedding {
    /// 创建新的嵌入层
    pub fn new(vocab_size: usize, embed_dim: usize) -> Self {
        Self {
            vocab_size,
            embed_dim,
        }
    }

    /// 词表大小
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// 嵌入维度
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    /// 创建随机初始化的嵌入权重
    pub fn init_weights(&self) -> TensorF32 {
        let total = self.vocab_size * self.embed_dim;
        // 简单的随机初始化（Xavier uniform 近似）
        let scale = (2.0 / (self.vocab_size as f32 + self.embed_dim as f32)).sqrt();
        let data: Vec<f32> = (0..total)
            .map(|_| {
                // 简单伪随机：使用线性同余生成器
                let seed = (total.wrapping_mul(1103515245).wrapping_add(12345)) as f32;
                let normalized = (seed % 10000) as f32 / 5000.0 - 1.0;
                normalized * scale
            })
            .collect();
        TensorF32::from_vec(data, &[self.vocab_size, self.embed_dim])
            .expect("嵌入权重形状应有效")
    }

    /// 将嵌入层构建为计算图节点
    pub fn build_graph(
        builder: &mut GraphBuilder,
        name: &str,
        input_id: &str,
    ) -> Result<String, GraphError> {
        let node = crate::engine::graph::Node::new(
            name,
            NodeType::Embedding {
                vocab_size: self.vocab_size,
                embed_dim: self.embed_dim,
            },
        )
        .with_input(input_id)
        .with_param(format!("{}.weight", name));

        builder.graph_add_node(node)?;
        Ok(name.to_string())
    }
}

/// 完整的 Transformer 编码器
///
/// 由嵌入层和多个 Transformer Block 堆叠而成。
pub struct TransformerEncoder {
    /// 模型配置
    config: TransformerConfig,
    /// 嵌入层
    embedding: Embedding,
    /// Transformer Block 列表
    blocks: Vec<TransformerBlock>,
}

impl TransformerEncoder {
    /// 创建新的 Transformer 编码器
    pub fn new(config: TransformerConfig) -> Result<Self, TensorError> {
        config.validate().map_err(TensorError::OperationFailed)?;

        let embedding = Embedding::new(config.vocab_size, config.d_model);

        let mut blocks = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            let block = TransformerBlock::new(&config)?;
            blocks.push(block);
            log::debug!("创建 Transformer Block {}", i);
        }

        Ok(Self {
            config,
            embedding,
            blocks,
        })
    }

    /// 获取配置引用
    pub fn config(&self) -> &TransformerConfig {
        &self.config
    }

    /// 将完整编码器构建为计算图
    pub fn build_graph(
        &self,
        builder: &mut GraphBuilder,
        input_name: &str,
        output_name: &str,
    ) -> Result<(), GraphError> {
        // 输入节点
        let input_id = builder.input(input_name, "i32", vec![1, self.config.max_seq_len])?;

        // Token 嵌入
        let token_embed_id = self.embedding.build_graph(builder, "token_embedding", &input_id)?;

        // 位置嵌入（简化：使用可学习参数）
        let pos_embed_id = {
            let node = crate::engine::graph::Node::new(
                "pos_embedding",
                NodeType::Embedding {
                    vocab_size: self.config.max_seq_len,
                    embed_dim: self.config.d_model,
                },
            )
            .with_input("_position_ids") // 位置 ID（运行时注入）
            .with_param("pos_embedding.weight");

            // 需要先添加位置 ID 输入
            builder.input("_position_ids", "i32", vec![1, self.config.max_seq_len])?;
            builder.graph_add_node(node)?;
            "pos_embedding".to_string()
        };

        // 嵌入相加
        let embed_sum = builder.add("embed_sum", &token_embed_id, &pos_embed_id)?;

        // 堆叠 Transformer Blocks
        let mut current_id = embed_sum;
        for i in 0..self.config.n_layers {
            let block_name = format!("block_{}", i);
            current_id = TransformerBlock::build_graph(
                builder,
                &block_name,
                &current_id,
                &self.config,
            )?;
        }

        // 最终 LayerNorm
        let final_norm = builder.layer_norm(
            "final_norm",
            &current_id,
            vec![self.config.d_model],
            self.config.layer_norm_epsilon,
        )?;

        // 输出节点
        builder.output(output_name, &final_norm)?;

        Ok(())
    }

    /// 生成初始化参数
    ///
    /// 为模型中所有可学习参数生成随机初始值。
    pub fn init_params(&self) -> HashMap<String, TensorF32> {
        let mut params = HashMap::new();

        // Token 嵌入权重
        params.insert(
            "token_embedding.weight".to_string(),
            self.embedding.init_weights(),
        );

        // 位置嵌入权重
        let pos_embed = Embedding::new(self.config.max_seq_len, self.config.d_model);
        params.insert("pos_embedding.weight".to_string(), pos_embed.init_weights());

        // 各层参数
        for i in 0..self.config.n_layers {
            let prefix = format!("block_{}", i);

            // 注意力投影权重和偏置
            for proj in &["q_proj", "k_proj", "v_proj", "out_proj"] {
                let w_shape = vec![self.config.d_model, self.config.d_model];
                params.insert(
                    format!("{}.{}.weight", prefix, proj),
                    TensorF32::zeros(&w_shape),
                );
                params.insert(
                    format!("{}.{}.bias", prefix, proj),
                    TensorF32::zeros(&[self.config.d_model]),
                );
            }

            // LayerNorm 参数
            for ln in &["norm1", "norm2"] {
                params.insert(
                    format!("{}.{}.weight", prefix, ln),
                    TensorF32::ones(&[self.config.d_model]),
                );
                params.insert(
                    format!("{}.{}.bias", prefix, ln),
                    TensorF32::zeros(&[self.config.d_model]),
                );
            }

            // FFN 参数
            let ffn_prefix = format!("{}.ffn", prefix);
            params.insert(
                format!("{}.fc1.weight", ffn_prefix),
                TensorF32::zeros(&[self.config.d_ff, self.config.d_model]),
            );
            params.insert(
                format!("{}.fc1.bias", ffn_prefix),
                TensorF32::zeros(&[self.config.d_ff]),
            );
            params.insert(
                format!("{}.fc2.weight", ffn_prefix),
                TensorF32::zeros(&[self.config.d_model, self.config.d_ff]),
            );
            params.insert(
                format!("{}.fc2.bias", ffn_prefix),
                TensorF32::zeros(&[self.config.d_model]),
            );
        }

        // 最终 LayerNorm
        params.insert(
            "final_norm.weight".to_string(),
            TensorF32::ones(&[self.config.d_model]),
        );
        params.insert(
            "final_norm.bias".to_string(),
            TensorF32::zeros(&[self.config.d_model]),
        );

        params
    }
}

/// 模型参数统计
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelStats {
    /// 总参数量
    pub total_params: usize,
    /// 可训练参数量
    pub trainable_params: usize,
    /// 各层参数统计
    pub layer_stats: Vec<LayerParamStats>,
}

/// 单层参数统计
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerParamStats {
    /// 层名称
    pub name: String,
    /// 参数量
    pub params: usize,
    /// 形状
    pub shape: Vec<usize>,
}

/// 计算模型的参数统计
pub fn compute_model_stats(config: &TransformerConfig) -> ModelStats {
    let mut layer_stats = Vec::new();
    let mut total = 0usize;

    // Token 嵌入
    let embed_params = config.vocab_size * config.d_model;
    layer_stats.push(LayerParamStats {
        name: "token_embedding".to_string(),
        params: embed_params,
        shape: vec![config.vocab_size, config.d_model],
    });
    total += embed_params;

    // 位置嵌入
    let pos_params = config.max_seq_len * config.d_model;
    layer_stats.push(LayerParamStats {
        name: "pos_embedding".to_string(),
        params: pos_params,
        shape: vec![config.max_seq_len, config.d_model],
    });
    total += pos_params;

    // Transformer Blocks
    for i in 0..config.n_layers {
        let prefix = format!("block_{}", i);

        // Attention: Q, K, V, O 投影
        for proj in &["q_proj", "k_proj", "v_proj", "out_proj"] {
            let w_params = config.d_model * config.d_model;
            let b_params = config.d_model;
            layer_stats.push(LayerParamStats {
                name: format!("{}.{}.weight", prefix, proj),
                params: w_params,
                shape: vec![config.d_model, config.d_model],
            });
            layer_stats.push(LayerParamStats {
                name: format!("{}.{}.bias", prefix, proj),
                params: b_params,
                shape: vec![config.d_model],
            });
            total += w_params + b_params;
        }

        // LayerNorm 1, 2
        for ln in &["norm1", "norm2"] {
            let w = config.d_model;
            layer_stats.push(LayerParamStats {
                name: format!("{}.{}.weight", prefix, ln),
                params: w,
                shape: vec![config.d_model],
            });
            layer_stats.push(LayerParamStats {
                name: format!("{}.{}.bias", prefix, ln),
                params: w,
                shape: vec![config.d_model],
            });
            total += 2 * w;
        }

        // FFN
        let ffn_prefix = format!("{}.ffn", prefix);
        let fc1_w = config.d_ff * config.d_model;
        let fc1_b = config.d_ff;
        let fc2_w = config.d_model * config.d_ff;
        let fc2_b = config.d_model;

        layer_stats.push(LayerParamStats {
            name: format!("{}.fc1.weight", ffn_prefix),
            params: fc1_w,
            shape: vec![config.d_ff, config.d_model],
        });
        layer_stats.push(LayerParamStats {
            name: format!("{}.fc1.bias", ffn_prefix),
            params: fc1_b,
            shape: vec![config.d_ff],
        });
        layer_stats.push(LayerParamStats {
            name: format!("{}.fc2.weight", ffn_prefix),
            params: fc2_w,
            shape: vec![config.d_model, config.d_ff],
        });
        layer_stats.push(LayerParamStats {
            name: format!("{}.fc2.bias", ffn_prefix),
            params: fc2_b,
            shape: vec![config.d_model],
        });
        total += fc1_w + fc1_b + fc2_w + fc2_b;
    }

    // Final LayerNorm
    layer_stats.push(LayerParamStats {
        name: "final_norm.weight".to_string(),
        params: config.d_model,
        shape: vec![config.d_model],
    });
    layer_stats.push(LayerParamStats {
        name: "final_norm.bias".to_string(),
        params: config.d_model,
        shape: vec![config.d_model],
    });
    total += 2 * config.d_model;

    ModelStats {
        total_params: total,
        trainable_params: total, // 当前所有参数都可训练
        layer_stats,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transformer_config() {
        let config = TransformerConfig::small();
        assert_eq!(config.d_model, 64);
        assert_eq!(config.head_dim(), 16);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_transformer_config_validation() {
        let mut config = TransformerConfig::default();
        config.n_heads = 3;
        config.d_model = 10;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_attention_creation() {
        let attn = Attention::new(64, 4).unwrap();
        assert_eq!(attn.num_heads(), 4);
        assert_eq!(attn.head_dim(), 16);
    }

    #[test]
    fn test_embedding_weights() {
        let embed = Embedding::new(100, 32);
        let weights = embed.init_weights();
        assert_eq!(weights.shape(), &[100, 32]);
    }

    #[test]
    fn test_model_stats() {
        let config = TransformerConfig::small();
        let stats = compute_model_stats(&config);
        assert!(stats.total_params > 0);
        assert_eq!(stats.total_params, stats.trainable_params);
    }

    #[test]
    fn test_transformer_encoder_creation() {
        let config = TransformerConfig::small();
        let encoder = TransformerEncoder::new(config).unwrap();
        assert_eq!(encoder.blocks.len(), 2);
    }
}
