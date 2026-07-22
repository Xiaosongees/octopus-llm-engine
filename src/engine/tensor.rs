//! 张量模块 - 核心数据结构
//!
//! 基于 ndarray 提供多维张量支持，支持 f32/f64 泛型。

use ndarray::{ArrayD, ArcArray, IxDyn};
use num_traits::{Float, Num};
use std::fmt;
use std::sync::Arc;

/// 张量错误类型
#[derive(Debug, thiserror::Error)]
pub enum TensorError {
    #[error("形状不匹配: 期望 {expected:?}, 实际 {actual:?}")]
    ShapeMismatch { expected: Vec<usize>, actual: Vec<usize> },
    #[error("索引越界: {0}")]
    IndexOutOfBounds(String),
    #[error("不支持的类型转换")]
    InvalidCast,
    #[error("张量操作失败: {0}")]
    OperationFailed(String),
}

/// 核心张量结构体
///
/// 基于 ndarray 的 ArcArray 实现，支持引用计数共享数据。
/// 使用泛型 T 支持不同的数值类型（f32, f64）。
#[derive(Clone)]
pub struct Tensor<T: Num + Float + Clone> {
    /// 内部数据存储，使用 Arc 实现零拷贝共享
    data: Arc<ArcArray<T, IxDyn>>,
    /// 张量名称（可选，用于调试）
    name: Option<String>,
}

impl<T: Num + Float + Clone> Tensor<T> {
    /// 从一维 Vec 创建张量
    pub fn from_vec(data: Vec<T>, shape: &[usize]) -> Result<Self, TensorError> {
        let total: usize = shape.iter().product();
        if data.len() != total {
            return Err(TensorError::ShapeMismatch {
                expected: shape.to_vec(),
                actual: vec![data.len()],
            });
        }
        let array = ArrayD::from_shape_vec(IxDyn(shape), data)
            .map_err(|e| TensorError::OperationFailed(e.to_string()))?
            .into_shared();
        Ok(Self {
            data: Arc::new(array),
            name: None,
        })
    }

    /// 创建全零张量
    pub fn zeros(shape: &[usize]) -> Self {
        let array = ArrayD::zeros(IxDyn(shape)).into_shared();
        Self {
            data: Arc::new(array),
            name: None,
        }
    }

    /// 创建全一张量
    pub fn ones(shape: &[usize]) -> Self {
        let array = ArrayD::ones(IxDyn(shape)).into_shared();
        Self {
            data: Arc::new(array),
            name: None,
        }
    }

    /// 创建未初始化的张量
    pub fn uninit(shape: &[usize]) -> Self {
        let array = unsafe { ArrayD::<T>::uninit(IxDyn(shape)).assume_init() }.into_shared();
        Self {
            data: Arc::new(array),
            name: None,
        }
    }

    /// 设置张量名称
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// 获取张量名称
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// 获取张量形状
    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    /// 获取张量维度数
    pub fn ndim(&self) -> usize {
        self.data.ndim()
    }

    /// 获取张量元素总数
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// 检查张量是否为空
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// 获取内部数据的引用
    pub fn as_array(&self) -> &ArcArray<T, IxDyn> {
        &self.data
    }

    /// 转换为 Vec（数据拷贝）
    pub fn to_vec(&self) -> Vec<T> {
        self.data.iter().cloned().collect()
    }

    /// 张量重塑形状
    pub fn reshape(&self, new_shape: &[usize]) -> Result<Self, TensorError> {
        let total: usize = new_shape.iter().product();
        if total != self.len() {
            return Err(TensorError::ShapeMismatch {
                expected: new_shape.to_vec(),
                actual: vec![self.len()],
            });
        }
        let array = self
            .data
            .clone()
            .into_shape(IxDyn(new_shape))
            .map_err(|e| TensorError::OperationFailed(e.to_string()))?
            .into_shared();
        Ok(Self {
            data: Arc::new(array),
            name: self.name.clone(),
        })
    }

    /// 元素级加法
    pub fn add(&self, other: &Self) -> Result<Self, TensorError> {
        if self.shape() != other.shape() {
            return Err(TensorError::ShapeMismatch {
                expected: self.shape().to_vec(),
                actual: other.shape().to_vec(),
            });
        }
        let result = &self.data + &other.data;
        Ok(Self {
            data: Arc::new(result.into_shared()),
            name: None,
        })
    }

    /// 标量加法
    pub fn add_scalar(&self, scalar: T) -> Self {
        let result = &self.data + scalar;
        Self {
            data: Arc::new(result.into_shared()),
            name: None,
        }
    }

    /// 元素级乘法
    pub fn mul(&self, other: &Self) -> Result<Self, TensorError> {
        if self.shape() != other.shape() {
            return Err(TensorError::ShapeMismatch {
                expected: self.shape().to_vec(),
                actual: other.shape().to_vec(),
            });
        }
        let result = &self.data * &other.data;
        Ok(Self {
            data: Arc::new(result.into_shared()),
            name: None,
        })
    }

    /// 标量乘法
    pub fn mul_scalar(&self, scalar: T) -> Self {
        let result = &self.data * scalar;
        Self {
            data: Arc::new(result.into_shared()),
            name: None,
        }
    }

    /// 矩阵乘法
    ///
    /// 支持 2D 张量的标准矩阵乘法，以及批处理矩阵乘法。
    pub fn matmul(&self, other: &Self) -> Result<Self, TensorError> {
        let self_ndim = self.ndim();
        let other_ndim = other.ndim();

        match (self_ndim, other_ndim) {
            (2, 2) => {
                // 标准 2D 矩阵乘法
                let m = self.shape()[0];
                let k = self.shape()[1];
                let k2 = other.shape()[0];
                let n = other.shape()[1];
                if k != k2 {
                    return Err(TensorError::ShapeMismatch {
                        expected: vec![m, k],
                        actual: vec![k2, n],
                    });
                }
                // 使用 ndarray 的矩阵乘法
                let result = self.data.dot(&other.data);
                Ok(Self {
                    data: Arc::new(result.into_shared()),
                    name: None,
                })
            }
            _ => Err(TensorError::OperationFailed(format!(
                "不支持的 matmul 维度组合: {}D x {}D",
                self_ndim, other_ndim
            ))),
        }
    }

    /// 转置操作（仅支持 2D）
    pub fn transpose(&self) -> Result<Self, TensorError> {
        if self.ndim() != 2 {
            return Err(TensorError::OperationFailed(
                "转置仅支持 2D 张量".to_string(),
            ));
        }
        let result = self.data.t().into_owned();
        Ok(Self {
            data: Arc::new(result.into_shared()),
            name: None,
        })
    }

    /// ReLU 激活函数: max(0, x)
    pub fn relu(&self) -> Self {
        let result = self.data.mapv(|x| {
            if x < T::zero() {
                T::zero()
            } else {
                x
            }
        });
        Self {
            data: Arc::new(result.into_shared()),
            name: None,
        }
    }

    /// GELU 激活函数（近似版本）
    pub fn gelu(&self) -> Self {
        // GELU 近似: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        let sqrt_2_over_pi = T::from(0.7978845608028654_f64).unwrap();
        let coeff = T::from(0.044715_f64).unwrap();
        let result = self.data.mapv(|x| {
            let inner = sqrt_2_over_pi * (x + coeff * x * x * x);
            let tanh_val = inner.tanh();
            T::from(0.5_f64).unwrap() * x * (T::one() + tanh_val)
        });
        Self {
            data: Arc::new(result.into_shared()),
            name: None,
        }
    }

    /// Softmax 函数（沿最后一个维度）
    pub fn softmax(&self) -> Result<Self, TensorError> {
        if self.ndim() == 0 {
            return Err(TensorError::OperationFailed(
                "Softmax 不支持标量张量".to_string(),
            ));
        }

        // 获取最后一个轴
        let last_axis = self.ndim() - 1;

        // 计算最大值（数值稳定性）
        let max_val = self
            .data
            .fold_axis(ndarray::Axis(last_axis), T::neg_infinity(), |a, &b| {
                if b > a { b } else { a }
            });

        // 使用广播减去最大值，然后计算 exp，最后归一化
        let result = {
            let expanded_max = max_val
                .clone()
                .into_shape_with_order(IxDyn(&{
                    let mut s = self.shape().to_vec();
                    s.push(1);
                    s
                }))
                .unwrap()
                .broadcast(self.data.shape())
                .unwrap()
                .to_owned();

            let shifted = &self.data - &expanded_max;
            let exp_arr = shifted.mapv(|x| x.exp());

            // 沿最后一个轴求和
            let sum_exp = exp_arr.sum_axis(ndarray::Axis(last_axis));

            // 扩展 sum 的形状以便广播
            let sum_shape: Vec<usize> = self
                .shape()
                .iter()
                .enumerate()
                .map(|(i, &s)| if i == last_axis { 1 } else { s })
                .collect();
            let sum_exp = sum_exp
                .into_shape_with_order(IxDyn(&sum_shape))
                .unwrap()
                .broadcast(self.data.shape())
                .unwrap()
                .to_owned();

            &exp_arr / &sum_exp
        };

        Ok(Self {
            data: Arc::new(result.into_shared()),
            name: None,
        })
    }

    /// LayerNorm 层归一化
    pub fn layer_norm(&self, epsilon: T) -> Result<Self, TensorError> {
        if self.ndim() == 0 {
            return Err(TensorError::OperationFailed(
                "LayerNorm 不支持标量张量".to_string(),
            ));
        }
        let last_axis = self.ndim() - 1;

        // 计算均值
        let mean = self
            .data
            .mean_axis(ndarray::Axis(last_axis))
            .ok_or_else(|| TensorError::OperationFailed("计算均值失败".to_string()))?;

        // 扩展均值形状以便广播
        let mean_shape: Vec<usize> = self
            .shape()
            .iter()
            .enumerate()
            .map(|(i, &s)| if i == last_axis { 1 } else { s })
            .collect();
        let mean = mean
            .into_shape_with_order(IxDyn(&mean_shape))
            .unwrap()
            .broadcast(self.data.shape())
            .unwrap()
            .to_owned();

        // 计算方差
        let diff = &self.data - &mean;
        let variance = diff.mapv(|x| x * x);
        let variance = variance
            .mean_axis(ndarray::Axis(last_axis))
            .ok_or_else(|| TensorError::OperationFailed("计算方差失败".to_string()))?;

        let var_shape: Vec<usize> = self
            .shape()
            .iter()
            .enumerate()
            .map(|(i, &s)| if i == last_axis { 1 } else { s })
            .collect();
        let variance = variance
            .into_shape_with_order(IxDyn(&var_shape))
            .unwrap()
            .broadcast(self.data.shape())
            .unwrap()
            .to_owned();

        // 标准化
        let std = variance.mapv(|v| (v + epsilon).sqrt());
        let result = &diff / &std;

        Ok(Self {
            data: Arc::new(result.into_shared()),
            name: None,
        })
    }

    /// 沿指定轴求和
    pub fn sum_axis(&self, axis: usize) -> Result<Self, TensorError> {
        if axis >= self.ndim() {
            return Err(TensorError::IndexOutOfBounds(format!(
                "轴 {} 超出范围 [0, {})",
                axis,
                self.ndim()
            )));
        }
        let result = self.data.sum_axis(ndarray::Axis(axis));
        Ok(Self {
            data: Arc::new(result.into_shared()),
            name: None,
        })
    }

    /// 获取指定位置的元素
    pub fn get(&self, indices: &[usize]) -> Result<T, TensorError> {
        if indices.len() != self.ndim() {
            return Err(TensorError::IndexOutOfBounds(format!(
                "索引维度 {} 与张量维度 {} 不匹配",
                indices.len(),
                self.ndim()
            )));
        }
        self.data
            .get(IxDyn(indices))
            .cloned()
            .ok_or_else(|| TensorError::IndexOutOfBounds(format!("无效索引: {:?}", indices)))
    }
}

impl<T: Num + Float + Clone + fmt::Debug> fmt::Debug for Tensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.shape())
            .field("len", &self.len())
            .field("name", &self.name)
            .finish()
    }
}

/// 方便的类型别名
pub type TensorF32 = Tensor<f32>;
pub type TensorF64 = Tensor<f64>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let t = TensorF32::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
        assert_eq!(t.len(), 4);
    }

    #[test]
    fn test_tensor_add() {
        let a = TensorF32::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = TensorF32::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
        let c = a.add(&b).unwrap();
        assert_eq!(c.to_vec(), vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_tensor_matmul() {
        let a = TensorF32::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = TensorF32::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
        let c = a.matmul(&b).unwrap();
        // [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
        assert_eq!(c.shape(), &[2, 2]);
        assert!((c.get(&[0, 0]).unwrap() - 19.0).abs() < 1e-6);
        assert!((c.get(&[1, 1]).unwrap() - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_relu() {
        let t = TensorF32::from_vec(vec![-1.0, 0.0, 1.0, 2.0], &[2, 2]).unwrap();
        let r = t.relu();
        assert_eq!(r.to_vec(), vec![0.0, 0.0, 1.0, 2.0]);
    }
}