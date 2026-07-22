//! CUDA 后端桩（stub）
//!
//! CUDA 功能尚未实现，当前会回退到 CPU 后端执行。

/// CUDA 相关错误
#[derive(Debug, thiserror::Error)]
pub enum CudaError {
    #[error("CUDA 功能尚未实现")]
    NotImplemented,
    #[error("CUDA 运行时错误: {0}")]
    RuntimeError(String),
    #[error("CUDA 设备未找到: 设备 ID {0}")]
    DeviceNotFound(usize),
    #[error("CUDA 内存分配失败: 需要 {0} 字节")]
    OutOfMemory(usize),
    #[error("CUDA 内核启动失败: {0}")]
    KernelLaunchFailed(String),
}

/// CUDA 计算后端（桩实现）
///
/// 当前版本中，CUDA 后端尚未实现实际的 GPU 计算。
/// 所有操作会记录警告日志，并返回适当的错误信息。
///
/// 未来将使用 cudarc 或 burn 框架集成 CUDA 支持。
pub struct CudaBackend {
    /// GPU 设备 ID
    #[allow(dead_code)]
    device_id: usize,
}

impl CudaBackend {
    /// 创建 CUDA 后端实例
    ///
    /// # 返回
    ///
    /// 由于 CUDA 功能尚未实现，始终返回错误。
    pub fn new(_device_id: usize) -> Result<Self, CudaError> {
        log::warn!("CUDA 后端尚未实现，将回退到 CPU 后端");
        Err(CudaError::NotImplemented)
    }

    /// 检查 CUDA 是否可用
    pub fn is_available() -> bool {
        false
    }

    /// 获取可用的 GPU 设备数量
    pub fn device_count() -> usize {
        0
    }
}

impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CudaError::NotImplemented => {
                write!(f, "CUDA 后端尚未实现，请使用 CPU 后端（--backend cpu）")
            }
            _ => std::fmt::Debug::fmt(self, f),
        }
    }
}

/// 当用户尝试使用 CUDA 但不可用时的提示信息
pub fn cuda_unavailable_message() -> String {
    String::from(
        "CUDA 后端尚未实现。\n\
         当前版本仅支持 CPU 后端。\n\
         未来版本将支持通过 CUDA/cuDNN 加速推理。\n\
         请使用 --backend cpu 参数来使用 CPU 后端。",
    )
}

// 注意：CudaBackend 暂不实现 Backend trait，
// 因为 CudaBackend::new() 始终返回错误，无法创建实例。
// 当 CUDA 功能实现后，将添加 Backend trait 实现。