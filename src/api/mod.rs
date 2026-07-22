//! API 模块
//!
//! 提供 HTTP REST API 服务，用于远程调用推理引擎。

pub mod server;

pub use server::create_router;