//! HTTP API 服务器
//!
//! 基于 axum 构建的 REST API，提供推理服务、健康检查和模型信息查询。

use crate::engine::graph::GraphBuilder;
use crate::engine::inference::{InferenceEngine, InferenceError};
use crate::engine::tensor::TensorF32;
use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

/// API 服务器状态
///
/// 包含推理引擎和运行时状态。
#[derive(Clone)]
pub struct AppState {
    /// 推理引擎（线程安全）
    engine: Arc<RwLock<InferenceEngine>>,
    /// 服务器启动时间
    start_time: std::time::Instant,
}

/// 推理请求
#[derive(Debug, Deserialize, Serialize)]
pub struct InferenceRequest {
    /// 输入数据（扁平化的一维数组）
    pub input_data: Vec<f32>,
    /// 输入形状
    pub input_shape: Vec<usize>,
    /// 输入名称（可选）
    pub input_name: Option<String>,
    /// 输出名称（可选）
    pub output_name: Option<String>,
}

/// 推理响应
#[derive(Debug, Serialize)]
pub struct InferenceResponse {
    /// 推理结果（扁平化的一维数组）
    pub output_data: Vec<f32>,
    /// 输出形状
    pub output_shape: Vec<usize>,
    /// 推理耗时（毫秒）
    pub inference_time_ms: f64,
    /// 后端名称
    pub backend: String,
}

/// 健康检查响应
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    /// 健康状态
    pub status: String,
    /// 引擎版本
    pub version: String,
    /// 运行时间（秒）
    pub uptime_secs: f64,
}

/// 模型信息响应
#[derive(Debug, Serialize)]
pub struct ModelInfoResponse {
    /// 是否已加载模型
    pub model_loaded: bool,
    /// 节点数量
    pub node_count: usize,
    /// 输入数量
    pub input_count: usize,
    /// 输出数量
    pub output_count: usize,
    /// 后端信息
    pub backend: String,
    /// 设备信息
    pub device_info: String,
}

/// 错误响应
#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    /// 错误信息
    pub error: String,
    /// 错误详情
    pub detail: Option<String>,
}

impl IntoResponse for ErrorResponse {
    fn into_response(self) -> Response {
        let status = StatusCode::INTERNAL_SERVER_ERROR;
        (status, Json(self)).into_response()
    }
}

/// 创建 API 路由
pub fn create_router(engine: InferenceEngine) -> Router {
    let state = AppState {
        engine: Arc::new(RwLock::new(engine)),
        start_time: std::time::Instant::now(),
    };

    Router::new()
        .route("/v1/inference", post(inference_handler))
        .route("/v1/health", get(health_handler))
        .route("/v1/model/info", get(model_info_handler))
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive())
        .with_state(state)
}

/// 创建带有默认示例图的 API 路由
pub fn create_router_with_sample_graph() -> Router {
    let mut engine = InferenceEngine::new_cpu();

    // 创建一个简单的示例计算图
    let mut builder = GraphBuilder::new();
    let input_id = builder.input("input", "f32", vec![1, 4]).unwrap();
    let dense_id = builder.dense("fc1", 4, 3, &input_id, true).unwrap();
    let relu_id = builder.relu("relu1", &dense_id).unwrap();
    let _output_id = builder.output("output", &relu_id).unwrap();
    let graph = builder.build().unwrap();

    engine.load_graph(graph);

    create_router(engine)
}

// ---- API 处理函数 ----

/// POST /v1/inference - 执行推理
async fn inference_handler(
    State(state): State<AppState>,
    Json(req): Json<InferenceRequest>,
) -> Result<Json<InferenceResponse>, ErrorResponse> {
    log::info!(
        "收到推理请求: 输入形状 {:?}, 数据量 {}",
        req.input_shape,
        req.input_data.len()
    );

    // 验证输入数据
    let expected_len: usize = req.input_shape.iter().product();
    if req.input_data.len() != expected_len {
        return Err(ErrorResponse {
            error: "输入数据长度与形状不匹配".to_string(),
            detail: Some(format!(
                "期望 {} 个元素，实际 {} 个",
                expected_len,
                req.input_data.len()
            )),
        });
    }

    // 创建输入张量
    let tensor = TensorF32::from_vec(req.input_data.clone(), &req.input_shape)
        .map_err(|e| ErrorResponse {
            error: "创建输入张量失败".to_string(),
            detail: Some(e.to_string()),
        })?;

    // 执行推理
    let mut engine = state.engine.write().await;
    let input_name = req.input_name.as_deref().unwrap_or("input");
    engine.set_input(input_name, tensor).map_err(|e| ErrorResponse {
        error: "设置输入失败".to_string(),
        detail: Some(e.to_string()),
    })?;

    engine.run().map_err(|e| ErrorResponse {
        error: "推理执行失败".to_string(),
        detail: Some(e.to_string()),
    })?;

    // 获取输出
    let output = engine
        .get_output(req.output_name.as_deref())
        .map_err(|e| ErrorResponse {
            error: "获取输出失败".to_string(),
            detail: Some(e.to_string()),
        })?;

    // 构建响应
    let stats = engine.last_stats().cloned().unwrap_or_default();
    let response = InferenceResponse {
        output_data: output.to_vec(),
        output_shape: output.shape().to_vec(),
        inference_time_ms: stats.total_time_ms,
        backend: stats.backend_name,
    };

    log::info!(
        "推理完成: 耗时 {:.3}ms, 输出形状 {:?}",
        response.inference_time_ms,
        response.output_shape
    );

    Ok(Json(response))
}

/// GET /v1/health - 健康检查
async fn health_handler(State(state): State<AppState>) -> Json<HealthResponse> {
    let uptime = state.start_time.elapsed().as_secs_f64();

    Json(HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        uptime_secs: uptime,
    })
}

/// GET /v1/model/info - 模型信息
async fn model_info_handler(State(state): State<AppState>) -> Json<ModelInfoResponse> {
    let engine = state.engine.read().await;
    let info = engine.engine_info();

    Json(ModelInfoResponse {
        model_loaded: info.graph_loaded,
        node_count: info.node_count,
        input_count: info.input_count,
        output_count: info.output_count,
        backend: info.backend,
        device_info: info.device_info,
    })
}

/// 启动 HTTP 服务器
///
/// # 参数
///
/// - `addr`: 绑定地址（如 "0.0.0.0:8080"）
/// - `engine`: 推理引擎实例
pub async fn serve(addr: &str, engine: InferenceEngine) -> anyhow::Result<()> {
    let router = create_router(engine);

    log::info!("启动 HTTP 推理服务器: http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, router).await?;

    Ok(())
}

/// 启动带示例图的 HTTP 服务器（用于演示和测试）
pub async fn serve_with_sample(addr: &str) -> anyhow::Result<()> {
    let router = create_router_with_sample_graph();

    log::info!(
        "启动演示服务器（含示例计算图）: http://{}",
        addr
    );
    log::info!("可用端点:");
    log::info!("  POST /v1/inference - 执行推理");
    log::info!("  GET  /v1/health    - 健康检查");
    log::info!("  GET  /v1/model/info - 模型信息");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, router).await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_health_endpoint() {
        let engine = InferenceEngine::new_cpu();
        let router = create_router(engine);
        let app = router.into_app();

        use axum::body::Body;
        use axum::http::{Request, StatusCode};

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/v1/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_model_info_endpoint() {
        let engine = InferenceEngine::new_cpu();
        let router = create_router(engine);
        let app = router.into_app();

        use axum::body::Body;
        use axum::http::Request, StatusCode;

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/v1/model/info")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }
}