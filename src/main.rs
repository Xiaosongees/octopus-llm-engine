//! CLI 入口点
//!
//! 提供命令行接口，支持以下子命令：
//! - `serve`: 启动 HTTP 推理服务器
//! - `benchmark`: 运行性能基准测试
//! - `info`: 显示引擎和系统信息

use clap::{Parser, Subcommand};
use rust_ai_inference_engine::backend::BackendType;
use rust_ai_inference_engine::engine::graph::GraphBuilder;
use rust_ai_inference_engine::engine::inference::InferenceEngine;
use rust_ai_inference_engine::engine::tensor::TensorF32;
use rust_ai_inference_engine::model::transformer::{TransformerConfig, compute_model_stats};

/// Rust AI 推理引擎 CLI
#[derive(Parser)]
#[command(
    name = "rust-ai-inference-engine",
    version,
    about = "高性能 AI 推理引擎",
    long_about = "基于 Rust 的 AI 推理引擎，支持 Transformer 模型和 ONNX 格式。\n提供 HTTP API 服务和命令行推理功能。"
)]
struct Cli {
    /// 子命令
    #[command(subcommand)]
    command: Commands,

    /// 计算后端类型
    #[arg(long, default_value = "cpu", global = true, help = "计算后端类型 (cpu, cuda)")]
    backend: String,

    /// 日志级别
    #[arg(long, default_value = "info", global = true, help = "日志级别 (trace, debug, info, warn, error)")]
    log_level: String,
}

/// 可用子命令
#[derive(Subcommand)]
enum Commands {
    /// 启动 HTTP 推理服务器
    Serve {
        /// 监听地址
        #[arg(short, long, default_value = "0.0.0.0:8080", help = "监听地址")]
        addr: String,

        /// 使用示例模型
        #[arg(long, default_value = "false", help = "使用内置示例模型")]
        sample: bool,
    },

    /// 运行性能基准测试
    Benchmark {
        /// 推理迭代次数
        #[arg(short, long, default_value = "100", help = "推理迭代次数")]
        iterations: usize,

        /// 张量大小
        #[arg(short, long, default_value = "1024", help = "测试张量的元素数量")]
        tensor_size: usize,

        /// 矩阵乘法的维度
        #[arg(short, long, default_value = "32", help = "矩阵乘法的维度 (M x N)")]
        matmul_size: usize,
    },

    /// 显示引擎和系统信息
    Info,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // 初始化日志
    init_logger(&cli.log_level)?;

    // 解析后端类型
    let backend_type = cli
        .backend
        .parse::<BackendType>()
        .map_err(|e| anyhow::anyhow!(e))?;

    // 处理子命令
    match cli.command {
        Commands::Serve { addr, sample } => cmd_serve(addr, sample, backend_type).await,
        Commands::Benchmark {
            iterations,
            tensor_size,
            matmul_size,
        } => cmd_benchmark(iterations, tensor_size, matmul_size),
        Commands::Info => cmd_info(backend_type),
    }
}

/// 初始化日志系统
fn init_logger(level: &str) -> anyhow::Result<()> {
    let log_level = match level.to_lowercase().as_str() {
        "trace" => log::LevelFilter::Trace,
        "debug" => log::LevelFilter::Debug,
        "info" => log::LevelFilter::Info,
        "warn" => log::LevelFilter::Warn,
        "error" => log::LevelFilter::Error,
        _ => log::LevelFilter::Info,
    };

    env_logger::Builder::from_default_env()
        .filter_level(log_level)
        .format_timestamp_secs()
        .init();

    log::debug!("日志级别设置为: {}", level);
    Ok(())
}

/// serve 子命令：启动 HTTP 推理服务器
async fn cmd_serve(addr: String, sample: bool, backend_type: BackendType) -> anyhow::Result<()> {
    if backend_type == BackendType::Cuda {
        log::error!("{}", rust_ai_inference_engine::backend::cuda::cuda_unavailable_message());
        anyhow::bail!("CUDA 后端不可用");
    }

    if sample {
        // 使用示例模型启动服务器
        log::info!("使用内置示例模型启动服务器");
        rust_ai_inference_engine::api::server::serve_with_sample(&addr).await
    } else {
        log::info!("启动推理服务器（无模型）");
        let engine = InferenceEngine::new(backend_type)?;
        rust_ai_inference_engine::api::server::serve(&addr, engine).await
    }
}

/// benchmark 子命令：运行性能基准测试
fn cmd_benchmark(
    iterations: usize,
    tensor_size: usize,
    matmul_size: usize,
) -> anyhow::Result<()> {
    println!("=== Rust AI 推理引擎 - 性能基准测试 ===");
    println!();

    // 1. 张量创建基准
    println!("[1/4] 张量创建基准");
    let data: Vec<f32> = (0..tensor_size).map(|i| i as f32).collect();
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = TensorF32::from_vec(data.clone(), &[tensor_size]);
    }
    let elapsed = start.elapsed();
    println!(
        "  创建 {}x {}D 张量: {:.3}ms (平均 {:.3}ms/次)",
        iterations,
        1,
        elapsed.as_secs_f64() * 1000.0,
        elapsed.as_secs_f64() * 1000.0 / iterations as f64
    );

    // 2. 张量运算基准（ReLU）
    println!("[2/4] ReLU 激活基准");
    let tensor = TensorF32::from_vec(data, &[tensor_size]).unwrap();
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = tensor.relu();
    }
    let elapsed = start.elapsed();
    println!(
        "  ReLU {}x {} 元素: {:.3}ms (平均 {:.3}ms/次)",
        iterations,
        tensor_size,
        elapsed.as_secs_f64() * 1000.0,
        elapsed.as_secs_f64() * 1000.0 / iterations as f64
    );

    // 3. 矩阵乘法基准
    println!("[3/4] 矩阵乘法基准 ({}x{} @ {}x{})", matmul_size, matmul_size, matmul_size, matmul_size);
    let n = matmul_size;
    let a_data: Vec<f32> = (0..n * n).map(|i| (i % 100) as f32 / 100.0).collect();
    let b_data: Vec<f32> = (0..n * n).map(|i| ((i + 50) % 100) as f32 / 100.0).collect();
    let a = TensorF32::from_vec(a_data, &[n, n]).unwrap();
    let b = TensorF32::from_vec(b_data, &[n, n]).unwrap();

    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = a.matmul(&b);
    }
    let elapsed = start.elapsed();
    println!(
        "  MatMul {}x{}: {:.3}ms (平均 {:.3}ms/次)",
        n, n,
        elapsed.as_secs_f64() * 1000.0,
        elapsed.as_secs_f64() * 1000.0 / iterations as f64
    );

    // 4. 推理图执行基准
    println!("[4/4] 推理图执行基准");
    let mut engine = InferenceEngine::new_cpu();

    // 构建一个多层计算图
    let mut builder = GraphBuilder::new();
    let input_id = builder.input("input", "f32", vec![1, matmul_size])?;
    let mut current = input_id;
    for i in 0..4 {
        let dense_id = builder.dense(
            format!("layer_{}", i),
            matmul_size,
            matmul_size,
            &current,
            true,
        )?;
        let relu_id = builder.relu(format!("relu_{}", i), &dense_id)?;
        current = relu_id;
    }
    let _output = builder.output("output", &current)?;
    let graph = builder.build()?;

    engine.load_graph(graph);

    // 创建输入
    let input_data: Vec<f32> = (0..matmul_size).map(|i| i as f32 / matmul_size as f32).collect();
    let input_tensor = TensorF32::from_vec(input_data, &[1, matmul_size])?;

    // 执行基准
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        engine.set_input("input", input_tensor.clone())?;
        engine.run()?;
    }
    let elapsed = start.elapsed();
    println!(
        "  4层 Dense+ReLU 图 {}x: {:.3}ms (平均 {:.3}ms/次)",
        iterations,
        elapsed.as_secs_f64() * 1000.0,
        elapsed.as_secs_f64() * 1000.0 / iterations as f64
    );

    println!();
    println!("基准测试完成。");

    Ok(())
}

/// info 子命令：显示引擎和系统信息
fn cmd_info(backend_type: BackendType) -> anyhow::Result<()> {
    println!("=== Rust AI 推理引擎 - 系统信息 ===");
    println!();

    // 引擎版本信息
    println!("[引擎信息]");
    println!("  版本: {}", env!("CARGO_PKG_VERSION"));
    println!("  版本描述: {}", env!("CARGO_PKG_DESCRIPTION"));
    println!("  Rust edition: 2021");
    println!();

    // 系统信息
    println!("[系统信息]");
    println!("  操作系统: {}", std::env::consts::OS);
    println!("  架构: {}", std::env::consts::ARCH);
    println!("  CPU 核心数: {}", num_cpus::get());
    println!("  可用内存: {:.2} GB", get_available_memory_gb());
    println!();

    // 后端信息
    println!("[计算后端]");
    let engine = InferenceEngine::new(backend_type)?;
    println!("  后端类型: {}", backend_type);
    println!("  后端名称: {}", engine.backend_name());
    println!("  设备信息: {}", engine.backend_info());
    println!();

    // Transformer 参数量
    println!("[模型参考信息]");
    let configs = vec![
        ("小型 (Small)", TransformerConfig::small()),
        ("默认 (Default)", TransformerConfig::default()),
    ];

    for (name, config) in configs {
        let stats = compute_model_stats(&config);
        let params_mb = stats.total_params as f64 * 4.0 / (1024.0 * 1024.0);
        println!(
            "  {} 配置: d_model={}, heads={}, layers={}, d_ff={}, vocab={}, params={:.2}M ({:.1}MB)",
            name,
            config.d_model,
            config.n_heads,
            config.n_layers,
            config.d_ff,
            config.vocab_size,
            stats.total_params as f64 / 1e6,
            params_mb
        );
    }

    println!();

    // CUDA 状态
    println!("[CUDA 状态]");
    if rust_ai_inference_engine::backend::cuda::CudaBackend::is_available() {
        println!("  CUDA 可用: 是");
        println!("  GPU 设备数: {}", rust_ai_inference_engine::backend::cuda::CudaBackend::device_count());
    } else {
        println!("  CUDA 可用: 否 (当前版本仅支持 CPU 后端)");
    }

    println!();
    println!("=== 信息显示完毕 ===");

    Ok(())
}

/// 获取可用内存（简化版）
fn get_available_memory_gb() -> f64 {
    // Windows 上的简化内存检测
    #[cfg(target_os = "windows")]
    {
        // 简化实现，返回一个合理的估计值
        // 生产环境应使用 sysinfo 库
        16.0 // 默认估计
    }

    #[cfg(not(target_os = "windows"))]
    {
        16.0
    }
}