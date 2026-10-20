//! lingjing patch (0006): `/lingjing/load` is served by the router itself and `/metrics` on the
//! business port no longer falls through to the transparent proxy (which forwarded it to one
//! fixed worker and counted it as a processed request).
mod common;

use axum::{body::Body, extract::Request, http::StatusCode};
use common::mock_worker::{HealthStatus, MockWorker, MockWorkerConfig, WorkerType};
use std::sync::Arc;
use tower::ServiceExt;
use vllm_router_rs::config::{
    CircuitBreakerConfig, ConnectionMode, PolicyConfig, RetryConfig, RouterConfig, RoutingMode,
};
use vllm_router_rs::routers::RouterFactory;
use vllm_router_rs::server::{build_app_with_wasm_middleware, AppState};

fn config(worker_urls: Vec<String>) -> RouterConfig {
    RouterConfig {
        mode: RoutingMode::Regular { worker_urls },
        policy: PolicyConfig::Random,
        host: "127.0.0.1".to_string(),
        port: 3002,
        max_payload_size: 256 * 1024 * 1024,
        request_timeout_secs: 600,
        worker_startup_timeout_secs: 10,
        worker_startup_check_interval_secs: 1,
        discovery: None,
        intra_node_data_parallel_size: 1,
        api_key: None,
        api_key_validation_urls: vec![],
        metrics: None,
        log_dir: None,
        log_level: None,
        request_id_headers: None,
        max_concurrent_requests: 64,
        queue_size: 0,
        queue_timeout_secs: 60,
        rate_limit_tokens_per_second: None,
        cors_allowed_origins: vec![],
        retry: RetryConfig::default(),
        circuit_breaker: CircuitBreakerConfig::default(),
        disable_retries: false,
        disable_circuit_breaker: false,
        health_check: vllm_router_rs::config::HealthCheckConfig::default(),
        enable_igw: false,
        connection_mode: ConnectionMode::Http,
        history_backend: vllm_router_rs::config::HistoryBackend::Memory,
        enable_profiling: false,
        profile_timeout_secs: 30,
        epd: None,
        kv_connector: vllm_router_rs::config::KvConnector::Nixl,
        program_scheduling: None,
    }
}

async fn get(app: &axum::Router, uri: &str) -> (StatusCode, String) {
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(uri)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let status = resp.status();
    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    (status, String::from_utf8_lossy(&body).into_owned())
}

#[tokio::test]
async fn load_summary_and_metrics_are_router_routes() {
    let mut worker = MockWorker::new(MockWorkerConfig {
        port: 18761,
        worker_type: WorkerType::Regular,
        health_status: HealthStatus::Healthy,
        response_delay_ms: 0,
        fail_rate: 0.0,
    });
    let url = worker.start().await.unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    let cfg = config(vec![url]);
    let ctx = common::create_test_context(cfg.clone());
    let router = RouterFactory::create_router(&ctx).await.unwrap();
    let state = Arc::new(AppState {
        router: Arc::from(router),
        context: ctx.clone(),
        concurrency_queue_tx: None,
        router_manager: None,
    });
    let app = build_app_with_wasm_middleware(
        state,
        cfg.max_payload_size,
        vec![],
        vec![],
        true,
        false,
        None,
    );

    // Sampler off (the default in this process): the summary is not served.
    let (status, _) = get(&app, "/lingjing/load").await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    std::env::set_var("LINGJING_LOAD_INTERVAL_MS", "50");
    vllm_router_rs::lingjing_load::start_from_env(
        ctx.worker_registry.clone(),
        reqwest::Client::new(),
        None,
    );
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;

    let (status, body) = get(&app, "/lingjing/load").await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let summary: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(summary["schema"], 1);
    assert_eq!(summary["available"], 1);
    assert_eq!(summary["interval_ms"], 50);
    assert!(summary["registry_revision"].is_u64());
    assert_eq!(summary["replicas"].as_array().unwrap().len(), 1);
    assert!(summary["sampled_at_ms"].as_u64().unwrap() > 0);

    let (status, body) = get(&app, "/metrics").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert!(body.contains("use :29000/metrics"), "{body}");

    let workers = ctx.worker_registry.get_all();
    assert_eq!(workers.len(), 1);
    assert_eq!(
        workers[0].processed_requests(),
        0,
        "neither /metrics nor /lingjing/load may reach a worker"
    );
    worker.stop().await;
}
