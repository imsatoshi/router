//! lingjing patch (0009) over real HTTP: a charge lives from dispatch to the first response chunk,
//! and a client that leaves gives it back. Process-wide switches, so one scenario in its own file.
mod common;

use axum::{
    body::{Body, Bytes},
    extract::{DefaultBodyLimit, State},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router as AxumRouter,
};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, watch};
use vllm_router_rs::config::{
    CircuitBreakerConfig, ConnectionMode, PolicyConfig, RetryConfig, RouterConfig, RoutingMode,
};
use vllm_router_rs::routers::RouterFactory;
use vllm_router_rs::server::{build_app_with_wasm_middleware, AppState};

#[derive(Clone)]
struct Gates {
    first: watch::Receiver<bool>,
    last: watch::Receiver<bool>,
}

/// Streams one chunk after `first` opens and ends after `last` opens.
async fn chat(State(mut gates): State<Gates>, _body: Bytes) -> Response {
    let (tx, rx) = mpsc::unbounded_channel::<Result<Bytes, std::io::Error>>();
    tokio::spawn(async move {
        let _ = gates.first.wait_for(|open| *open).await;
        let _ = tx.send(Ok(Bytes::from_static(b"data: {\"choices\":[]}\n\n")));
        let _ = gates.last.wait_for(|open| *open).await;
        let _ = tx.send(Ok(Bytes::from_static(b"data: [DONE]\n\n")));
    });
    Response::builder()
        .header("content-type", "text/event-stream")
        .body(Body::from_stream(tokio_stream::wrappers::UnboundedReceiverStream::new(rx)))
        .unwrap()
        .into_response()
}

fn config(worker_urls: Vec<String>) -> RouterConfig {
    RouterConfig {
        mode: RoutingMode::Regular { worker_urls },
        policy: PolicyConfig::CacheAware {
            cache_threshold: 0.3,
            balance_abs_threshold: 8,
            balance_rel_threshold: 1.2,
            eviction_interval_secs: 0,
            max_tree_size: 1 << 24,
        },
        host: "127.0.0.1".to_string(),
        port: 3002,
        max_payload_size: 256 * 1024 * 1024,
        request_timeout_secs: 30,
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
        disable_retries: true,
        disable_circuit_breaker: true,
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

async fn engine(gates: Gates) -> String {
    let app = AxumRouter::new()
        .route("/health", get(|| async { "ok" }))
        .route("/v1/models", get(|| async { Json(serde_json::json!({"data": []})) }))
        .route("/metrics", get(|| async { "" }))
        .route("/v1/chat/completions", post(chat))
        .layer(DefaultBodyLimit::disable())
        .with_state(gates);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    url
}

async fn charged(client: &reqwest::Client, base: &str, url: &str) -> u64 {
    let body: serde_json::Value = client
        .get(format!("{base}/lingjing/uncached"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    body["tokens"][url].as_u64().unwrap_or(0)
}

async fn until(client: &reqwest::Client, base: &str, url: &str, want: u64) {
    for _ in 0..200 {
        if charged(client, base, url).await == want {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("charge never reached {want}: {}", charged(client, base, url).await);
}

fn body(bytes: usize) -> Vec<u8> {
    serde_json::to_vec(&serde_json::json!({
        "model": "m",
        "messages": [{"role": "user", "content": "x".repeat(bytes)}],
        "session_params": {"session_id": "0123456789abcdef"},
        "stream": true,
    }))
    .unwrap()
}

#[tokio::test(flavor = "multi_thread")]
async fn a_charge_ends_at_the_first_chunk_or_when_the_client_leaves() {
    for (name, value) in [
        ("LINGJING_UNCACHED_MODE", "shadow"),
        ("LINGJING_UNCACHED_PEAK_TPS", "1000"),
        ("LINGJING_UNCACHED_BYTES_PER_TOKEN", "10"),
    ] {
        std::env::set_var(name, value);
    }
    let (first_tx, first) = watch::channel(false);
    let (last_tx, last) = watch::channel(false);
    let url = engine(Gates { first, last }).await;
    let cfg = config(vec![url.clone()]);
    let ctx = common::create_test_context(cfg.clone());
    let router = RouterFactory::create_router(&ctx).await.unwrap();
    let state = Arc::new(AppState {
        router: Arc::from(router),
        context: ctx.clone(),
        concurrency_queue_tx: None,
        router_manager: None,
    });
    let app = build_app_with_wasm_middleware(state, cfg.max_payload_size, vec![], vec![], true, false, None);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let client = reqwest::Client::builder().no_proxy().build().unwrap();

    let send = |bytes: usize| {
        let request = client
            .post(format!("{base}/v1/chat/completions"))
            .header("content-type", "application/json")
            .body(body(bytes));
        tokio::spawn(async move { request.send().await.unwrap() })
    };

    // Prefill: the whole body is charged until the first chunk, not until the stream ends.
    let whole = body(1000).len() as u64 / 10;
    let turn = send(1000);
    until(&client, &base, &url, whole).await;
    first_tx.send(true).unwrap();
    until(&client, &base, &url, 0).await;
    last_tx.send(true).unwrap();
    turn.await.unwrap().bytes().await.unwrap();

    // The next turn on the same replica pays for the new bytes only.
    first_tx.send(false).unwrap();
    last_tx.send(false).unwrap();
    let delta = (body(3000).len() - body(1000).len()) as u64 / 10;
    let turn = send(3000);
    until(&client, &base, &url, delta).await;

    // The client leaves during prefill: the charge goes with it.
    drop(turn.await.unwrap());
    until(&client, &base, &url, 0).await;
}
