//! lingjing patch (0008) end to end inside one process: a real router on a real socket, real
//! HTTP clients and fake engines that record which session landed where. The features are
//! process-wide, so this file is one scenario that walks scale-out, a health eviction, a
//! recovery, an eviction that turns into a removal, and a scale-in.
mod common;

use axum::{
    body::Bytes,
    extract::{DefaultBodyLimit, State},
    response::IntoResponse,
    routing::{get, post},
    Json, Router as AxumRouter,
};
use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};
use vllm_router_rs::config::{
    CircuitBreakerConfig, ConnectionMode, PolicyConfig, RetryConfig, RouterConfig, RoutingMode,
};
use vllm_router_rs::routers::RouterFactory;
use vllm_router_rs::server::{build_app_with_wasm_middleware, AppState};

type Seen = Arc<Mutex<Vec<String>>>;

async fn chat(State(seen): State<Seen>, body: Bytes) -> impl IntoResponse {
    let value: serde_json::Value = serde_json::from_slice(&body).unwrap_or_default();
    let session = value["session_params"]["session_id"].as_str().unwrap_or("");
    seen.lock().unwrap().push(session.to_string());
    Json(serde_json::json!({"id": "x", "object": "chat.completion", "choices": []}))
}

/// A fake engine; returns its URL and the sessions it served.
async fn engine() -> (String, Seen) {
    let seen: Seen = Arc::default();
    let app = AxumRouter::new()
        .route("/health", get(|| async { "ok" }))
        .route("/v1/models", get(|| async { Json(serde_json::json!({"data": []})) }))
        .route("/metrics", get(|| async { "" }))
        .route("/v1/chat/completions", post(chat))
        .layer(DefaultBodyLimit::disable())
        .with_state(seen.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (url, seen)
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

struct Fixture {
    base: String,
    client: reqwest::Client,
}

fn session(tag: u64, i: u64) -> String {
    format!("{:016x}", (tag << 32 | i).wrapping_mul(0x9e37_79b9_7f4a_7c15).rotate_left(23))
}

fn chat_body(session: &str, bytes: usize) -> Vec<u8> {
    serde_json::to_vec(&serde_json::json!({
        "model": "m",
        "messages": [{"role": "user", "content": "x".repeat(bytes)}],
        "session_params": {"session_id": session},
        "stream": false,
    }))
    .unwrap()
}

impl Fixture {
    async fn send(&self, session: &str, bytes: usize) {
        let status = self
            .client
            .post(format!("{}/v1/chat/completions", self.base))
            .header("content-type", "application/json")
            .body(chat_body(session, bytes))
            .send()
            .await
            .unwrap()
            .status();
        assert!(status.is_success(), "{status}");
    }

    /// Transfer-Encoding: chunked, so no content-length reaches the router.
    async fn send_chunked(&self, session: &str) {
        let body = chat_body(session, 16);
        let stream =
            futures_util::stream::iter([Ok::<_, std::io::Error>(Bytes::from(body))]);
        let status = self
            .client
            .post(format!("{}/v1/chat/completions", self.base))
            .header("content-type", "application/json")
            .body(reqwest::Body::wrap_stream(stream))
            .send()
            .await
            .unwrap()
            .status();
        assert!(status.is_success(), "{status}");
    }

    async fn admin(&self, path: &str, worker: &str, extra: &str) -> u16 {
        self.client
            .post(format!("{}/{path}?url={worker}{extra}", self.base))
            .send()
            .await
            .unwrap()
            .status()
            .as_u16()
    }

    async fn workers(&self) -> BTreeSet<String> {
        let list: serde_json::Value = self
            .client
            .get(format!("{}/list_workers", self.base))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        list["urls"]
            .as_array()
            .unwrap()
            .iter()
            .map(|u| u.as_str().unwrap().to_string())
            .collect()
    }

    async fn warming(&self) -> serde_json::Value {
        self.client
            .get(format!("{}/lingjing/warmup", self.base))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap()
    }
}

fn holds(seen: &Seen, session: &str) -> bool {
    seen.lock().unwrap().iter().any(|s| s == session)
}

const LARGE: usize = 300 * 1024;

#[tokio::test(flavor = "multi_thread")]
async fn slow_start_and_kept_affinity_over_real_http() {
    for (name, value) in [
        ("LINGJING_WARMUP_MODE", "on"),
        ("LINGJING_KEEP_HIT_ON_IMBALANCE", "1"),
        ("LINGJING_EVICT_KEEP_AFFINITY", "1"),
        ("LINGJING_SESSION_EXACT_MATCH", "1"),
        ("LINGJING_ROUTE_QUALITY", "1"),
        ("LINGJING_WARMUP_STARTUP_GRACE_SECS", "0"),
    ] {
        std::env::set_var(name, value);
    }

    let (url_a, seen_a) = engine().await;
    let (url_b, seen_b) = engine().await;
    let (url_c, seen_c) = engine().await;
    let cfg = config(vec![url_a.clone(), url_b.clone()]);
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
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let f = Fixture { base, client };

    // Two replicas with history.
    for i in 0..40 {
        f.send(&session(1, i), 64).await;
    }
    assert!(f.warming().await["entries"].as_array().unwrap().is_empty());

    // Scale-out: the third replica is warming and owes the old replicas its tree size.
    assert_eq!(f.admin("add_worker", &url_c, "").await, 200);
    let warming = f.warming().await;
    assert_eq!(warming["entries"][0]["url"], url_c.as_str(), "{warming}");
    assert!(warming["entries"][0]["credit"].as_u64().unwrap() > 0, "{warming}");

    // Known sessions stay, even when their bodies have grown past the gate.
    for i in 0..40 {
        let s = session(1, i);
        let on_a = holds(&seen_a, &s);
        f.send(&s, LARGE).await;
        assert!(!holds(&seen_c, &s), "known session moved to the new replica");
        assert_eq!(holds(&seen_a, &s), on_a);
        assert_eq!(seen_a.lock().unwrap().iter().filter(|x| **x == s).count() == 2, on_a);
    }

    // New sessions at the threshold stay off the warming replica; a real content-length
    // reached the selection path, or the small ones below could not get through.
    for i in 0..30 {
        f.send(&session(2, i), LARGE).await;
    }
    assert!(seen_c.lock().unwrap().is_empty(), "no large session on the warming replica");

    // Chunked: length unknown, treated as large.
    for i in 0..10 {
        f.send_chunked(&session(3, i)).await;
    }
    assert!(seen_c.lock().unwrap().is_empty(), "no unknown-length session on the warming replica");

    // Small new sessions do reach it.
    for i in 0..300 {
        f.send(&session(4, i), 64).await;
    }
    let small_on_c = seen_c.lock().unwrap().len();
    assert!(small_on_c > 0, "the new replica must take small sessions");
    assert!(small_on_c < 300, "and not all of them");

    // Two hashes sharing their first chars are two sessions: each one is placed on its own, and
    // both come back to where they went.
    let (one, two) = ("c0ffee0000000001", "c0ffee1111111111");
    f.send(one, 64).await;
    f.send(two, 64).await;
    let (on_a, on_b) = (holds(&seen_a, one), holds(&seen_b, one));
    f.send(one, 64).await;
    assert_eq!((holds(&seen_a, one), holds(&seen_b, one)), (on_a, on_b));
    let count = |seen: &Seen, s: &str| seen.lock().unwrap().iter().filter(|x| *x == s).count();
    let served_two = [&seen_a, &seen_b, &seen_c].map(|seen| count(seen, two));
    f.send(two, 64).await;
    let again_two = [&seen_a, &seen_b, &seen_c].map(|seen| count(seen, two));
    assert_eq!(
        served_two.iter().position(|n| *n == 1),
        again_two.iter().position(|n| *n == 2),
        "a whole match stays put"
    );

    // Health eviction keeps the affinity; a session waits elsewhere and comes back.
    let s = session(1, (0..40).find(|&i| holds(&seen_a, &session(1, i))).unwrap());
    assert_eq!(f.admin("remove_worker", &url_a, "&keep_affinity=1").await, 200);
    assert!(!f.workers().await.contains(&url_a));
    f.send(&s, 64).await;
    let during_b = seen_b.lock().unwrap().iter().filter(|x| **x == s).count();
    let during_c = seen_c.lock().unwrap().iter().filter(|x| **x == s).count();
    assert_eq!(during_b + during_c, 1, "served once by a surviving replica");
    for _ in 0..5 {
        f.send(&s, 64).await;
    }
    let after_b = seen_b.lock().unwrap().iter().filter(|x| **x == s).count();
    let after_c = seen_c.lock().unwrap().iter().filter(|x| **x == s).count();
    assert!(after_b == 0 || after_c == 0, "no hopping between survivors: {after_b}/{after_c}");
    assert_eq!(f.warming().await["entries"][0]["url"], url_c.as_str());

    let before = seen_a.lock().unwrap().len();
    assert_eq!(f.admin("add_worker", &url_a, "").await, 200);
    assert_eq!(f.warming().await["entries"].as_array().unwrap().len(), 1, "restore is no scale-out");
    f.send(&s, 64).await;
    assert_eq!(seen_a.lock().unwrap().len(), before + 1, "back on its original replica");

    // Evicted, then really gone: the later plain removal clears the tenants.
    assert_eq!(f.admin("remove_worker", &url_a, "&keep_affinity=1").await, 200);
    assert_eq!(f.admin("remove_worker", &url_a, "").await, 200);
    assert_eq!(f.admin("add_worker", &url_a, "").await, 200);
    let before = seen_a.lock().unwrap().len();
    f.send(&s, 64).await;
    assert_eq!(seen_a.lock().unwrap().len(), before, "a replaced replica gets nothing back");

    // Scale-in forgets the warming replica.
    assert_eq!(f.admin("remove_worker", &url_c, "").await, 200);
    assert!(f.warming().await["entries"].as_array().unwrap().is_empty());
    let served = seen_a.lock().unwrap().len() + seen_b.lock().unwrap().len();
    f.send(&session(5, 1), LARGE).await;
    assert_eq!(seen_a.lock().unwrap().len() + seen_b.lock().unwrap().len(), served + 1);
}
