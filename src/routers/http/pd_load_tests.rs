//! QPDLOAD20261118CLI: real HTTP dispatch, cancellation and policy regressions.
use super::*;
use crate::core::{DPAwareWorker, WorkerRegistry};
use axum::{extract::State, routing::post, Json};
use std::time::Duration;
use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::UnboundedReceiverStream;

type Chunk = Result<bytes::Bytes, std::io::Error>;
struct Call {
    rank: Option<usize>,
    request: Value,
    reply: oneshot::Sender<Response>,
}

async fn backend(
    State((tx, fixed_rank)): State<(mpsc::UnboundedSender<Call>, Option<usize>)>,
    headers: HeaderMap,
    Json(request): Json<Value>,
) -> Response {
    let (reply, rx) = oneshot::channel();
    tx.send(Call {
        rank: fixed_rank.or_else(|| {
            headers
                .get("x-data-parallel-rank")
                .map(|v| v.to_str().unwrap().parse().unwrap())
        }),
        request,
        reply,
    })
    .unwrap_or_else(|_| panic!("fixture receiver dropped"));
    rx.await
        .unwrap_or_else(|_| StatusCode::SERVICE_UNAVAILABLE.into_response())
}

async fn bounded<T>(future: impl std::future::Future<Output = T>) -> T {
    tokio::time::timeout(Duration::from_secs(3), future)
        .await
        .expect("fixture timed out")
}

fn streaming_reply(call: Call, status: StatusCode) -> mpsc::UnboundedSender<Chunk> {
    let (tx, rx) = mpsc::unbounded_channel();
    let response = Response::builder()
        .status(status)
        .header("content-type", "text/event-stream")
        .body(Body::from_stream(UnboundedReceiverStream::new(rx)))
        .unwrap();
    assert!(call.reply.send(response).is_ok());
    tx
}

fn prefill_done(call: Call) {
    assert!(call
        .reply
        .send(Json(json!({"kv_transfer_params": {"remote_engine_id": "p"}})).into_response())
        .is_ok());
}

struct Fixture {
    router: Arc<VllmPDRouter>,
    p: Arc<dyn Worker>,
    d: Vec<Arc<dyn Worker>>,
    pcalls: mpsc::UnboundedReceiver<Call>,
    dcalls: mpsc::UnboundedReceiver<Call>,
    server: tokio::task::JoinHandle<()>,
    load_mode: Arc<AtomicUsize>,
    health: mpsc::UnboundedReceiver<()>,
}
impl Drop for Fixture {
    fn drop(&mut self) {
        self.server.abort();
    }
}
impl Fixture {
    async fn new(push: bool) -> Self {
        Self::with_ranks(push, true).await
    }
    async fn with_ranks(push: bool, ranked: bool) -> Self {
        let (ptx, pcalls) = mpsc::unbounded_channel();
        let (dtx, dcalls) = mpsc::unbounded_channel();
        let load_mode = Arc::new(AtomicUsize::new(0));
        let (health_tx, health) = mpsc::unbounded_channel();
        let routes = |tx, fixed_rank| {
            let mode = load_mode.clone();
            let health_tx = health_tx.clone();
            axum::Router::new()
                .route("/{*path}", post(backend))
                .route(
                    "/get_load",
                    axum::routing::get(move || {
                        let mode = mode.clone();
                        async move {
                            match mode.load(Ordering::SeqCst) {
                                0 => Json(json!({"load":0})).into_response(),
                                1 => StatusCode::SERVICE_UNAVAILABLE.into_response(),
                                2 => Json(json!({"load":"invalid"})).into_response(),
                                3 => futures_util::future::pending::<Response>().await,
                                4 => Json(json!({"load":-1})).into_response(),
                                5 => "not JSON".into_response(),
                                _ => Json(json!({"missing_load":0})).into_response(),
                            }
                        }
                    }),
                )
                .route(
                    "/health",
                    axum::routing::get(move || {
                        let tx = health_tx.clone();
                        async move {
                            let _ = tx.send(());
                            StatusCode::OK
                        }
                    }),
                )
                .with_state((tx, fixed_rank))
        };
        let app = axum::Router::new()
            .nest("/p", routes(ptx, None))
            .nest("/d", routes(dtx.clone(), None))
            .nest("/d0", routes(dtx.clone(), Some(0)))
            .nest("/d1", routes(dtx, Some(1)));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let registry = Arc::new(WorkerRegistry::new());
        let p: Arc<dyn Worker> = Arc::new(BasicWorker::new(
            format!("http://{address}/p"),
            WorkerType::Prefill {
                bootstrap_port: None,
            },
        ));
        registry.register(p.clone());
        let d: Vec<Arc<dyn Worker>> = (0..2)
            .map(|rank| {
                if ranked {
                    Arc::new(DPAwareWorker::new(
                        format!("http://{address}/d"),
                        rank,
                        2,
                        WorkerType::Decode,
                    )) as Arc<dyn Worker>
                } else {
                    Arc::new(BasicWorker::new(
                        format!("http://{address}/d{rank}"),
                        WorkerType::Decode,
                    )) as Arc<dyn Worker>
                }
            })
            .collect();
        for worker in &d {
            registry.register(worker.clone());
        }
        let policies = Arc::new(PolicyRegistry::new(
            crate::config::PolicyConfig::PowerOfTwo {
                load_check_interval_secs: 1,
            },
        ));
        let client = reqwest::Client::builder()
            .no_proxy()
            .timeout(Duration::from_secs(10))
            .build()
            .unwrap();
        let router = VllmPDRouter {
            epd: None,
            pd_router: PdRouterBase {
                worker_registry: registry,
                policy_registry: policies.clone(),
                worker_startup_timeout_secs: 5,
                worker_startup_check_interval_secs: 1,
                worker_loads: Arc::new(tokio::sync::watch::channel(HashMap::new()).1),
                load_monitor_handle: None,
                client: client.clone(),
                circuit_breaker_config: crate::core::CircuitBreakerConfig::default(),
                dp_size: 2,
            },
            service_registry: Arc::new(ServiceRegistry::new()),
            http_client: client,
            policy_registry: policies,
            use_discovery: false,
            enable_profiling: false,
            profile_timeout_secs: 1,
            profiling_tasks: Arc::new(Mutex::new(HashMap::new())),
            intra_node_data_parallel_size: 2,
            prefill_dp_round_robin: Arc::new(AtomicUsize::new(0)),
            kv_connector: KvConnector::Nixl,
            mooncake_prefill_info: Arc::new(Mutex::new(HashMap::new())),
            nixl_prefill_info: RwLock::new(HashMap::new()),
        };
        if push {
            // Exactly the cached push identity gate in try_build_concurrent_requests.
            router.store_nixl_info(p.base_url(), None, json!({"transfer_mode":"push", "remote_engine_id":"p", "remote_host":"127.0.0.1", "remote_port":9100}));
        }
        Self {
            router: Arc::new(router),
            p,
            d,
            pcalls,
            dcalls,
            server,
            load_mode,
            health,
        }
    }
    fn start(&self, stream: bool, logprobs: bool) -> tokio::task::JoinHandle<Response> {
        let router = self.router.clone();
        tokio::spawn(async move {
            let mut request = json!({"model":"test", "prompt":"hello", "stream":stream});
            if logprobs {
                request["logprobs"] = json!(1);
            }
            router
                .route_transparent(None, "/v1/completions", &Method::POST, request)
                .await
        })
    }
    async fn stages(&mut self, push: bool) -> Call {
        let p = bounded(self.pcalls.recv()).await.unwrap();
        if push {
            // D must be dispatched while P has neither headers nor body.
            let d = bounded(self.dcalls.recv()).await.unwrap();
            assert!(d.request["kv_transfer_params"]["remote_request_id"].is_string());
            prefill_done(p);
            d
        } else {
            assert!(matches!(
                self.dcalls.try_recv(),
                Err(mpsc::error::TryRecvError::Empty)
            ));
            prefill_done(p);
            bounded(self.dcalls.recv()).await.unwrap()
        }
    }
}

#[tokio::test]
async fn pd_load_policy_dispatch_and_reset_preserve_other_request() {
    for push in [false, true] {
        let mut f = Fixture::new(push).await;
        let a = f.start(true, false);
        let d = f.stages(push).await;
        let rank_a = d.rank.unwrap();
        let upstream_a = streaming_reply(d, StatusCode::OK);
        let response_a = bounded(a).await.unwrap();
        assert_eq!(f.p.load(), 0);
        assert_eq!(f.d[rank_a].load(), 1);
        assert!(crate::core::worker_registry::reset_drifted_loads(
            &f.router.pd_router.worker_registry.get_all()
        ));
        assert_eq!(f.d[rank_a].load(), 1, "one long stream must survive reset");
        let b = f.start(true, false);
        let d = f.stages(push).await;
        assert_eq!(
            d.rank,
            Some(1 - rank_a),
            "actual power_of_two HTTP dispatch must select idle rank"
        );
        let upstream_b = streaming_reply(d, StatusCode::OK);
        let response_b = bounded(b).await.unwrap();
        crate::core::worker_registry::reset_drifted_loads(
            &f.router.pd_router.worker_registry.get_all(),
        );
        drop(response_a);
        bounded(upstream_a.closed()).await; // no next upstream chunk needed to cancel
        assert_eq!(f.d[rank_a].load(), 0);
        assert_eq!(f.d[1 - rank_a].load(), 1);
        drop(response_b);
        bounded(upstream_b.closed()).await;
        assert!(f.d.iter().all(|w| w.load() == 0));
    }
}

#[tokio::test]
async fn pd_load_reset_same_rank_a_finishes_after_b_arrives() {
    for push in [false, true] {
        let mut f = Fixture::new(push).await;
        // Keep just one rank to exercise A and B sharing the very same counter.
        f.router
            .pd_router
            .worker_registry
            .remove_by_url(f.d[1].url());
        let a = f.start(true, false);
        let d = f.stages(push).await;
        let txa = streaming_reply(d, StatusCode::OK);
        let a = bounded(a).await.unwrap();
        crate::core::worker_registry::reset_drifted_loads(&[f.p.clone(), f.d[0].clone()]);
        let b = f.start(true, false);
        let d = f.stages(push).await;
        let txb = streaming_reply(d, StatusCode::OK);
        let b = bounded(b).await.unwrap();
        assert_eq!(f.d[0].load(), 2);
        crate::core::worker_registry::reset_drifted_loads(&[f.p.clone(), f.d[0].clone()]);
        drop(a);
        bounded(txa.closed()).await;
        assert_eq!(
            f.d[0].load(),
            1,
            "A releases only its reservation after reset"
        );
        drop(b);
        bounded(txb.closed()).await;
        assert_eq!(f.d[0].load(), 0);
    }
}

#[tokio::test]
async fn pd_load_reserves_decode_before_prefill_and_cancels_before_headers() {
    for push in [false, true] {
        for stream in [false, true] {
            let mut f = Fixture::new(push).await;
            let task = f.start(stream, false);
            let p = bounded(f.pcalls.recv()).await.unwrap();
            let d = if push {
                Some(bounded(f.dcalls.recv()).await.unwrap())
            } else {
                None
            };
            assert_eq!(f.p.load(), 1);
            assert_eq!(f.d.iter().map(|w| w.load()).sum::<usize>(), 1);
            crate::core::worker_registry::reset_drifted_loads(
                &f.router.pd_router.worker_registry.get_all(),
            );
            assert_eq!(f.p.load(), 1, "P is also exempt from drift reset");
            task.abort();
            assert!(task.await.unwrap_err().is_cancelled());
            assert_eq!(f.p.load(), 0);
            assert!(f.d.iter().all(|w| w.load() == 0));
            drop((p, d));
        }
    }
}

#[tokio::test]
async fn pd_load_prefill_body_and_decode_header_cancellation() {
    for push in [false, true] {
        let mut f = Fixture::new(push).await;
        let task = f.start(true, false);
        let p = bounded(f.pcalls.recv()).await.unwrap();
        let d = if push {
            Some(bounded(f.dcalls.recv()).await.unwrap())
        } else {
            None
        };
        let pbody = streaming_reply(p, StatusCode::OK);
        // Yield via a backend body chunk; P headers alone cannot release its load.
        pbody.send(Ok(bytes::Bytes::from_static(b"{"))).unwrap();
        assert_eq!(f.p.load(), 1);
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        bounded(pbody.closed()).await;
        assert_eq!(f.p.load(), 0);
        assert!(f.d.iter().all(|w| w.load() == 0));
        drop(d);

        let task = f.start(true, false);
        let d = f.stages(push).await;
        bounded(async {
            while f.p.load() != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await;
        assert_eq!(f.p.load(), 0, "P complete even while D headers stall");
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        assert!(f.d.iter().all(|w| w.load() == 0));
        drop(d);
    }
}

#[tokio::test]
async fn pd_load_decode_eof_drop_error_and_buffer_cancellation() {
    for push in [false, true] {
        for (stream, logprobs) in [(true, false), (false, false), (false, true)] {
            for outcome in ["eof", "cancel", "error", "non2xx"] {
                let mut f = Fixture::new(push).await;
                let task = f.start(stream, logprobs);
                let d = f.stages(push).await;
                let rank = d.rank.unwrap();
                // A separate in-flight reservation must survive every exit exactly once.
                f.d[rank].increment_load();
                let status = if outcome == "non2xx" {
                    StatusCode::TOO_MANY_REQUESTS
                } else {
                    StatusCode::OK
                };
                let upstream = streaming_reply(d, status);
                if outcome == "non2xx" {
                    let response = bounded(task).await.unwrap();
                    assert_eq!(response.status(), status);
                    assert_eq!(f.d[rank].load(), 1);
                    drop(response);
                } else if outcome == "cancel" && !stream {
                    // finalize_decode_response buffers JSON / merges logprobs here.
                    upstream.send(Ok(bytes::Bytes::from_static(b"{"))).unwrap();
                    task.abort();
                    assert!(task.await.unwrap_err().is_cancelled());
                } else if stream {
                    let response = bounded(task).await.unwrap();
                    assert_eq!(f.d[rank].load(), 2);
                    if outcome == "cancel" {
                        drop(response);
                    } else {
                        if outcome == "error" {
                            upstream
                                .send(Err(std::io::Error::other("fixture body error")))
                                .unwrap();
                        } else {
                            upstream
                                .send(Ok(bytes::Bytes::from_static(b"data: [DONE]\n\n")))
                                .unwrap();
                        }
                        drop(upstream);
                        let result =
                            bounded(axum::body::to_bytes(response.into_body(), 4096)).await;
                        assert_eq!(result.is_ok(), outcome == "eof");
                        assert_eq!(f.d[rank].load(), 1);
                        f.d[rank].decrement_load();
                        continue;
                    }
                } else {
                    if outcome == "error" {
                        upstream
                            .send(Err(std::io::Error::other("fixture body error")))
                            .unwrap();
                    } else {
                        upstream
                            .send(Ok(bytes::Bytes::from_static(b"{\"choices\":[]}")))
                            .unwrap();
                    }
                    drop(upstream);
                    let response = bounded(task).await.unwrap();
                    if outcome == "error" {
                        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
                    } else {
                        // The rewritten body owns D until the downstream finishes/drops it.
                        assert_eq!(f.d[rank].load(), 2);
                        bounded(axum::body::to_bytes(response.into_body(), 4096))
                            .await
                            .unwrap();
                    }
                    assert_eq!(f.d[rank].load(), 1);
                    f.d[rank].decrement_load();
                    continue;
                }
                bounded(upstream.closed()).await;
                assert_eq!(f.d[rank].load(), 1, "{push}/{stream}/{logprobs}/{outcome}");
                f.d[rank].decrement_load();
                assert_eq!(f.p.load(), 0);
            }
        }
    }
}

#[tokio::test]
async fn pd_load_prefill_errors_release_both_reservations() {
    for push in [false, true] {
        for failure in ["non2xx", "body"] {
            let mut f = Fixture::new(push).await;
            let task = f.start(true, false);
            let p = bounded(f.pcalls.recv()).await.unwrap();
            let d = if push {
                Some(bounded(f.dcalls.recv()).await.unwrap())
            } else {
                None
            };
            let pbody = streaming_reply(
                p,
                if failure == "non2xx" {
                    StatusCode::BAD_REQUEST
                } else {
                    StatusCode::OK
                },
            );
            if failure == "body" {
                pbody
                    .send(Err(std::io::Error::other("P body error")))
                    .unwrap();
            }
            drop(pbody);
            if let Some(d) = d {
                assert!(d.reply.send(Json(json!({})).into_response()).is_ok());
            }
            let response = bounded(task).await.unwrap();
            assert_eq!(
                response.status(),
                if failure == "non2xx" {
                    StatusCode::BAD_REQUEST
                } else {
                    StatusCode::INTERNAL_SERVER_ERROR
                }
            );
            assert_eq!(f.p.load(), 0);
            assert!(f.d.iter().all(|w| w.load() == 0));
        }
    }
}

#[tokio::test]
async fn pd_load_failed_external_sampling_falls_back_to_real_dispatch() {
    for push in [false, true] {
        for failure_mode in 1..=6 {
            let mut f = Fixture::with_ranks(push, false).await;
            let client = reqwest::Client::builder()
                .no_proxy()
                .timeout(Duration::from_millis(100))
                .build()
                .unwrap();
            let registry = &f.router.pd_router.worker_registry;
            let policy = f.router.policy_registry.get_decode_policy();
            let (tx, mut rx) = tokio::sync::watch::channel(HashMap::new());
            let monitor = tokio::spawn(PdRouterBase::monitor_worker_loads_with_client(
                registry.clone(),
                tx,
                0,
                client,
                policy.clone(),
                policy.clone(),
            ));
            bounded(rx.changed()).await.unwrap();
            assert_eq!(
                rx.borrow().get(f.d[0].url()),
                Some(&0),
                "real zero must be cached"
            );
            assert_eq!(rx.borrow().get(f.d[1].url()), Some(&0));
            f.load_mode.store(failure_mode, Ordering::SeqCst);
            bounded(async {
                loop {
                    rx.changed().await.unwrap();
                    if rx.borrow().is_empty() {
                        break;
                    }
                }
            })
            .await;
            // Freeze the actual monitor's failed round before exercising fallback.
            monitor.abort();
            assert!(monitor.await.unwrap_err().is_cancelled());
            let a = f.start(true, false);
            let d = f.stages(push).await;
            let rank = d.rank.unwrap();
            let txa = streaming_reply(d, StatusCode::OK);
            let a = bounded(a).await.unwrap();
            let b = f.start(true, false);
            let d = f.stages(push).await;
            assert_eq!(
                d.rank,
                Some(1 - rank),
                "stale zero must not mask A's local reservation"
            );
            let txb = streaming_reply(d, StatusCode::OK);
            let b = bounded(b).await.unwrap();
            drop((a, b));
            bounded(txa.closed()).await;
            bounded(txb.closed()).await;
        }
    }
}

#[tokio::test]
async fn pd_load_dynamic_registry_sampling_never_fabricates_rank_samples() {
    let f = Fixture::new(false).await;
    let registry = &f.router.pd_router.worker_registry;
    let client = &f.router.pd_router.client;
    let loads = PdRouterBase::sample_worker_loads(registry, client).await;
    assert_eq!(loads.len(), 1, "only unranked P has a server-wide sample");
    assert!(f.d.iter().all(|w| !loads.contains_key(w.url())));
    let dynamic: Arc<dyn Worker> = Arc::new(BasicWorker::new(
        f.d[0].base_url().to_owned(),
        WorkerType::Decode,
    ));
    registry.register(dynamic.clone());
    let loads = PdRouterBase::sample_worker_loads(registry, client).await;
    assert_eq!(
        loads.get(dynamic.url()),
        Some(&0),
        "new workers sampled without restarting monitor"
    );
    assert!(
        f.d.iter().all(|w| !loads.contains_key(w.url())),
        "pod aggregate must not be copied to ranks"
    );
    registry.remove_by_url(dynamic.url());
    registry.remove_by_url(f.p.url());
    assert!(
        PdRouterBase::sample_worker_loads(registry, client)
            .await
            .is_empty(),
        "removed keys expire next round"
    );
}

#[tokio::test]
async fn pd_load_legacy_health_checker_preserves_low_concurrency() {
    let mut f = Fixture::new(false).await;
    let a = f.start(true, false);
    let d = f.stages(false).await;
    let rank = d.rank.unwrap();
    let txa = streaming_reply(d, StatusCode::OK);
    let a = bounded(a).await.unwrap();
    let workers = f.router.pd_router.worker_registry.get_all();
    let count = workers.len();
    let checker = crate::core::worker::start_health_checker(Arc::new(RwLock::new(workers)), 1);
    tokio::time::timeout(Duration::from_secs(12), async {
        // The tenth real health-check round runs the legacy reset entry point.
        for _ in 0..10 * count {
            f.health.recv().await.unwrap();
        }
    })
    .await
    .expect("legacy health checker never reached reset round");
    assert_eq!(f.d[rank].load(), 1);
    let b = f.start(true, false);
    let d = f.stages(false).await;
    assert_eq!(d.rank, Some(1 - rank));
    let txb = streaming_reply(d, StatusCode::OK);
    let b = bounded(b).await.unwrap();
    drop(a);
    bounded(txa.closed()).await;
    assert_eq!(f.d[1 - rank].load(), 1);
    drop(b);
    bounded(txb.closed()).await;
    checker.shutdown().await;
}

#[tokio::test]
async fn pd_load_reservation_follows_selected_worker_after_registry_replacement() {
    for push in [false, true] {
        let mut f = Fixture::new(push).await;
        let a = f.start(true, false);
        let d = f.stages(push).await;
        let rank = d.rank.unwrap();
        let txa = streaming_reply(d, StatusCode::OK);
        let a = bounded(a).await.unwrap();
        let registry = &f.router.pd_router.worker_registry;
        registry.remove_by_url(f.d[rank].url());
        registry.remove_by_url(f.d[1 - rank].url());
        let replacement: Arc<dyn Worker> = Arc::new(DPAwareWorker::new(
            f.d[rank].base_url().to_owned(),
            rank,
            2,
            WorkerType::Decode,
        ));
        registry.register(replacement.clone());
        let b = f.start(true, false);
        let d = f.stages(push).await;
        assert_eq!(d.rank, Some(rank));
        let txb = streaming_reply(d, StatusCode::OK);
        let b = bounded(b).await.unwrap();
        assert_eq!(replacement.load(), 1);
        drop(a);
        bounded(txa.closed()).await;
        assert_eq!(f.d[rank].load(), 0);
        assert_eq!(
            replacement.load(),
            1,
            "old guard owns the selected Arc, not a registry URL lookup"
        );
        drop(b);
        bounded(txb.closed()).await;
        assert_eq!(replacement.load(), 0);
    }
}

#[tokio::test]
async fn pd_load_waiting_prefill_reservation_changes_actual_decode_dispatch() {
    for push in [false, true] {
        let mut f = Fixture::new(push).await;
        let a = f.start(true, false);
        let pa = bounded(f.pcalls.recv()).await.unwrap();
        let da = if push {
            Some(bounded(f.dcalls.recv()).await.unwrap())
        } else {
            None
        };
        let b = f.start(true, false);
        let pb = bounded(f.pcalls.recv()).await.unwrap();
        let db = if push {
            Some(bounded(f.dcalls.recv()).await.unwrap())
        } else {
            None
        };
        assert!(
            f.d.iter().all(|w| w.load() == 1),
            "B must reserve the other D while A awaits P"
        );
        prefill_done(pa);
        prefill_done(pb);
        let da = match da {
            Some(d) => d,
            None => bounded(f.dcalls.recv()).await.unwrap(),
        };
        let db = match db {
            Some(d) => d,
            None => bounded(f.dcalls.recv()).await.unwrap(),
        };
        assert_ne!(
            da.rank, db.rank,
            "prove actual HTTP rank dispatch, not just local counters"
        );
        let txa = streaming_reply(da, StatusCode::OK);
        let txb = streaming_reply(db, StatusCode::OK);
        drop((bounded(a).await.unwrap(), bounded(b).await.unwrap()));
        bounded(txa.closed()).await;
        bounded(txb.closed()).await;
        assert!(f.d.iter().all(|w| w.load() == 0));
    }
}

#[tokio::test]
async fn pd_load_concurrent_decode_non2xx_releases_before_prefill_finishes() {
    let mut f = Fixture::new(true).await;
    let task = f.start(true, false);
    let p = bounded(f.pcalls.recv()).await.unwrap();
    let d = bounded(f.dcalls.recv()).await.unwrap();
    let rank = d.rank.unwrap();
    let upstream = streaming_reply(d, StatusCode::TOO_MANY_REQUESTS);
    bounded(async {
        while f.d[rank].load() != 0 {
            tokio::task::yield_now().await;
        }
    })
    .await;
    assert_eq!(f.p.load(), 1);
    prefill_done(p);
    let response = bounded(task).await.unwrap();
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    drop(response);
    bounded(upstream.closed()).await;
    assert_eq!(f.p.load(), 0);
    assert_eq!(f.d[rank].load(), 0);
}

#[tokio::test]
async fn pd_load_prefill_non2xx_body_cancellation_keeps_prefill_reserved() {
    for push in [false, true] {
        let mut f = Fixture::new(push).await;
        let task = f.start(true, false);
        let p = bounded(f.pcalls.recv()).await.unwrap();
        let d = if push {
            Some(bounded(f.dcalls.recv()).await.unwrap())
        } else {
            None
        };
        let upstream = streaming_reply(p, StatusCode::BAD_REQUEST);
        upstream
            .send(Ok(bytes::Bytes::from_static(b"partial error detail")))
            .unwrap();
        let decode_body = d.map(|d| streaming_reply(d, StatusCode::OK));
        if !push {
            // Sequential D is released when P rejects it, but P's error detail
            // is still being read. Cancellation must drop that body too.
            bounded(async {
                while f.d.iter().any(|w| w.load() != 0) {
                    tokio::task::yield_now().await;
                }
            })
            .await;
        }
        assert_eq!(f.p.load(), 1);
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        bounded(upstream.closed()).await;
        if let Some(body) = decode_body {
            bounded(body.closed()).await;
        }
        assert_eq!(f.p.load(), 0);
        assert!(f.d.iter().all(|w| w.load() == 0));
    }
}

#[tokio::test]
async fn pd_load_concurrent_prefill_5xx_does_not_wait_for_error_body() {
    let mut f = Fixture::new(true).await;
    f.router
        .pd_router
        .worker_registry
        .remove_by_url(f.d[1].url());

    // A separate real request owns one reservation on each of the same workers.
    let other = f.start(true, false);
    let p = bounded(f.pcalls.recv()).await.unwrap();
    let d = bounded(f.dcalls.recv()).await.unwrap();
    let other_prefill = streaming_reply(p, StatusCode::OK);
    let other_decode = streaming_reply(d, StatusCode::OK);
    assert_eq!((f.p.load(), f.d[0].load()), (1, 1));

    let mut failing = f.start(true, false);
    let p = bounded(f.pcalls.recv()).await.unwrap();
    let d = bounded(f.dcalls.recv()).await.unwrap();
    assert_eq!((f.p.load(), f.d[0].load()), (2, 2));
    let failed_prefill = streaming_reply(p, StatusCode::SERVICE_UNAVAILABLE);
    let failed_decode = streaming_reply(d, StatusCode::OK);
    // Both bodies stay open without sending a chunk. The client's timeout is
    // ten seconds, so this must fail because of P's status rather than timeout.
    let result = tokio::time::timeout(Duration::from_secs(1), &mut failing).await;
    if result.is_err() {
        failing.abort();
        let _ = failing.await;
        other.abort();
        let _ = other.await;
        panic!("P 503 headers must fail without waiting for its error body");
    }
    let response = result.unwrap().unwrap();
    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    bounded(failed_prefill.closed()).await;
    bounded(failed_decode.closed()).await;
    assert_eq!((f.p.load(), f.d[0].load()), (1, 1));
    drop(response);
    assert_eq!((f.p.load(), f.d[0].load()), (1, 1));

    other.abort();
    assert!(other.await.unwrap_err().is_cancelled());
    bounded(other_prefill.closed()).await;
    bounded(other_decode.closed()).await;
    assert_eq!((f.p.load(), f.d[0].load()), (0, 0));
}

#[tokio::test]
async fn pd_load_decode_error_releases_before_downstream_body_is_dropped() {
    use http_body_util::BodyExt;

    for push in [false, true] {
        let mut f = Fixture::new(push).await;
        f.router
            .pd_router
            .worker_registry
            .remove_by_url(f.d[1].url());
        let a = f.start(true, false);
        let d = f.stages(push).await;
        let upstream_a = streaming_reply(d, StatusCode::OK);
        let mut body = bounded(a).await.unwrap().into_body();
        let b = f.start(true, false);
        let d = f.stages(push).await;
        let upstream_b = streaming_reply(d, StatusCode::OK);
        let response_b = bounded(b).await.unwrap();
        assert_eq!(f.d[0].load(), 2);

        upstream_a
            .send(Err(std::io::Error::other("fixture body error")))
            .unwrap();
        assert!(bounded(body.frame()).await.unwrap().is_err());
        assert_eq!(f.d[0].load(), 1, "Err must release A while Body is alive");
        drop(body);
        assert_eq!(f.d[0].load(), 1, "dropping A must not also release B");

        drop(response_b);
        bounded(upstream_b.closed()).await;
        assert_eq!((f.p.load(), f.d[0].load()), (0, 0));
    }
}
