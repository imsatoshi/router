//! lingjing patch (0006): instance-level load summary and saturation-aware selection.
//!
//! The gateway used to probe `<router>/metrics` for engine saturation. That path is not a
//! router route, so the transparent-proxy fallback forwarded it to one worker (the routing
//! text of an empty GET is "null", so always the same worker): the gateway saw one replica
//! of N and every probe was counted as a processed request. Instead the router samples its
//! own workers' `/metrics` (inside the instance, every `LINGJING_LOAD_INTERVAL_MS`) and
//! serves one summary at `GET /lingjing/load`; the gateway asks once per instance.
//!
//! Per replica: `saturated_now = (kv >= K && waiting >= 1) || waiting >= W_hard ||
//! (max_num_seqs > 0 && router_inflight >= max_num_seqs)`, with 2-sample hysteresis for
//! `saturated`. A replica whose scrape fails keeps its last verdict (fail-closed) and is
//! not counted as fresh; a replica never sampled is unsaturated (fail-open).
//!
//! With `LINGJING_LOAD_AVOID_SATURATED=1`, cache_aware steers *new* sessions away from
//! saturated replicas (sessions that hit an existing prefix keep their affinity); when
//! every replica is saturated nothing is avoided. No request is ever rejected here.
//!
//! Environment (all optional): LINGJING_LOAD_INTERVAL_MS (default 2000, 0 disables the
//! feature and `/lingjing/load` returns 404), LINGJING_SAT_KV (0.85), LINGJING_SAT_WAITING
//! (0 = auto: max(8, max_num_seqs / 4)), LINGJING_MAX_NUM_SEQS (0 = unknown),
//! LINGJING_LOAD_AVOID_SATURATED (0).

use crate::core::{Worker, WorkerRegistry};
use crate::program_scheduling::{
    BackendObservationProvider, ProgramTarget, VllmMetricsObservationProvider,
};
use crate::routers::http::dp_utils;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tracing::{info, warn};

/// Consecutive samples needed to flip a replica's hysteresis verdict either way.
const HYSTERESIS: u8 = 2;
/// A replica is fresh while its last successful sample is within this many intervals.
const FRESH_INTERVALS: u32 = 3;

#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub interval_ms: u64,
    pub sat_kv: f64,
    /// 0 = automatic.
    pub sat_waiting: usize,
    /// 0 = unknown.
    pub max_num_seqs: usize,
    pub avoid_saturated: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            interval_ms: 2000,
            sat_kv: 0.85,
            sat_waiting: 0,
            max_num_seqs: 0,
            avoid_saturated: false,
        }
    }
}

impl Config {
    pub fn from_env() -> Self {
        fn var<T: std::str::FromStr>(name: &str) -> Option<T> {
            std::env::var(name).ok().and_then(|v| v.trim().parse().ok())
        }
        let d = Config::default();
        Config {
            interval_ms: var("LINGJING_LOAD_INTERVAL_MS").unwrap_or(d.interval_ms),
            sat_kv: var("LINGJING_SAT_KV").unwrap_or(d.sat_kv),
            sat_waiting: var("LINGJING_SAT_WAITING").unwrap_or(d.sat_waiting),
            max_num_seqs: var("LINGJING_MAX_NUM_SEQS").unwrap_or(d.max_num_seqs),
            avoid_saturated: matches!(
                std::env::var("LINGJING_LOAD_AVOID_SATURATED")
                    .as_deref()
                    .map(str::trim),
                Ok("1") | Ok("true")
            ),
        }
    }

    pub fn waiting_hard(&self) -> usize {
        if self.sat_waiting > 0 {
            self.sat_waiting
        } else {
            8.max(self.max_num_seqs / 4)
        }
    }

    fn fresh_window(&self) -> Duration {
        Duration::from_millis(self.interval_ms) * FRESH_INTERVALS
    }

    /// The instantaneous verdict for one sample.
    pub fn saturated_now(&self, waiting: usize, kv: Option<f64>, inflight: usize) -> bool {
        kv.is_some_and(|kv| kv >= self.sat_kv && waiting >= 1)
            || waiting >= self.waiting_hard()
            || (self.max_num_seqs > 0 && inflight >= self.max_num_seqs)
    }
}

#[derive(Debug, Clone, Default)]
struct Replica {
    last_ok: Option<Instant>,
    running: Option<usize>,
    waiting: Option<usize>,
    kv: Option<f64>,
    saturated_now: bool,
    saturated: bool,
    hot: u8,
    cold: u8,
}

impl Replica {
    fn observe(
        &mut self,
        now: Instant,
        running: Option<usize>,
        waiting: Option<usize>,
        kv: Option<f64>,
        verdict: bool,
    ) {
        self.last_ok = Some(now);
        self.running = running;
        self.waiting = waiting;
        self.kv = kv;
        self.saturated_now = verdict;
        if verdict {
            self.cold = 0;
            self.hot = self.hot.saturating_add(1);
            if self.hot >= HYSTERESIS {
                self.saturated = true;
            }
        } else {
            self.hot = 0;
            self.cold = self.cold.saturating_add(1);
            if self.cold >= HYSTERESIS {
                self.saturated = false;
            }
        }
    }

    fn fresh(&self, now: Instant, window: Duration) -> bool {
        self.last_ok
            .is_some_and(|t| now.duration_since(t) <= window)
    }
}

/// Sampled state for every worker URL the registry currently holds.
#[derive(Debug)]
pub struct LoadState {
    cfg: Config,
    replicas: Mutex<HashMap<String, Replica>>,
    sampled_at_ms: Mutex<u64>,
}

impl LoadState {
    pub fn new(cfg: Config) -> Self {
        Self {
            cfg,
            replicas: Mutex::new(HashMap::new()),
            sampled_at_ms: Mutex::new(0),
        }
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// Run one sampling round against the registry's current workers.
    pub async fn sample(
        &self,
        registry: &WorkerRegistry,
        provider: &dyn BackendObservationProvider,
    ) {
        let workers = http_workers(registry);
        let targets: Vec<ProgramTarget> = workers
            .iter()
            .filter(|w| w.is_available())
            .map(|w| {
                let (base_url, rank) = dp_utils::parse_worker_url(w.url());
                ProgramTarget {
                    id: w.url().to_string(),
                    base_url,
                    dp_rank: w.dp_rank().or(rank),
                }
            })
            .collect();
        let (observations, failures) = if targets.is_empty() {
            (Vec::new(), Vec::new())
        } else {
            provider.observe(&targets).await
        };
        for failure in &failures {
            warn!(base_url = %failure.base_url, error = %failure.error, "lingjing load sample failed");
        }
        let now = Instant::now();
        let by_id: HashMap<&str, _> = observations
            .iter()
            .map(|o| (o.target_id.as_str(), o))
            .collect();
        let mut replicas = self.replicas.lock().unwrap_or_else(|e| e.into_inner());
        // Follow the registry: forget workers it no longer holds.
        replicas.retain(|url, _| workers.iter().any(|w| w.url() == url));
        for worker in &workers {
            let entry = replicas.entry(worker.url().to_string()).or_default();
            if let Some(obs) = by_id.get(worker.url()) {
                let waiting = obs.waiting_requests.unwrap_or(0);
                let verdict = self
                    .cfg
                    .saturated_now(waiting, obs.kv_cache_usage, worker.load());
                entry.observe(
                    now,
                    obs.running_requests,
                    obs.waiting_requests,
                    obs.kv_cache_usage,
                    verdict,
                );
            }
            // No observation: keep the previous verdict (fail-closed); freshness lapses on its own.
            metrics::gauge!("vllm_router_lingjing_replica_saturated", "worker" => worker.url().to_string())
                .set(if entry.saturated { 1.0 } else { 0.0 });
        }
        drop(replicas);
        let wall = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        *self.sampled_at_ms.lock().unwrap_or_else(|e| e.into_inner()) = wall;
        // Sample age = time() - this, computed by Prometheus (a gauge set at sample time cannot age).
        metrics::gauge!("vllm_router_lingjing_load_last_sample_timestamp_seconds")
            .set(wall as f64 / 1000.0);
    }

    /// True when new sessions should avoid this worker.
    pub fn is_saturated(&self, url: &str) -> bool {
        self.replicas
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(url)
            .is_some_and(|r| r.saturated)
    }

    /// The `/lingjing/load` document.
    pub fn summary(&self, registry: &WorkerRegistry) -> Value {
        let now = Instant::now();
        let window = self.cfg.fresh_window();
        let replicas = self.replicas.lock().unwrap_or_else(|e| e.into_inner());
        let sampled_at_ms = *self.sampled_at_ms.lock().unwrap_or_else(|e| e.into_inner());
        let mut items = Vec::new();
        let (mut fresh, mut saturated, mut running_sum, mut waiting_sum) =
            (0usize, 0usize, 0usize, 0usize);
        let mut kvs = Vec::new();
        let available: Vec<_> = http_workers(registry)
            .into_iter()
            .filter(|w| w.is_available())
            .collect();
        for worker in &available {
            let r = replicas.get(worker.url()).cloned().unwrap_or_default();
            let is_fresh = r.fresh(now, window);
            if is_fresh {
                fresh += 1;
                running_sum += r.running.unwrap_or(0);
                waiting_sum += r.waiting.unwrap_or(0);
                if let Some(kv) = r.kv {
                    kvs.push(kv);
                }
            }
            if r.saturated {
                saturated += 1;
            }
            items.push(json!({
                "id": replica_id(worker.url()),
                "fresh": is_fresh,
                "saturated": r.saturated,
                "saturated_now": r.saturated_now,
                "running": r.running,
                "waiting": r.waiting,
                "kv": r.kv,
                "router_inflight": worker.load(),
                "max_num_seqs": self.cfg.max_num_seqs,
                "age_ms": r.last_ok.map(|t| now.duration_since(t).as_millis() as u64),
            }));
        }
        let kv_mean = (!kvs.is_empty()).then(|| kvs.iter().sum::<f64>() / kvs.len() as f64);
        let kv_min = kvs.iter().copied().reduce(f64::min);
        let kv_max = kvs.iter().copied().reduce(f64::max);
        json!({
            "schema": 1,
            "registry_revision": registry.revision(),
            "sampled_at_ms": sampled_at_ms,
            "interval_ms": self.cfg.interval_ms,
            "replicas": items,
            "available": available.len(),
            "fresh": fresh,
            "saturated": saturated,
            "all_saturated": !available.is_empty() && saturated == available.len(),
            "running_sum": running_sum,
            "waiting_sum": waiting_sum,
            "kv_mean": kv_mean,
            "kv_min": kv_min,
            "kv_max": kv_max,
            "thresholds": {"waiting_hard": self.cfg.waiting_hard(), "kv": self.cfg.sat_kv},
        })
    }
}

fn http_workers(registry: &WorkerRegistry) -> Vec<Arc<dyn Worker>> {
    registry
        .get_all()
        .into_iter()
        .filter(|w| !crate::backend::is_grpc_url(w.url()))
        .collect()
}

/// Stable, address-free replica id: first 8 hex chars of sha256(worker url).
pub fn replica_id(url: &str) -> String {
    let digest = Sha256::digest(url.as_bytes());
    digest.iter().take(4).map(|b| format!("{b:02x}")).collect()
}

static STATE: OnceLock<Arc<LoadState>> = OnceLock::new();
static STARTED: AtomicBool = AtomicBool::new(false);

/// The process-wide state, when the feature is enabled.
pub fn state() -> Option<&'static Arc<LoadState>> {
    STATE.get()
}

/// Steer new sessions away from `url`? Only with LINGJING_LOAD_AVOID_SATURATED.
pub fn avoid(url: &str) -> bool {
    STATE
        .get()
        .is_some_and(|s| s.cfg.avoid_saturated && s.is_saturated(url))
}

pub fn record_avoided() {
    metrics::counter!("vllm_router_lingjing_new_session_avoided_saturated_total").increment(1);
}

/// Start the sampler from the environment (idempotent).
pub fn start_from_env(
    registry: Arc<WorkerRegistry>,
    client: reqwest::Client,
    api_key: Option<String>,
) {
    let cfg = Config::from_env();
    if cfg.interval_ms == 0 || STARTED.swap(true, Ordering::SeqCst) {
        return;
    }
    info!(?cfg, "lingjing load sampler started");
    let state = Arc::new(LoadState::new(cfg));
    let _ = STATE.set(state.clone());
    let timeout = Duration::from_millis(state.cfg.interval_ms.clamp(500, 1000));
    let provider =
        VllmMetricsObservationProvider::new(client, api_key).with_request_timeout(timeout);
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_millis(state.cfg.interval_ms));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            interval.tick().await;
            state.sample(&registry, &provider).await;
        }
    });
}

#[cfg(test)]
pub(crate) fn test_global() -> &'static Arc<LoadState> {
    STATE.get_or_init(|| {
        Arc::new(LoadState::new(Config {
            avoid_saturated: true,
            ..Config::default()
        }))
    })
}

#[cfg(test)]
impl LoadState {
    /// Pin a replica's hysteresis verdict (cache_aware tests).
    pub(crate) fn force(&self, url: &str, saturated: bool) {
        let mut replicas = self.replicas.lock().unwrap_or_else(|e| e.into_inner());
        replicas.entry(url.to_string()).or_default().saturated = saturated;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorker, WorkerType};
    use std::sync::Mutex as StdMutex;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    /// A fake vLLM `/metrics` endpoint whose body (or failure) the test controls.
    struct FakeEngine {
        url: String,
        body: Arc<StdMutex<Option<String>>>,
        hits: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl FakeEngine {
        async fn start() -> Self {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let url = format!("http://{}", listener.local_addr().unwrap());
            let body: Arc<StdMutex<Option<String>>> = Arc::new(StdMutex::new(None));
            let hits = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let (b, h) = (body.clone(), hits.clone());
            tokio::spawn(async move {
                loop {
                    let Ok((mut socket, _)) = listener.accept().await else {
                        return;
                    };
                    let (b, h) = (b.clone(), h.clone());
                    tokio::spawn(async move {
                        let mut buf = Vec::new();
                        let mut byte = [0u8; 1];
                        while !buf.ends_with(b"\r\n\r\n") {
                            if socket.read(&mut byte).await.unwrap_or(0) == 0 {
                                return;
                            }
                            buf.push(byte[0]);
                        }
                        h.fetch_add(1, Ordering::SeqCst);
                        let current = b.lock().unwrap().clone();
                        let reply = match current {
                            Some(text) => format!(
                                "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                                text.len(),
                                text
                            ),
                            None => "HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_string(),
                        };
                        let _ = socket.write_all(reply.as_bytes()).await;
                    });
                }
            });
            Self { url, body, hits }
        }

        fn set(&self, running: usize, waiting: usize, kv: f64) {
            *self.body.lock().unwrap() = Some(format!(
                "# HELP vllm:num_requests_running x\nvllm:num_requests_running{{engine=\"0\",model_name=\"m\"}} {running}.0\n\
                 vllm:num_requests_waiting{{engine=\"0\",model_name=\"m\"}} {waiting}.0\n\
                 vllm:num_requests_waiting_by_reason{{engine=\"0\",reason=\"capacity\"}} 99.0\n\
                 vllm:kv_cache_usage_perc{{engine=\"0\",model_name=\"m\"}} {kv}\n"
            ));
        }

        fn fail(&self) {
            *self.body.lock().unwrap() = None;
        }
    }

    fn registry_with(urls: &[&str]) -> WorkerRegistry {
        let registry = WorkerRegistry::new();
        for url in urls {
            registry.register(Arc::new(BasicWorker::new(
                url.to_string(),
                WorkerType::Regular,
            )));
        }
        registry
    }

    fn provider() -> VllmMetricsObservationProvider {
        VllmMetricsObservationProvider::new(reqwest::Client::new(), None)
            .with_request_timeout(Duration::from_millis(500))
    }

    fn replica<'a>(summary: &'a Value, url: &str) -> &'a Value {
        let id = replica_id(url);
        summary["replicas"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["id"] == id)
            .unwrap()
    }

    #[tokio::test]
    async fn summary_reports_injected_engine_values_with_hysteresis() {
        let (a, b) = (FakeEngine::start().await, FakeEngine::start().await);
        a.set(3, 0, 0.2);
        b.set(5, 9, 0.5); // waiting 9 >= W_hard 8
        let registry = registry_with(&[&a.url, &b.url]);
        let state = LoadState::new(Config::default());
        let p = provider();

        state.sample(&registry, &p).await;
        let s = state.summary(&registry);
        assert_eq!(s["schema"], 1);
        assert_eq!(s["available"], 2);
        assert_eq!(s["fresh"], 2);
        assert_eq!(s["running_sum"], 8);
        assert_eq!(s["waiting_sum"], 9);
        assert_eq!(s["kv_min"], 0.2);
        assert_eq!(s["kv_max"], 0.5);
        assert_eq!(s["thresholds"]["waiting_hard"], 8);
        assert_eq!(replica(&s, &b.url)["saturated_now"], true);
        // One hot sample is not enough.
        assert_eq!(replica(&s, &b.url)["saturated"], false);
        assert_eq!(s["saturated"], 0);
        let id = s["replicas"][0]["id"].as_str().unwrap();
        assert!(
            id.len() == 8 && id.chars().all(|c| c.is_ascii_hexdigit()),
            "{id}"
        );
        assert!(
            !s.to_string().contains("127.0.0.1"),
            "replica ids must not leak addresses"
        );

        state.sample(&registry, &p).await;
        let s = state.summary(&registry);
        assert_eq!(replica(&s, &b.url)["saturated"], true);
        assert_eq!(replica(&s, &a.url)["saturated"], false);
        assert_eq!(s["saturated"], 1);
        assert_eq!(s["all_saturated"], false);
        assert!(state.is_saturated(&b.url) && !state.is_saturated(&a.url));

        // Two cold samples clear it again.
        b.set(5, 0, 0.5);
        state.sample(&registry, &p).await;
        assert!(state.is_saturated(&b.url), "one cold sample must not clear");
        state.sample(&registry, &p).await;
        assert!(!state.is_saturated(&b.url));
    }

    #[tokio::test]
    async fn all_saturated_only_when_every_available_replica_is() {
        let (a, b) = (FakeEngine::start().await, FakeEngine::start().await);
        a.set(1, 1, 0.9); // kv >= 0.85 with a queue
        b.set(1, 20, 0.1);
        let registry = registry_with(&[&a.url, &b.url]);
        let state = LoadState::new(Config::default());
        let p = provider();
        state.sample(&registry, &p).await;
        state.sample(&registry, &p).await;
        let s = state.summary(&registry);
        assert_eq!(s["saturated"], 2);
        assert_eq!(s["all_saturated"], true);
    }

    #[tokio::test]
    async fn failed_scrape_keeps_last_verdict_and_goes_stale() {
        let (a, b) = (FakeEngine::start().await, FakeEngine::start().await);
        a.set(1, 30, 0.1);
        b.fail(); // never sampled
        let registry = registry_with(&[&a.url, &b.url]);
        let state = LoadState::new(Config {
            interval_ms: 10,
            ..Config::default()
        });
        let p = provider();
        state.sample(&registry, &p).await;
        state.sample(&registry, &p).await;
        assert!(state.is_saturated(&a.url));
        // Never sampled: fail-open.
        assert!(!state.is_saturated(&b.url));
        a.fail();
        state.sample(&registry, &p).await;
        tokio::time::sleep(Duration::from_millis(60)).await;
        let s = state.summary(&registry);
        // Fail-closed: the last verdict holds, but the sample is no longer fresh.
        assert_eq!(replica(&s, &a.url)["saturated"], true);
        assert_eq!(replica(&s, &a.url)["fresh"], false);
        assert_eq!(replica(&s, &b.url)["fresh"], false);
        assert!(replica(&s, &b.url)["age_ms"].is_null());
        assert_eq!(s["fresh"], 0);
        assert_eq!(s["running_sum"], 0, "stale samples are not summed");
    }

    #[tokio::test]
    async fn sampler_follows_registry_changes() {
        let (a, b) = (FakeEngine::start().await, FakeEngine::start().await);
        a.set(1, 0, 0.1);
        b.set(2, 0, 0.1);
        let registry = registry_with(&[&a.url]);
        let state = LoadState::new(Config::default());
        let p = provider();
        state.sample(&registry, &p).await;
        assert_eq!(state.summary(&registry)["available"], 1);
        registry.register(Arc::new(BasicWorker::new(
            b.url.clone(),
            WorkerType::Regular,
        )));
        state.sample(&registry, &p).await;
        let s = state.summary(&registry);
        assert_eq!(s["available"], 2);
        assert_eq!(replica(&s, &b.url)["running"], 2);
        registry.remove_by_url(&a.url);
        state.sample(&registry, &p).await;
        let s = state.summary(&registry);
        assert_eq!(s["available"], 1);
        assert!(
            !state.replicas.lock().unwrap().contains_key(&a.url),
            "removed worker must be forgotten"
        );
        assert!(a.hits.load(Ordering::SeqCst) >= 1);
    }

    #[test]
    fn saturation_rule() {
        let cfg = Config::default();
        assert!(
            !cfg.saturated_now(0, Some(0.99), 0),
            "high KV with no queue still accepts work"
        );
        assert!(cfg.saturated_now(1, Some(0.85), 0));
        assert!(!cfg.saturated_now(7, Some(0.1), 0));
        assert!(cfg.saturated_now(8, Some(0.1), 0));
        assert!(
            !cfg.saturated_now(0, None, 500),
            "inflight rule needs a known max_num_seqs"
        );
        let cfg = Config {
            max_num_seqs: 64,
            ..Config::default()
        };
        assert_eq!(cfg.waiting_hard(), 16);
        assert!(cfg.saturated_now(0, Some(0.1), 64));
        assert!(!cfg.saturated_now(0, Some(0.1), 63));
        let cfg = Config {
            sat_waiting: 3,
            max_num_seqs: 64,
            ..Config::default()
        };
        assert_eq!(cfg.waiting_hard(), 3);
    }
}
