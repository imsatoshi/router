//! lingjing patch (0008): keep long sessions where their KV cache is across scale-out,
//! imbalance and health eviction. Design: docs/platform/2026-11-05-inf-385-session-affinity-design.md
//! in the ai-platform repository. Three independent features, all off by default:
//!
//! 1. Slow start (`LINGJING_WARMUP_MODE=off|shadow|on`). A worker that pushes the replica count
//!    above the high-water mark of the last `LINGJING_WARMUP_SECS` (3600) is a scale-out, not a
//!    rolling restart (2→1→2) and not a restored eviction. For that window new sessions of at
//!    least `LINGJING_WARMUP_MAX_BODY_BYTES` (262144) stay off it, unless every other replica
//!    is saturated. A request without a usable `content-length` counts as large. The worker
//!    also gets a tree-size credit so that an empty tree does not win every tie; the credit
//!    outlives the window. Workers seen in the first `LINGJING_WARMUP_STARTUP_GRACE_SECS` (120)
//!    of this process are the existing fleet.
//! 2. Hits stay put (`LINGJING_KEEP_HIT_ON_IMBALANCE=0|shadow|1`). When the fleet is imbalanced,
//!    a request that matches a usable, unsaturated worker takes the hit path.
//! 3. Eviction keeps affinity (`LINGJING_EVICT_KEEP_AFFINITY=0|shadow|1`).
//!    `POST /remove_worker?url=&keep_affinity=1` leaves the worker's tenants in the tree. A hit
//!    on an unavailable tenant uses another usable tenant of the same node, then the new-session
//!    path, and never deletes the tenant. Leftovers are dropped after
//!    `LINGJING_KEEP_AFFINITY_TTL_SECS` (600) unless the worker is registered again.
//!
//! `shadow` records the decision (metrics, one log line) and routes as `off`. Logs carry the
//! worker URL, the result and the body size, never request content or routing text.

use crate::core::{Worker, WorkerRegistry};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, RwLock};
use std::time::{Duration, Instant};
use tracing::{info, warn};

const PURGE_INTERVAL: Duration = Duration::from_secs(5);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Off,
    Shadow,
    On,
}

impl Mode {
    /// An unreadable value means off: a typo must not change routing.
    fn parse(name: &str, raw: Option<&str>) -> Self {
        match raw.map(|v| v.trim().to_ascii_lowercase()).as_deref() {
            None | Some("") | Some("0") | Some("off") | Some("false") => Self::Off,
            Some("1") | Some("on") | Some("true") => Self::On,
            Some("shadow") => Self::Shadow,
            Some(other) => {
                warn!("{name}={other} not understood; treating it as off");
                Self::Off
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub warmup_mode: Mode,
    pub warmup_secs: u64,
    pub max_body_bytes: u64,
    pub startup_grace_secs: u64,
    pub keep_hit: Mode,
    pub keep_affinity: Mode,
    pub keep_affinity_ttl_secs: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            warmup_mode: Mode::Off,
            warmup_secs: 3600,
            max_body_bytes: 262_144,
            startup_grace_secs: 120,
            keep_hit: Mode::Off,
            keep_affinity: Mode::Off,
            keep_affinity_ttl_secs: 600,
        }
    }
}

impl Config {
    fn from_lookup(get: impl Fn(&str) -> Option<String>) -> Self {
        let number = |name: &str, default: u64| match get(name) {
            None => default,
            Some(raw) => raw.trim().parse().unwrap_or_else(|_| {
                warn!("{name}={raw} not a number; using {default}");
                default
            }),
        };
        let defaults = Self::default();
        let mode = |name: &str| Mode::parse(name, get(name).as_deref());
        Self {
            warmup_mode: mode("LINGJING_WARMUP_MODE"),
            warmup_secs: number("LINGJING_WARMUP_SECS", defaults.warmup_secs),
            max_body_bytes: number("LINGJING_WARMUP_MAX_BODY_BYTES", defaults.max_body_bytes),
            startup_grace_secs: number(
                "LINGJING_WARMUP_STARTUP_GRACE_SECS",
                defaults.startup_grace_secs,
            ),
            keep_hit: mode("LINGJING_KEEP_HIT_ON_IMBALANCE"),
            keep_affinity: mode("LINGJING_EVICT_KEEP_AFFINITY"),
            keep_affinity_ttl_secs: number(
                "LINGJING_KEEP_AFFINITY_TTL_SECS",
                defaults.keep_affinity_ttl_secs,
            ),
        }
    }

    pub fn from_env() -> Self {
        Self::from_lookup(|name| std::env::var(name).ok())
    }

    fn any_enabled(&self) -> bool {
        self.warmup_mode != Mode::Off
            || self.keep_hit != Mode::Off
            || self.keep_affinity != Mode::Off
    }
}

/// What selection knows about one request.
#[derive(Debug, Clone, Copy, Default)]
pub struct Placement {
    /// `content-length`; `None` is treated as a large request.
    pub body_bytes: Option<u64>,
    /// A hit moved off a saturated worker: stays off warming workers whatever its size.
    pub moved_hit: bool,
}

impl Placement {
    /// Counts `no_length` for every request, warming or not: the stage-0 check that the gateway's
    /// `content-length` really reaches this path must not depend on a scale-out happening.
    pub fn from_headers(headers: Option<&crate::policies::RequestHeaders>, count: bool) -> Self {
        let body_bytes = headers
            .and_then(|h| h.get("content-length"))
            .and_then(|v| v.trim().parse().ok());
        if count && body_bytes.is_none() {
            metrics::counter!("vllm_router_lingjing_warmup_total", "result" => "no_length")
                .increment(1);
        }
        Self { body_bytes, moved_hit: false }
    }
}

#[derive(Debug, Clone)]
struct Warm {
    until: Instant,
    credit: usize,
}

#[derive(Debug, Default)]
struct Inner {
    warm: HashMap<String, Warm>,
    /// model → (when the count changed, the count it left behind).
    history: HashMap<String, VecDeque<(Instant, usize)>>,
    /// Tenants kept for a worker that is not available, and since when.
    retained: HashMap<String, Instant>,
}

/// One entry of `GET /lingjing/warmup`; `remaining_secs == 0` is a credit that outlived its window.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PeerEntry {
    pub url: String,
    pub remaining_secs: u64,
    pub credit: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PeerSnapshot {
    pub schema: u32,
    pub entries: Vec<PeerEntry>,
}

#[derive(Debug)]
pub struct State {
    pub cfg: Config,
    started: Instant,
    skew_ms: AtomicU64,
    /// Selection takes no lock until the first scale-out.
    has_warm: AtomicBool,
    inner: RwLock<Inner>,
}

fn decision(kind: &'static str, result: &'static str, worker: &str, body_bytes: Option<u64>) {
    // Default target on purpose: the router's log filter only passes its own crate.
    info!(
        kind,
        result,
        worker,
        body_bytes = body_bytes.map(|n| n as i64).unwrap_or(-1),
    );
}

pub fn record_warmup(result: &'static str, worker: &str, body_bytes: Option<u64>) {
    metrics::counter!("vllm_router_lingjing_warmup_total", "result" => result).increment(1);
    decision("warmup", result, worker, body_bytes);
}

pub fn record_imbalance(action: &'static str, worker: &str) {
    metrics::counter!("vllm_router_lingjing_imbalance_total", "action" => action).increment(1);
    decision("imbalance", action, worker, None);
}

pub fn record_keep_affinity(action: &'static str, worker: &str) {
    metrics::counter!("vllm_router_lingjing_keep_affinity_total", "action" => action).increment(1);
    decision("keep_affinity", action, worker, None);
}

impl State {
    pub fn new(cfg: Config) -> Self {
        Self {
            cfg,
            started: Instant::now(),
            skew_ms: AtomicU64::new(0),
            has_warm: AtomicBool::new(false),
            inner: RwLock::new(Inner::default()),
        }
    }

    fn read(&self) -> std::sync::RwLockReadGuard<'_, Inner> {
        self.inner.read().unwrap_or_else(|e| e.into_inner())
    }

    fn write(&self) -> std::sync::RwLockWriteGuard<'_, Inner> {
        self.inner.write().unwrap_or_else(|e| e.into_inner())
    }

    fn now(&self) -> Instant {
        Instant::now() + Duration::from_millis(self.skew_ms.load(Ordering::Relaxed))
    }

    /// Move this state's clock forward (tests).
    pub fn advance(&self, by: Duration) {
        self.skew_ms.fetch_add(by.as_millis() as u64, Ordering::Relaxed);
    }

    fn publish_gauge(&self, inner: &Inner) {
        let now = self.now();
        let warming = inner.warm.values().filter(|w| w.until > now).count();
        metrics::gauge!("vllm_router_lingjing_warming_workers").set(warming as f64);
    }

    /// Call before the worker becomes routable. `current` = registered workers of the model.
    /// Returns true when this worker just started warming.
    pub fn on_worker_adding(&self, model: &str, url: &str, current: usize) -> bool {
        let now = self.now();
        let mut inner = self.write();
        if inner.retained.remove(url).is_some() {
            record_keep_affinity("restored", url);
        }
        if self.cfg.warmup_mode == Mode::Off {
            return false;
        }
        let window = Duration::from_secs(self.cfg.warmup_secs);
        let history = inner.history.entry(model.to_string()).or_default();
        while history
            .front()
            .is_some_and(|(at, _)| now.saturating_duration_since(*at) > window)
        {
            history.pop_front();
        }
        let high_water = history.iter().map(|(_, n)| *n).fold(current, usize::max);
        history.push_back((now, current));
        let in_grace = now.saturating_duration_since(self.started)
            < Duration::from_secs(self.cfg.startup_grace_secs);
        if current + 1 <= high_water || in_grace || inner.warm.contains_key(url) {
            return false;
        }
        inner.warm.insert(url.to_string(), Warm { until: now + window, credit: 0 });
        self.has_warm.store(true, Ordering::Relaxed);
        self.publish_gauge(&inner);
        info!(
            kind = "warmup",
            result = "started",
            worker = url,
            high_water,
            window_secs = self.cfg.warmup_secs,
        );
        true
    }

    /// `kept`: the worker may come back (eviction with kept affinity), so it stays warming.
    pub fn on_worker_removed(&self, model: &str, url: &str, current: usize, kept: bool) {
        let now = self.now();
        let mut inner = self.write();
        if self.cfg.warmup_mode != Mode::Off {
            inner.history.entry(model.to_string()).or_default().push_back((now, current));
        }
        if !kept && inner.warm.remove(url).is_some() {
            self.has_warm.store(!inner.warm.is_empty(), Ordering::Relaxed);
            self.publish_gauge(&inner);
        }
    }

    /// The worker is gone for good and was not in the table any more.
    pub fn forget(&self, url: &str) {
        let mut inner = self.write();
        if inner.warm.remove(url).is_some() {
            self.has_warm.store(!inner.warm.is_empty(), Ordering::Relaxed);
            self.publish_gauge(&inner);
        }
    }

    pub fn set_credit(&self, url: &str, credit: usize) {
        if let Some(warm) = self.write().warm.get_mut(url) {
            warm.credit = credit;
        }
    }

    pub fn is_warming(&self, url: &str) -> bool {
        self.has_warm.load(Ordering::Relaxed)
            && self.read().warm.get(url).is_some_and(|w| w.until > self.now())
    }

    pub fn has_warming(&self, workers: &[Arc<dyn Worker>], candidates: &[usize]) -> bool {
        self.has_warm.load(Ordering::Relaxed)
            && candidates.iter().any(|&idx| self.is_warming(workers[idx].url()))
    }

    /// Tree-size credit, whatever the mode; callers decide whether it changes routing.
    pub fn credit(&self, url: &str) -> usize {
        if !self.has_warm.load(Ordering::Relaxed) {
            return 0;
        }
        self.read().warm.get(url).map_or(0, |w| w.credit)
    }

    /// Keep this worker's tenants until the TTL; a first sighting starts the clock.
    pub fn retain(&self, url: &str) {
        let now = self.now();
        self.write().retained.entry(url.to_string()).or_insert(now);
    }

    pub fn take_retained(&self, url: &str) -> bool {
        self.write().retained.remove(url).is_some()
    }

    pub fn is_retained(&self, url: &str) -> bool {
        self.read().retained.contains_key(url)
    }

    /// Past the TTL: drop tenants of workers that never came back. A worker still registered
    /// (open breaker, unhealthy) is not gone; its tenants stay.
    pub fn purge_retained(&self, is_registered: impl Fn(&str) -> bool, remove: impl Fn(&str)) {
        let now = self.now();
        let ttl = Duration::from_secs(self.cfg.keep_affinity_ttl_secs);
        let expired: Vec<String> = {
            let mut inner = self.write();
            let expired = inner
                .retained
                .iter()
                .filter(|(_, since)| now.saturating_duration_since(**since) >= ttl)
                .map(|(url, _)| url.clone())
                .collect::<Vec<_>>();
            for url in &expired {
                inner.retained.remove(url);
            }
            expired
        };
        for url in expired {
            if !is_registered(&url) {
                remove(&url);
                self.forget(&url);
                record_keep_affinity("expired", &url);
            }
        }
    }

    pub fn snapshot(&self) -> PeerSnapshot {
        let now = self.now();
        let mut entries: Vec<PeerEntry> = self
            .read()
            .warm
            .iter()
            .map(|(url, warm)| PeerEntry {
                url: url.clone(),
                remaining_secs: warm.until.saturating_duration_since(now).as_secs(),
                credit: warm.credit,
            })
            .collect();
        entries.sort_by(|a, b| a.url.cmp(&b.url));
        PeerSnapshot { schema: 1, entries }
    }

    /// Adopt a peer's warming workers (router restarted inside a window). Local state wins.
    pub fn apply_peer(&self, snapshot: &PeerSnapshot) -> usize {
        let now = self.now();
        let mut inner = self.write();
        let mut applied = 0;
        for entry in &snapshot.entries {
            if !inner.warm.contains_key(&entry.url) {
                let until = now + Duration::from_secs(entry.remaining_secs);
                inner.warm.insert(entry.url.clone(), Warm { until, credit: entry.credit });
                applied += 1;
            }
        }
        if !inner.warm.is_empty() {
            self.has_warm.store(true, Ordering::Relaxed);
        }
        self.publish_gauge(&inner);
        applied
    }

    /// New-session placement: `ranked` is the policy's preference order, `pick` the 0006
    /// saturation-aware choice from a candidate list.
    pub fn place(
        &self,
        workers: &[Arc<dyn Worker>],
        ranked: &[usize],
        placement: Placement,
        is_saturated: impl Fn(&str) -> bool,
        pick: impl Fn(&[usize]) -> Option<usize>,
    ) -> Option<usize> {
        if self.cfg.warmup_mode == Mode::Off || !self.has_warming(workers, ranked) {
            return pick(ranked);
        }
        let warming = |idx: usize| self.is_warming(workers[idx].url());
        let body = placement.body_bytes;
        let large = body.is_none_or(|bytes| bytes >= self.cfg.max_body_bytes);
        if !large && !placement.moved_hit {
            let chosen = pick(ranked)?;
            if warming(chosen) {
                record_warmup("allowed_short", workers[chosen].url(), body);
            }
            return Some(chosen);
        }
        let others: Vec<usize> = ranked.iter().copied().filter(|&idx| !warming(idx)).collect();
        if others.iter().all(|&idx| is_saturated(workers[idx].url())) {
            let chosen = pick(ranked)?;
            if warming(chosen) {
                record_warmup("bypass_saturated", workers[chosen].url(), body);
            }
            return Some(chosen);
        }
        match self.cfg.warmup_mode {
            Mode::On => {
                if warming(ranked[0]) {
                    record_warmup("blocked_long", workers[ranked[0]].url(), body);
                }
                pick(&others)
            }
            _ => {
                let chosen = pick(ranked)?;
                if warming(chosen) {
                    record_warmup("would_block_long", workers[chosen].url(), body);
                }
                Some(chosen)
            }
        }
    }
}

static STATE: OnceLock<Option<Arc<State>>> = OnceLock::new();

/// The process-wide state; `None` while every feature is off, so 0007 paths are untouched.
pub fn state() -> Option<&'static Arc<State>> {
    STATE
        .get_or_init(|| {
            let cfg = Config::from_env();
            cfg.any_enabled().then(|| {
                info!(?cfg, "lingjing session affinity features enabled");
                Arc::new(State::new(cfg))
            })
        })
        .as_ref()
}

/// Drop leftover tenants on a timer. Must run inside a tokio runtime.
pub fn start_from_env(registry: Arc<WorkerRegistry>) {
    let Some(state) = state().cloned() else {
        return;
    };
    if state.cfg.warmup_mode != Mode::Off && crate::lingjing_load::state().is_none() {
        warn!("lingjing warmup without the load sampler: saturation cannot release the size gate");
    }
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(PURGE_INTERVAL);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            interval.tick().await;
            state.purge_retained(
                |url| registry.get_by_url(url).is_some(),
                crate::lingjing_affinity::remove_tenant,
            );
            state.publish_gauge(&state.read());
        }
    });
}

/// Startup: adopt a peer's warming workers. An older peer answers 404, which is fine.
pub async fn pull_from_peer(client: &reqwest::Client, peer: &str) {
    let Some(state) = state() else {
        return;
    };
    match client.get(format!("http://{peer}/lingjing/warmup")).send().await {
        Ok(response) if response.status().is_success() => {
            match response.json::<PeerSnapshot>().await {
                Ok(snapshot) => {
                    let applied = state.apply_peer(&snapshot);
                    info!("lingjing warmup: {applied} workers adopted from {peer}");
                }
                Err(e) => warn!("lingjing warmup bootstrap: decode from {peer}: {e}"),
            }
        }
        Ok(response) => info!("lingjing warmup bootstrap: {peer} answered {}", response.status()),
        Err(e) => warn!("lingjing warmup bootstrap: {peer}: {e}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(cfg: Config) -> State {
        State::new(Config { warmup_mode: Mode::On, startup_grace_secs: 0, ..cfg })
    }

    #[test]
    fn lingjing_warmup_modes_parse_conservatively() {
        for (raw, want) in [
            (None, Mode::Off),
            (Some(""), Mode::Off),
            (Some("0"), Mode::Off),
            (Some("off"), Mode::Off),
            (Some("yes please"), Mode::Off),
            (Some("1"), Mode::On),
            (Some("ON"), Mode::On),
            (Some(" shadow "), Mode::Shadow),
        ] {
            assert_eq!(Mode::parse("X", raw), want, "{raw:?}");
        }
    }

    #[test]
    fn lingjing_warmup_defaults_are_all_off_with_the_decided_window() {
        let cfg = Config::from_lookup(|_| None);
        assert!(!cfg.any_enabled());
        assert_eq!((cfg.warmup_secs, cfg.max_body_bytes), (3600, 262_144));
        assert_eq!(cfg.keep_affinity_ttl_secs, 600);
        let cfg = Config::from_lookup(|name| match name {
            "LINGJING_WARMUP_MODE" => Some("shadow".into()),
            "LINGJING_WARMUP_SECS" => Some("bad".into()),
            "LINGJING_WARMUP_MAX_BODY_BYTES" => Some("1024".into()),
            _ => None,
        });
        assert_eq!(cfg.warmup_mode, Mode::Shadow);
        assert_eq!((cfg.warmup_secs, cfg.max_body_bytes), (3600, 1024));
    }

    #[test]
    fn lingjing_warmup_only_a_count_above_the_high_water_mark_warms() {
        let s = state(Config::default());
        assert!(s.on_worker_adding("m", "a", 0), "the very first count is above nothing");
        let s = state(Config::default());
        s.on_worker_removed("m", "x", 2, false); // rolling restart: 2 → 1
        assert!(!s.on_worker_adding("m", "new-pod", 1), "1 → 2 stays at the mark");
        assert!(s.on_worker_adding("m", "third", 2), "2 → 3 is a scale-out");
        assert!(s.is_warming("third"));
    }

    #[test]
    fn lingjing_warmup_high_water_mark_ages_out_of_the_window() {
        let s = state(Config { warmup_secs: 100, ..Config::default() });
        s.on_worker_removed("m", "w3", 3, false); // 3 → 2
        s.advance(Duration::from_secs(99));
        assert!(!s.on_worker_adding("m", "w4", 2), "still inside the window");
        s.on_worker_removed("m", "w4", 3, false);
        s.advance(Duration::from_secs(101));
        assert!(s.on_worker_adding("m", "w5", 2), "the old mark has aged out");
    }

    #[test]
    fn lingjing_warmup_startup_grace_means_existing_fleet() {
        let s = State::new(Config { warmup_mode: Mode::On, ..Config::default() });
        assert!(!s.on_worker_adding("m", "a", 0));
        assert!(!s.on_worker_adding("m", "b", 1));
        s.advance(Duration::from_secs(121));
        assert!(s.on_worker_adding("m", "c", 2));
    }

    #[test]
    fn lingjing_warmup_window_ends_but_credit_stays() {
        let s = state(Config { warmup_secs: 10, ..Config::default() });
        s.on_worker_adding("m", "w", 2);
        s.set_credit("w", 500);
        s.advance(Duration::from_secs(11));
        assert!(!s.is_warming("w"));
        assert_eq!(s.credit("w"), 500);
        s.on_worker_removed("m", "w", 3, false);
        assert_eq!(s.credit("w"), 0, "a real removal clears the credit");
    }

    #[test]
    fn lingjing_warmup_kept_eviction_keeps_warming_and_restore_is_not_a_scale_out() {
        let s = state(Config::default());
        s.on_worker_adding("m", "w", 2);
        s.on_worker_removed("m", "w", 3, true);
        assert!(s.is_warming("w"));
        assert!(!s.on_worker_adding("m", "w", 2));
        assert!(s.is_warming("w"));
    }

    #[test]
    fn lingjing_warmup_ttl_clears_only_workers_that_stayed_away() {
        let s = state(Config { keep_affinity_ttl_secs: 600, ..Config::default() });
        s.retain("gone");
        s.retain("still-registered");
        let removed = std::sync::Mutex::new(Vec::new());
        let sweep = || {
            s.purge_retained(
                |url| url == "still-registered",
                |url| removed.lock().unwrap().push(url.to_string()),
            )
        };
        s.advance(Duration::from_secs(599));
        sweep();
        assert!(removed.lock().unwrap().is_empty());
        s.advance(Duration::from_secs(1));
        sweep();
        assert_eq!(*removed.lock().unwrap(), vec!["gone".to_string()]);
        assert!(!s.is_retained("gone") && !s.is_retained("still-registered"));
    }

    #[test]
    fn lingjing_warmup_retain_does_not_restart_the_clock() {
        let s = state(Config { keep_affinity_ttl_secs: 10, ..Config::default() });
        s.retain("w");
        s.advance(Duration::from_secs(9));
        s.retain("w");
        s.advance(Duration::from_secs(1));
        let removed = AtomicBool::new(false);
        s.purge_retained(|_| false, |_| removed.store(true, Ordering::Relaxed));
        assert!(removed.load(Ordering::Relaxed));
    }

    #[test]
    fn lingjing_warmup_peer_snapshot_carries_window_and_credit_and_local_state_wins() {
        let a = state(Config::default());
        a.on_worker_adding("m", "w", 2);
        a.set_credit("w", 7);
        a.advance(Duration::from_secs(600));
        let wire = serde_json::to_string(&a.snapshot()).unwrap();
        let b = state(Config::default());
        assert_eq!(b.apply_peer(&serde_json::from_str(&wire).unwrap()), 1);
        assert!(b.is_warming("w"));
        assert_eq!(b.credit("w"), 7);
        assert_eq!(b.apply_peer(&serde_json::from_str(&wire).unwrap()), 0);
        b.advance(Duration::from_secs(3000));
        assert!(!b.is_warming("w"), "the peer's remaining time was adopted, not a full window");
    }
}
