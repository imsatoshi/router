//! lingjing patch (0009): replica load as uncached in-flight tokens, with a time gate.
//!
//! Request counts treat a 200k-token cold prefill and a 20-token follow-up alike. This module
//! charges each dispatched request the tokens its replica still has to compute and releases the
//! charge at the first response chunk, when prefill is over. Dividing by the replica's calibrated
//! prefill throughput gives a wait; decisions compare waits, not counts:
//!
//! - new sessions rank by wait;
//! - a hit leaves its replica only when staying is slower than moving by more than
//!   `max_penalty_ms`, counting the whole body as cold on the other replica;
//! - a replica whose wait reaches `sat_wait_ms` counts as saturated for the 0008 size gate.
//!
//! `LINGJING_UNCACHED_MODE=off|shadow|on`. Off keeps no state and is 0008. Shadow charges and
//! counts `would_*` and routes as off. There are no built-in throughput or size figures: both are
//! model- and hardware-specific, so without `LINGJING_UNCACHED_PEAK_TPS` and
//! `LINGJING_UNCACHED_BYTES_PER_TOKEN` the feature stays off.
//!
//! Each router sees only the requests pinned to it, so peers publish their charges on
//! `GET /lingjing/uncached` and the wait sums local and fresh peer values. A silent peer drops out
//! after `PEER_FRESH_INTERVALS`; the wait is then a lower bound, counted as `peer_stale`.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};
use tracing::warn;

use crate::lingjing_warmup::Mode;

const PEER_FRESH_INTERVALS: u32 = 3;

#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub mode: Mode,
    pub peak_tps: f64,
    pub bytes_per_token: f64,
    pub max_penalty_ms: u64,
    pub sat_wait_ms: u64,
    pub max_hold_secs: u64,
    pub peer_interval_ms: u64,
}

impl Config {
    fn from_lookup(get: impl Fn(&str) -> Option<String>) -> Self {
        let float = |name: &str| {
            get(name).and_then(|raw| raw.trim().parse::<f64>().ok()).filter(|v| *v > 0.0).unwrap_or(0.0)
        };
        let number = |name: &str, default: u64| {
            get(name).and_then(|raw| raw.trim().parse().ok()).unwrap_or(default)
        };
        let mut cfg = Self {
            mode: Mode::parse("LINGJING_UNCACHED_MODE", get("LINGJING_UNCACHED_MODE").as_deref()),
            peak_tps: float("LINGJING_UNCACHED_PEAK_TPS"),
            bytes_per_token: float("LINGJING_UNCACHED_BYTES_PER_TOKEN"),
            max_penalty_ms: number("LINGJING_UNCACHED_MAX_PENALTY_MS", 0),
            sat_wait_ms: number("LINGJING_UNCACHED_SAT_WAIT_MS", 0),
            max_hold_secs: number("LINGJING_UNCACHED_MAX_HOLD_SECS", 1800),
            peer_interval_ms: number("LINGJING_UNCACHED_PEER_INTERVAL_MS", 1000),
        };
        if cfg.mode != Mode::Off && (cfg.peak_tps == 0.0 || cfg.bytes_per_token == 0.0) {
            warn!("lingjing uncached load needs calibrated PEAK_TPS and BYTES_PER_TOKEN; staying off");
            cfg.mode = Mode::Off;
        }
        cfg
    }

    pub fn from_env() -> Self {
        Self::from_lookup(|name| std::env::var(name).ok())
    }
}

#[derive(Debug, Clone, Copy)]
struct Charge {
    tokens: u64,
    deadline: Instant,
}

#[derive(Debug, Default)]
struct Inner {
    local: HashMap<String, HashMap<u64, Charge>>,
    peers: HashMap<String, (Instant, HashMap<String, u64>)>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct PeerSnapshot {
    pub schema: u32,
    pub tokens: HashMap<String, u64>,
}

#[derive(Debug)]
pub struct State {
    pub cfg: Config,
    next_id: AtomicU64,
    skew_ms: AtomicU64,
    inner: Mutex<Inner>,
}

/// One request's charge. Releases once, at the first chunk or when dropped on any other exit.
#[derive(Debug)]
pub struct Reservation {
    state: Arc<State>,
    url: String,
    id: u64,
    released: bool,
}

impl Reservation {
    pub fn release(&mut self) {
        if !self.released {
            self.released = true;
            self.state.release(&self.url, self.id);
        }
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        self.release();
    }
}

pub fn record(result: &'static str) {
    metrics::counter!("vllm_router_lingjing_uncached_total", "result" => result).increment(1);
}

impl State {
    pub fn new(cfg: Config) -> Self {
        Self {
            cfg,
            next_id: AtomicU64::new(0),
            skew_ms: AtomicU64::new(0),
            inner: Mutex::new(Inner::default()),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn now(&self) -> Instant {
        Instant::now() + Duration::from_millis(self.skew_ms.load(Ordering::Relaxed))
    }

    /// Move this state's clock forward (tests).
    pub fn advance(&self, by: Duration) {
        self.skew_ms.fetch_add(by.as_millis() as u64, Ordering::Relaxed);
    }

    /// Tokens for `bytes`; an unknown length counts as nothing here because the 0008 gate already
    /// treats it as large, and inventing a size would move the wait by an arbitrary amount.
    pub fn tokens_for(&self, bytes: Option<u64>) -> u64 {
        (bytes.unwrap_or(0) as f64 / self.cfg.bytes_per_token) as u64
    }

    /// What this request costs on `url`: the bytes added since the session's last turn when that
    /// turn ran there, otherwise the whole body. A body that shrank may have a new prefix, so it
    /// counts whole; a wrong guess ends at the first chunk.
    pub fn cost(&self, url: &str, body: Option<u64>, last: Option<(&str, Option<u64>)>) -> u64 {
        match (last, body) {
            (Some((worker, Some(before))), Some(now)) if worker == url && now >= before => {
                self.tokens_for(Some(now - before))
            }
            _ => self.tokens_for(body),
        }
    }

    pub fn charge(self: &Arc<Self>, url: &str, tokens: u64) -> Reservation {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let deadline = self.now() + Duration::from_secs(self.cfg.max_hold_secs);
        self.lock().local.entry(url.to_string()).or_default().insert(id, Charge { tokens, deadline });
        self.publish(url);
        Reservation { state: self.clone(), url: url.to_string(), id, released: false }
    }

    fn release(&self, url: &str, id: u64) {
        if let Some(charges) = self.lock().local.get_mut(url) {
            charges.remove(&id);
        }
        self.publish(url);
    }

    fn publish(&self, url: &str) {
        metrics::gauge!("vllm_router_lingjing_uncached_inflight_tokens", "worker" => url.to_string())
            .set(self.tokens(url) as f64);
    }

    fn local_tokens(inner: &mut Inner, url: &str, now: Instant) -> u64 {
        inner.local.get_mut(url).map_or(0, |charges| {
            charges.retain(|_, charge| charge.deadline > now);
            charges.values().map(|charge| charge.tokens).sum()
        })
    }

    /// Local charges plus fresh peer reports.
    pub fn tokens(&self, url: &str) -> u64 {
        let now = self.now();
        let fresh = Duration::from_millis(self.cfg.peer_interval_ms * u64::from(PEER_FRESH_INTERVALS));
        let mut inner = self.lock();
        let local = Self::local_tokens(&mut inner, url, now);
        let peers: u64 = inner
            .peers
            .values()
            .filter(|(at, _)| now.saturating_duration_since(*at) <= fresh)
            .filter_map(|(_, tokens)| tokens.get(url))
            .sum();
        local + peers
    }

    pub fn wait_ms(&self, url: &str, extra_tokens: u64) -> u64 {
        ((self.tokens(url) + extra_tokens) as f64 * 1000.0 / self.cfg.peak_tps) as u64
    }

    /// The time gate: waiting here beats moving unless staying is slower by more than the penalty.
    /// A penalty of 0 means never move.
    pub fn should_move(&self, stay_ms: u64, move_ms: u64) -> bool {
        self.cfg.max_penalty_ms > 0 && stay_ms > move_ms.saturating_add(self.cfg.max_penalty_ms)
    }

    pub fn is_saturated(&self, url: &str) -> bool {
        self.cfg.sat_wait_ms > 0 && self.wait_ms(url, 0) >= self.cfg.sat_wait_ms
    }

    pub fn snapshot(&self) -> PeerSnapshot {
        let now = self.now();
        let mut inner = self.lock();
        let urls: Vec<String> = inner.local.keys().cloned().collect();
        let tokens = urls
            .into_iter()
            .map(|url| {
                let held = Self::local_tokens(&mut inner, &url, now);
                (url, held)
            })
            .filter(|(_, held)| *held > 0)
            .collect();
        PeerSnapshot { schema: 1, tokens }
    }

    pub fn apply_peer(&self, peer: &str, snapshot: PeerSnapshot) {
        let now = self.now();
        self.lock().peers.insert(peer.to_string(), (now, snapshot.tokens));
    }
}

static STATE: OnceLock<Option<Arc<State>>> = OnceLock::new();

/// The process-wide state; `None` while off, so 0008 paths are untouched.
pub fn state() -> Option<&'static Arc<State>> {
    STATE
        .get_or_init(|| {
            let cfg = Config::from_env();
            (cfg.mode != Mode::Off).then(|| Arc::new(State::new(cfg)))
        })
        .as_ref()
}

/// Charge a request the policy just placed on `url`. The memo already holds this turn, so the
/// price comes from the turn before it. A retry on the same replica sees its own attempt as the
/// last turn and costs nothing; the first attempt's charge has been released by then.
pub fn charge_selected(
    text: &str,
    url: &str,
    headers: Option<&axum::http::HeaderMap>,
) -> Option<Reservation> {
    let state = state()?;
    let body = headers
        .and_then(|h| h.get(axum::http::header::CONTENT_LENGTH))
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.trim().parse().ok());
    let turn = crate::lingjing_warmup::state().and_then(|w| w.last_turn(text));
    let before = turn.as_ref().and_then(|t| t.before.as_ref());
    let tokens = state.cost(url, body, before.map(|(worker, bytes)| (worker.as_str(), *bytes)));
    Some(state.charge(url, tokens))
}

/// Pull peers' charges on a timer. Must run inside a tokio runtime. An older peer answers 404
/// and simply stays out of the sum.
pub fn start_from_env() {
    let Some(state) = state().cloned() else {
        return;
    };
    let Some(host) = std::env::var("LINGJING_ROUTER_PEERS").ok().filter(|h| !h.trim().is_empty()) else {
        warn!("lingjing uncached load without LINGJING_ROUTER_PEERS: counting this router only");
        return;
    };
    let port: u16 = std::env::var("LINGJING_ROUTER_PEER_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8000);
    let self_ip = std::env::var("LINGJING_ROUTER_SELF_IP").unwrap_or_default();
    let interval = Duration::from_millis(state.cfg.peer_interval_ms.max(100));
    tokio::spawn(async move {
        let client = reqwest::Client::builder().timeout(interval).build().unwrap_or_default();
        loop {
            tokio::time::sleep(interval).await;
            for peer in crate::lingjing_affinity::resolve_peers(host.trim(), port, &self_ip).await {
                let url = format!("http://{peer}/lingjing/uncached");
                match client.get(url).send().await {
                    Ok(reply) if reply.status().is_success() => match reply.json().await {
                        Ok(snapshot) => state.apply_peer(&peer, snapshot),
                        Err(_) => record("peer_stale"),
                    },
                    _ => record("peer_stale"),
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(mode: Mode) -> Config {
        Config {
            mode,
            peak_tps: 1000.0,
            bytes_per_token: 10.0,
            max_penalty_ms: 2000,
            sat_wait_ms: 5000,
            max_hold_secs: 60,
            peer_interval_ms: 1000,
        }
    }

    fn state(mode: Mode) -> Arc<State> {
        Arc::new(State::new(cfg(mode)))
    }

    #[test]
    fn uncached_stays_off_without_calibration() {
        let on = |name: &str| (name == "LINGJING_UNCACHED_MODE").then(|| "1".to_string());
        assert_eq!(Config::from_lookup(on).mode, Mode::Off);
        let full = |name: &str| match name {
            "LINGJING_UNCACHED_MODE" => Some("shadow".to_string()),
            "LINGJING_UNCACHED_PEAK_TPS" => Some("1234".to_string()),
            "LINGJING_UNCACHED_BYTES_PER_TOKEN" => Some("110".to_string()),
            _ => None,
        };
        let parsed = Config::from_lookup(full);
        assert_eq!((parsed.mode, parsed.peak_tps, parsed.bytes_per_token), (Mode::Shadow, 1234.0, 110.0));
        assert_eq!(parsed.max_penalty_ms, 0, "no borrowed gate value");
    }

    #[test]
    fn uncached_cost_is_the_delta_only_on_the_same_worker() {
        let s = state(Mode::On);
        assert_eq!(s.cost("a", Some(1500), Some(("a", Some(1000)))), 50);
        assert_eq!(s.cost("b", Some(1500), Some(("a", Some(1000)))), 150, "cold elsewhere");
        assert_eq!(s.cost("a", Some(900), Some(("a", Some(1000)))), 90, "shrank: prefix unknown");
        assert_eq!(s.cost("a", Some(1500), None), 150, "first turn");
    }

    #[test]
    fn uncached_charge_releases_once_and_on_drop() {
        let s = state(Mode::On);
        let mut first = s.charge("a", 100);
        let second = s.charge("a", 50);
        assert_eq!(s.tokens("a"), 150);
        first.release();
        first.release();
        assert_eq!(s.tokens("a"), 50);
        drop(second);
        assert_eq!(s.tokens("a"), 0);
    }

    #[test]
    fn uncached_max_hold_expires_a_charge() {
        let s = state(Mode::On);
        let _held = s.charge("a", 100);
        s.advance(Duration::from_secs(59));
        assert_eq!(s.tokens("a"), 100);
        s.advance(Duration::from_secs(2));
        assert_eq!(s.tokens("a"), 0);
    }

    #[test]
    fn uncached_time_gate_needs_more_than_the_penalty() {
        let s = state(Mode::On);
        assert!(!s.should_move(3000, 1000));
        assert!(s.should_move(3001, 1000));
        let never = Arc::new(State::new(Config { max_penalty_ms: 0, ..cfg(Mode::On) }));
        assert!(!never.should_move(u64::MAX, 0));
    }

    #[test]
    fn uncached_saturation_is_a_wait() {
        let s = state(Mode::On);
        let _held = s.charge("a", 4999);
        assert!(!s.is_saturated("a"));
        let _more = s.charge("a", 1);
        assert!(s.is_saturated("a"));
    }

    #[test]
    fn uncached_peers_add_up_until_they_go_quiet() {
        let s = state(Mode::On);
        let _held = s.charge("a", 100);
        let tokens = [("a".to_string(), 40)].into();
        s.apply_peer("p", PeerSnapshot { schema: 1, tokens });
        assert_eq!(s.tokens("a"), 140);
        s.advance(Duration::from_millis(3001));
        assert_eq!(s.tokens("a"), 100, "a stale peer drops out");
    }

    #[test]
    fn uncached_snapshot_carries_local_charges_only() {
        let s = state(Mode::On);
        let _held = s.charge("a", 100);
        s.apply_peer("p", PeerSnapshot { schema: 1, tokens: [("a".to_string(), 40)].into() });
        assert_eq!(s.snapshot().tokens["a"], 100, "peers must not echo each other's load");
    }
}
