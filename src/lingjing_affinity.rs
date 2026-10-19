//! lingjing patch: share short session→worker affinities between router replicas.
//!
//! Two routers in front of the same replicas each keep their own prefix tree. The
//! gateway pins a session to one router (HRW over the headless service), so trees
//! stay complete in steady state; this module covers the failover moment: every
//! placement of a short routing text (the 16-char session hash the gateway injects)
//! is broadcast to the peer routers, which insert it into their own tree. A session
//! that moves to the surviving router then still lands on the replica that holds its
//! KV cache. Long prefix texts are not sent (cost); they rely on gateway pinning.
//!
//! A router that starts (rollout, replacement) first pulls a snapshot of the peers'
//! affinities and stays not-ready until it has applied it (or 60s passed), so the
//! headless service never hands it sessions it knows nothing about.
//!
//! Enabled by LINGJING_ROUTER_PEERS=<headless host>; LINGJING_ROUTER_SELF_IP excludes
//! this pod. Received entries are applied only for workers this router already knows
//! and are never re-broadcast.

use crate::tree::Tree;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TrySendError};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

/// Session hashes are 16 hex chars; anything longer is a prefix text and is not synced.
pub const MAX_SYNC_TEXT_BYTES: usize = 64;
const QUEUE_CAPACITY: usize = 20_000;
const FLUSH_INTERVAL: Duration = Duration::from_millis(100);
const PEER_REFRESH: Duration = Duration::from_secs(5);
const SEND_TIMEOUT: Duration = Duration::from_secs(1);
/// Affinities remembered for snapshots (most recent kept). Sessions are thousands, not millions.
const JOURNAL_CAPACITY: usize = 100_000;
const BOOTSTRAP_DEADLINE: Duration = Duration::from_secs(60);

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Affinity {
    pub model: String,
    pub text: String,
    pub url: String,
}

type Trees = Arc<DashMap<String, Arc<Tree>>>;

static TREES: OnceLock<Mutex<Vec<Trees>>> = OnceLock::new();
static SENDER: OnceLock<SyncSender<Affinity>> = OnceLock::new();
/// false only while a router with peers is pulling the startup snapshot.
static BOOTSTRAPPED: AtomicBool = AtomicBool::new(true);

#[derive(Default)]
struct Journal {
    latest: HashMap<(String, String), String>,
    order: VecDeque<(String, String)>,
}

static JOURNAL: OnceLock<Mutex<Journal>> = OnceLock::new();

fn record(model: &str, text: &str, url: &str) {
    let mut j = JOURNAL
        .get_or_init(|| Mutex::new(Journal::default()))
        .lock()
        .unwrap();
    let key = (model.to_string(), text.to_string());
    if j.latest.insert(key.clone(), url.to_string()).is_none() {
        j.order.push_back(key);
        while j.order.len() > JOURNAL_CAPACITY {
            if let Some(old) = j.order.pop_front() {
                j.latest.remove(&old);
            }
        }
    }
}

/// Every remembered affinity, for a peer that just started.
pub fn snapshot() -> Vec<Affinity> {
    let Some(j) = JOURNAL.get() else {
        return Vec::new();
    };
    let j = j.lock().unwrap();
    j.latest
        .iter()
        .map(|((model, text), url)| Affinity {
            model: model.clone(),
            text: text.clone(),
            url: url.clone(),
        })
        .collect()
}

/// Readiness gate: false while the startup snapshot is being pulled.
pub fn bootstrap_ready() -> bool {
    BOOTSTRAPPED.load(Ordering::Acquire)
}

/// Called by every CacheAwarePolicy so received affinities reach its trees.
pub fn register_trees(trees: Trees) {
    TREES
        .get_or_init(|| Mutex::new(Vec::new()))
        .lock()
        .unwrap()
        .push(trees);
}

/// Queue one placement for broadcast. Never blocks the routing path; drops when full.
pub fn publish(model: &str, text: &str, url: &str) {
    if text.is_empty() || text.len() > MAX_SYNC_TEXT_BYTES {
        return;
    }
    if let Some(tx) = SENDER.get() {
        record(model, text, url);
        match tx.try_send(Affinity {
            model: model.to_string(),
            text: text.to_string(),
            url: url.to_string(),
        }) {
            Ok(()) => {
                metrics::counter!("vllm_router_lingjing_affinity_total", "result" => "queued")
                    .increment(1)
            }
            Err(TrySendError::Full(_)) => {
                metrics::counter!("vllm_router_lingjing_affinity_total", "result" => "dropped")
                    .increment(1)
            }
            Err(TrySendError::Disconnected(_)) => {}
        }
    }
}

/// Apply affinities received from a peer (or a startup snapshot). The model's tree is
/// created if this router has not routed that model yet (trees are created lazily on the
/// first request, so a fresh router has none). The worker does not have to be known
/// yet: if it is gone by the time a session returns, cache_aware drops the stale tenant
/// and falls back to a healthy worker (existing behaviour). Entries are inserted
/// directly and not re-published, so there is no echo.
pub fn apply(items: &[Affinity]) -> usize {
    let Some(all) = TREES.get() else { return 0 };
    let all = all.lock().unwrap().clone();
    let Some(first) = all.first() else { return 0 };
    let mut applied = 0;
    let mut rejected = 0;
    for it in items {
        if it.text.is_empty()
            || it.text.len() > MAX_SYNC_TEXT_BYTES
            || it.url.is_empty()
            || it.model.is_empty()
        {
            rejected += 1;
            continue;
        }
        let mut inserted = false;
        for trees in &all {
            if let Some(tree) = trees.get(&it.model) {
                tree.insert(&it.text, &it.url);
                inserted = true;
            }
        }
        if !inserted {
            first
                .entry(it.model.clone())
                .or_insert_with(|| Arc::new(Tree::new()))
                .insert(&it.text, &it.url);
        }
        record(&it.model, &it.text, &it.url);
        applied += 1;
    }
    metrics::counter!("vllm_router_lingjing_affinity_total", "result" => "applied")
        .increment(applied as u64);
    metrics::counter!("vllm_router_lingjing_affinity_total", "result" => "rejected")
        .increment(rejected as u64);
    applied
}

/// Start the broadcaster when LINGJING_ROUTER_PEERS is set. Must run inside a tokio runtime.
pub fn start_from_env() {
    let Ok(peers_host) = std::env::var("LINGJING_ROUTER_PEERS") else {
        return;
    };
    let peers_host = peers_host.trim().to_string();
    if peers_host.is_empty() {
        return;
    }
    let port: u16 = std::env::var("LINGJING_ROUTER_PEER_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8000);
    let self_ip = std::env::var("LINGJING_ROUTER_SELF_IP").unwrap_or_default();
    let (tx, rx) = sync_channel(QUEUE_CAPACITY);
    if SENDER.set(tx).is_err() {
        return;
    }
    info!(
        "lingjing affinity sync: peers={} port={} self={}",
        peers_host, port, self_ip
    );
    BOOTSTRAPPED.store(false, Ordering::Release);
    tokio::spawn(bootstrap(peers_host.clone(), port, self_ip.clone()));
    tokio::spawn(run(peers_host, port, self_ip, rx));
}

fn drain(rx: &Receiver<Affinity>) -> Vec<Affinity> {
    let mut seen = std::collections::HashSet::new();
    let mut batch = Vec::new();
    while let Ok(a) = rx.try_recv() {
        if seen.insert(a.clone()) {
            batch.push(a);
        }
        if batch.len() >= 2000 {
            break;
        }
    }
    batch
}

async fn resolve_peers(host: &str, port: u16, self_ip: &str) -> Vec<String> {
    match tokio::net::lookup_host((host, port)).await {
        Ok(addrs) => {
            let mut v: Vec<String> = addrs
                .filter(|a| a.ip().to_string() != self_ip)
                .map(|a| a.to_string())
                .collect();
            v.sort();
            v.dedup();
            v
        }
        Err(e) => {
            debug!("lingjing affinity: resolve {} failed: {}", host, e);
            Vec::new()
        }
    }
}

/// Pull peers' snapshots; mark ready when done or at the deadline.
async fn bootstrap(host: String, port: u16, self_ip: String) {
    let start = Instant::now();
    // Trees are registered when the policy is built at startup; give that a moment.
    while TREES.get().is_none() && start.elapsed() < BOOTSTRAP_DEADLINE {
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .no_proxy()
        .build();
    let mut applied = 0;
    if let Ok(client) = client {
        for peer in resolve_peers(&host, port, &self_ip).await {
            if start.elapsed() >= BOOTSTRAP_DEADLINE {
                break;
            }
            match client
                .get(format!("http://{}/lingjing/affinity", peer))
                .send()
                .await
            {
                Ok(r) if r.status().is_success() => match r.json::<Vec<Affinity>>().await {
                    Ok(items) => applied += apply(&items),
                    Err(e) => warn!("lingjing affinity bootstrap: decode from {}: {}", peer, e),
                },
                Ok(r) => warn!(
                    "lingjing affinity bootstrap: {} answered {}",
                    peer,
                    r.status()
                ),
                Err(e) => warn!("lingjing affinity bootstrap: {}: {}", peer, e),
            }
        }
    }
    info!(
        "lingjing affinity bootstrap done: applied={} after {:?}",
        applied,
        start.elapsed()
    );
    BOOTSTRAPPED.store(true, Ordering::Release);
}

async fn run(host: String, port: u16, self_ip: String, rx: Receiver<Affinity>) {
    let client = match reqwest::Client::builder()
        .timeout(SEND_TIMEOUT)
        .no_proxy()
        .build()
    {
        Ok(c) => c,
        Err(e) => {
            warn!("lingjing affinity: http client: {}", e);
            return;
        }
    };
    let mut peers = Vec::new();
    let mut refreshed: Option<Instant> = None;
    loop {
        tokio::time::sleep(FLUSH_INTERVAL).await;
        let batch = drain(&rx);
        if batch.is_empty() {
            continue;
        }
        if refreshed.map_or(true, |t| t.elapsed() >= PEER_REFRESH) {
            peers = resolve_peers(&host, port, &self_ip).await;
            refreshed = Some(Instant::now());
        }
        for peer in &peers {
            let url = format!("http://{}/lingjing/affinity", peer);
            let client = client.clone();
            let body = batch.clone();
            tokio::spawn(async move {
                let ok = matches!(client.post(&url).json(&body).send().await, Ok(r) if r.status().is_success());
                let result = if ok { "sent" } else { "send_failed" };
                metrics::counter!("vllm_router_lingjing_affinity_total", "result" => result)
                    .increment(body.len() as u64);
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_inserts_short_texts_and_creates_missing_trees() {
        let trees: Trees = Arc::new(DashMap::new());
        let tree = Arc::new(Tree::new());
        tree.insert("", "http://10.0.0.1:8000");
        trees.insert("m".to_string(), tree.clone());
        register_trees(trees.clone());
        let long = "x".repeat(MAX_SYNC_TEXT_BYTES + 1);
        let n = apply(&[
            Affinity {
                model: "m".into(),
                text: "aaaaaaaaaaaaaaaa".into(),
                url: "http://10.0.0.2:8000".into(),
            },
            // fresh router: no tree for this model yet -> created
            Affinity {
                model: "fresh".into(),
                text: "cccccccccccccccc".into(),
                url: "http://10.0.0.1:8000".into(),
            },
            Affinity {
                model: "m".into(),
                text: long.clone(),
                url: "http://10.0.0.1:8000".into(),
            },
            Affinity {
                model: "m".into(),
                text: "".into(),
                url: "http://10.0.0.1:8000".into(),
            },
        ]);
        assert_eq!(n, 2);
        let r = tree.prefix_match_with_counts("aaaaaaaaaaaaaaaa");
        assert_eq!(&*r.tenant, "http://10.0.0.2:8000");
        assert_eq!(r.matched_char_count, 16);
        assert_eq!(tree.prefix_match_with_counts(&long).matched_char_count, 0);
        let fresh = trees.get("fresh").expect("tree created for unseen model");
        assert_eq!(
            &*fresh.prefix_match_with_counts("cccccccccccccccc").tenant,
            "http://10.0.0.1:8000"
        );
    }

    #[test]
    fn journal_snapshot_keeps_latest_per_session() {
        record("m2", "s1", "http://a:8000");
        record("m2", "s1", "http://b:8000");
        record("m2", "s2", "http://a:8000");
        let mut snap: Vec<_> = snapshot().into_iter().filter(|a| a.model == "m2").collect();
        snap.sort_by(|x, y| x.text.cmp(&y.text));
        assert_eq!(snap.len(), 2);
        assert_eq!(snap[0].url, "http://b:8000");
        assert_eq!(snap[1].text, "s2");
    }

    #[test]
    fn drain_dedupes() {
        let (tx, rx) = sync_channel(10);
        let a = Affinity {
            model: "m".into(),
            text: "t".into(),
            url: "u".into(),
        };
        tx.send(a.clone()).unwrap();
        tx.send(a.clone()).unwrap();
        assert_eq!(drain(&rx), vec![a]);
    }
}
