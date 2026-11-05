/*
    Cache-Aware Load Balancing Router

    This router combines two strategies to optimize both cache utilization and request distribution:

    1. Cache-Aware Routing (Approximate Tree)
    2. Load Balancing (Shortest Queue with Balance Thresholds)

    The router dynamically switches between these strategies based on load conditions:
    - Uses load balancing when the system is imbalanced
    - Uses cache-aware routing when the system is balanced

    A system is considered imbalanced if both conditions are met:
    1. (max - min) > abs_threshold
    2. max > rel_threshold * min

    Strategy Details:

    1. Cache-Aware Routing (Approximate Tree)
    -------------------------------------------
    This strategy maintains an approximate radix tree for each worker based on request history,
    eliminating the need for direct cache state queries. The tree stores raw text characters
    instead of token IDs to avoid tokenization overhead.

    Process:
    a. For each request, find the worker with the highest prefix match
    b. If match rate > cache_threshold:
    Route to the worker with highest match (likely has relevant data cached)
    c. If match rate ≤ cache_threshold:
    Route to the worker with smallest tree size (most available cache capacity)
    d. Background maintenance:
    Periodically evict least recently used leaf nodes to prevent memory overflow

    2. Load Balancing (Shortest Queue)
    -------------------------------------------
    This strategy tracks pending request counts per worker and routes new requests
    to the least busy worker when the system is detected to be imbalanced.

    Configuration Parameters:
    ------------------------
    1. cache_threshold: (float, 0.0 to 1.0)
    Minimum prefix match ratio to use highest-match routing.
    Below this threshold, routes to worker with most available cache space.

    2. balance_abs_threshold: (integer)
    Absolute difference threshold for load imbalance detection.
    System is potentially imbalanced if (max_load - min_load) > abs_threshold

    3. balance_rel_threshold: (float)
    Relative ratio threshold for load imbalance detection.
    System is potentially imbalanced if max_load > min_load * rel_threshold
    Used in conjunction with abs_threshold to determine final imbalance state.

    4. eviction_interval_secs: (integer)
    Interval between LRU eviction cycles for the approximate trees.

    5. max_tree_size: (integer)
    Maximum nodes per tree. When exceeded, LRU leaf nodes are evicted
    during the next eviction cycle.
*/

use super::{get_healthy_worker_indices, CacheAwareConfig, LoadBalancingPolicy, RequestHeaders};
use crate::core::Worker;
use crate::metrics::RouterMetrics;
use crate::policies::normalize_model_key;
use crate::tree::Tree;
use dashmap::DashMap;
use rand::Rng;
use std::collections::HashMap;
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use tracing::{debug, info};

/// Backend candidate for a side-effect-free cache-affinity query.
#[derive(Debug, Clone, Copy)]
pub struct CacheAwareCandidate<'a> {
    pub target_id: &'a str,
    pub kv_pressure: f64,
}

/// Cache-affinity result committed separately by the caller.
#[derive(Debug, Clone, PartialEq)]
pub struct CacheAwarePlacement {
    pub target_id: String,
    pub match_rate: f32,
}

/// Cache-aware routing policy
///
/// Routes requests based on cache affinity when load is balanced,
/// switches to shortest-queue routing when load is imbalanced.
/// Maintains separate trees per model for multi-model support.
#[derive(Debug)]
pub struct CacheAwarePolicy {
    config: CacheAwareConfig,
    trees: Arc<DashMap<String, Arc<Tree>>>, // model_id -> Arc<Tree>
    eviction_handle: Option<thread::JoinHandle<()>>,
    /// lingjing 0006: saturation state; None = the process-wide sampler (if started).
    lingjing_load: Option<Arc<crate::lingjing_load::LoadState>>,
    /// lingjing 0008: None = the process-wide state (None too while every feature is off).
    lingjing_warmup: Option<Arc<crate::lingjing_warmup::State>>,
    /// lingjing 0009: None = the process-wide state (None too while off).
    lingjing_uncached: Option<Arc<crate::lingjing_uncached::State>>,
}

impl CacheAwarePolicy {
    pub fn new() -> Self {
        Self::with_config(CacheAwareConfig::default())
    }

    pub fn with_config(config: CacheAwareConfig) -> Self {
        let trees = Arc::new(DashMap::<String, Arc<Tree>>::new());

        // Start background eviction thread if configured
        let eviction_handle = if config.eviction_interval_secs > 0 {
            let trees_clone = Arc::clone(&trees);
            let max_tree_size = config.max_tree_size;
            let interval = config.eviction_interval_secs;

            Some(thread::spawn(move || loop {
                thread::sleep(Duration::from_secs(interval));

                // Evict for all model trees
                for entry in trees_clone.iter() {
                    let model_id = entry.key();
                    let tree = entry.value();
                    tree.evict_tenant_by_size(max_tree_size);
                    debug!(
                        "Cache eviction completed for model {}, max_size: {}",
                        model_id, max_tree_size
                    );
                }
            }))
        } else {
            None
        };

        // lingjing patch: let peer routers' affinities reach these trees.
        crate::lingjing_affinity::register_trees(Arc::clone(&trees));

        Self {
            config,
            trees,
            eviction_handle,
            lingjing_load: None,
            lingjing_warmup: None,
            lingjing_uncached: None,
        }
    }

    /// lingjing 0006: use this saturation state instead of the process-wide one (tests).
    pub fn with_lingjing_load(mut self, load: Arc<crate::lingjing_load::LoadState>) -> Self {
        self.lingjing_load = Some(load);
        self
    }

    /// lingjing 0008: use this state instead of the process-wide one (tests).
    pub fn with_lingjing_warmup(mut self, state: Arc<crate::lingjing_warmup::State>) -> Self {
        self.lingjing_warmup = Some(state);
        self
    }

    /// lingjing 0009: use this state instead of the process-wide one (tests).
    pub fn with_lingjing_uncached(mut self, state: Arc<crate::lingjing_uncached::State>) -> Self {
        self.lingjing_uncached = Some(state);
        self
    }

    pub(crate) fn lingjing_uncached(&self) -> Option<&Arc<crate::lingjing_uncached::State>> {
        self.lingjing_uncached
            .as_ref()
            .or_else(|| crate::lingjing_uncached::state())
    }

    /// lingjing 0009 time gate: is this hit better off moving? Staying costs the replica's wait
    /// plus the new bytes; moving costs the best other replica's wait plus the whole body.
    fn lingjing_hit_should_move(
        &self,
        text: &str,
        target: usize,
        workers: &[Arc<dyn Worker>],
        healthy: &[usize],
        body: Option<u64>,
    ) -> bool {
        use crate::lingjing_warmup::Mode;
        let Some(state) = self.lingjing_uncached() else {
            return false;
        };
        let turn = self.lingjing_warmup().and_then(|w| w.last_turn(text));
        let last = turn.as_ref().map(|t| (t.worker.as_str(), t.body_bytes));
        let url = workers[target].url();
        let stay = state.wait_ms(url, state.cost(url, body, last));
        let away = healthy
            .iter()
            .filter(|&&idx| idx != target)
            .map(|&idx| state.wait_ms(workers[idx].url(), state.tokens_for(body)))
            .min();
        let moves = away.is_some_and(|away| state.should_move(stay, away));
        if moves {
            crate::lingjing_uncached::record(if state.cfg.mode == Mode::On {
                "moved_hit"
            } else {
                "would_move_hit"
            });
        }
        moves && state.cfg.mode == Mode::On
    }

    pub(crate) fn lingjing_warmup(&self) -> Option<&Arc<crate::lingjing_warmup::State>> {
        self.lingjing_warmup
            .as_ref()
            .or_else(|| crate::lingjing_warmup::state())
    }

    /// lingjing 0008: seed a warming worker's credit: the mean of the other workers' text minus
    /// its own, so its empty tree does not win every tie but starts level with an average
    /// replica. `workers` are the registered workers of the model.
    pub(crate) fn lingjing_seed_credit(&self, url: &str, workers: &[Arc<dyn Worker>]) {
        let (Some(state), Some(first)) = (self.lingjing_warmup(), workers.first()) else {
            return;
        };
        let model_id = normalize_model_key(first.model_id());
        let Some(tree) = self.trees.get(model_id).map(|entry| entry.value().clone()) else {
            return;
        };
        let held = |worker: &str| tree.tenant_char_count.get(worker).map_or(0, |count| *count);
        let others: Vec<usize> =
            workers.iter().filter(|w| w.url() != url).map(|w| held(w.url())).collect();
        let mean = others.iter().sum::<usize>().checked_div(others.len()).unwrap_or(0);
        state.set_credit(url, mean.saturating_sub(held(url)));
    }

    /// The worker a hit should go to: the matched tenant, or with kept affinity another usable
    /// tenant of the same node. The tree is not touched, so a returning worker gets its sessions
    /// back (`last_tenant` still names it) and a session does not hop between replicas meanwhile.
    fn lingjing_hit_target(
        &self,
        tree: &Tree,
        text: &str,
        tenant: &str,
        workers: &[Arc<dyn Worker>],
    ) -> Option<usize> {
        use crate::lingjing_warmup::{record_keep_affinity, Mode};
        let usable = |url: &str| {
            workers
                .iter()
                .position(|w| w.url() == url)
                .filter(|&idx| workers[idx].is_healthy())
        };
        if let Some(idx) = usable(tenant) {
            return Some(idx);
        }
        let state = self.lingjing_warmup().filter(|s| s.cfg.keep_affinity == Mode::On)?;
        state.retain(tenant);
        let other = tree.matched_tenants(text).iter().find_map(|t| usable(t));
        if let Some(idx) = other {
            record_keep_affinity("other_tenant", workers[idx].url());
        }
        other
    }

    pub(crate) fn lingjing_load(&self) -> Option<&crate::lingjing_load::LoadState> {
        self.lingjing_load
            .as_deref()
            .or_else(|| crate::lingjing_load::state().map(|s| &**s))
    }

    /// lingjing 0007: would `select_worker_with_headers` take its cache-hit branch for `text`
    /// (match rate above the threshold on a healthy worker)? Read-only: no insert, no
    /// processed/policy counters.
    pub(crate) fn has_cache_hit(&self, workers: &[Arc<dyn Worker>], text: &str) -> bool {
        let healthy = get_healthy_worker_indices(workers);
        let Some(&first) = healthy.first() else {
            return false;
        };
        if text.is_empty() {
            return false;
        }
        let model_id = normalize_model_key(workers[first].model_id());
        let Some(tree) = self.trees.get(model_id).map(|entry| entry.value().clone()) else {
            return false;
        };
        let result = tree.prefix_match_with_counts(text);
        if result.input_char_count == 0 {
            return false;
        }
        let match_rate = result.matched_char_count as f32 / result.input_char_count as f32;
        let tenant: &str = &result.tenant;
        match_rate > self.config.cache_threshold
            && !self.lingjing_is_false_hit(text, &result, crate::lingjing_warmup::Mode::On)
            && workers
                .iter()
                .any(|w| w.url() == tenant && w.is_healthy())
    }

    /// lingjing 0008: does this match clear the threshold only because two session hashes share
    /// a few leading chars? Answers true only while exact matching is in `mode`.
    fn lingjing_is_false_hit(
        &self,
        text: &str,
        result: &crate::tree::PrefixMatchResult,
        mode: crate::lingjing_warmup::Mode,
    ) -> bool {
        self.lingjing_warmup().is_some_and(|state| state.cfg.exact_session == mode)
            && result.matched_char_count < result.input_char_count
            && crate::lingjing_warmup::is_session_hash(text)
    }

    /// lingjing 0006: the `key`-minimal healthy worker, steered off a saturated replica for
    /// new sessions (see lingjing_load::place_new_session), and with 0008 off warming workers
    /// (see lingjing_warmup::State::place).
    fn pick_new_session<K: Ord>(
        &self,
        workers: &[Arc<dyn Worker>],
        healthy_indices: &[usize],
        placement: crate::lingjing_warmup::Placement,
        key: impl Fn(usize) -> K,
    ) -> Option<usize> {
        let mut ranked = healthy_indices.to_vec();
        ranked.sort_by_key(|&idx| key(idx));
        let load = self.lingjing_load();
        let pick = |candidates: &[usize]| {
            crate::lingjing_load::place_new_session(
                load,
                workers,
                candidates,
                self.config.balance_abs_threshold,
            )
        };
        match self.lingjing_warmup() {
            None => pick(&ranked),
            Some(state) => state.place(
                workers,
                &ranked,
                placement,
                |url| {
                    load.is_some_and(|l| l.is_saturated(url))
                        || self
                            .lingjing_uncached()
                            .filter(|u| u.cfg.mode == crate::lingjing_warmup::Mode::On)
                            .is_some_and(|u| u.is_saturated(url))
                },
                pick,
            ),
        }
    }

    /// Select by the product of text-prefix affinity and current KV headroom.
    pub fn evaluate_initial_placement(
        &self,
        model_id: &str,
        request_text: &str,
        candidates: &[CacheAwareCandidate<'_>],
    ) -> Option<CacheAwarePlacement> {
        let tree = self.trees.get(normalize_model_key(model_id));
        let input_chars = request_text.chars().count();
        candidates
            .iter()
            .map(|candidate| {
                let matched_chars = tree.as_ref().map_or(0, |tree| {
                    tree.prefix_match_tenant_char_count(request_text, candidate.target_id)
                });
                let affinity = if input_chars == 0 {
                    0.0
                } else {
                    matched_chars as f64 / input_chars as f64
                };
                let headroom = 1.0 - candidate.kv_pressure.clamp(0.0, 1.0);
                (candidate, affinity, headroom, (1.0 + affinity) * headroom)
            })
            .max_by(|left, right| {
                left.3
                    .total_cmp(&right.3)
                    .then_with(|| left.2.total_cmp(&right.2))
                    .then_with(|| left.1.total_cmp(&right.1))
                    .then_with(|| right.0.target_id.cmp(left.0.target_id))
            })
            .map(|(candidate, affinity, _, _)| CacheAwarePlacement {
                target_id: candidate.target_id.to_string(),
                match_rate: affinity as f32,
            })
    }

    /// Update the affinity tree after the selected target is committed.
    pub fn commit_placement(&self, model_id: &str, request_text: &str, target_id: &str) {
        if request_text.is_empty() {
            return;
        }
        self.trees
            .entry(normalize_model_key(model_id).to_string())
            .or_insert_with(|| Arc::new(Tree::new()))
            .insert(request_text, target_id);
    }

    /// Remove one unavailable target from a model's affinity tree.
    pub fn remove_placement_target(&self, model_id: &str, target_id: &str) {
        if let Some(tree) = self.trees.get(normalize_model_key(model_id)) {
            tree.remove_tenant(target_id);
        }
    }

    /// Add a single worker to the tree (incremental update)
    pub fn add_worker(&self, worker: &dyn Worker) {
        let tree_key = normalize_model_key(worker.model_id());
        let tree = self
            .trees
            .entry(tree_key.to_string())
            .or_insert_with(|| Arc::new(Tree::new()));
        tree.insert("", worker.url());
    }

    /// Add a worker by URL and model (for backward compatibility)
    pub fn add_worker_by_url(&self, url: &str, model_id: &str) {
        let tree = self
            .trees
            .entry(model_id.to_string())
            .or_insert_with(|| Arc::new(Tree::new()));
        tree.insert("", url);
    }

    /// Remove a worker from the tree
    pub fn remove_worker(&self, worker: &dyn Worker) {
        let tree_key = normalize_model_key(worker.model_id());
        if let Some(tree) = self.trees.get(tree_key) {
            tree.remove_tenant(worker.url());
        }
    }

    /// Remove a worker by URL (removes from all model trees for backward compatibility)
    pub fn remove_worker_by_url(&self, url: &str) {
        // Remove from all trees since we don't know which model it belongs to
        for tree_ref in self.trees.iter() {
            tree_ref.value().remove_tenant(url);
        }
    }

    /// Run cache eviction to prevent unbounded growth
    pub fn evict_cache(&self, max_size: usize) {
        for tree_ref in self.trees.iter() {
            let model_id = tree_ref.key();
            let tree = tree_ref.value();
            tree.evict_tenant_by_size(max_size);
            debug!(
                "Cache eviction for model {}, max_size: {}",
                model_id, max_size
            );
        }
    }

    fn select_worker_min_load(
        &self,
        workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
        healthy_indices: &[usize],
        model_id: &str,
        max_load: usize,
        min_load: usize,
        placement: crate::lingjing_warmup::Placement,
    ) -> Option<usize> {
        // Log load balancing trigger (only compute worker loads if debug enabled)
        if tracing::enabled!(tracing::Level::DEBUG) {
            let worker_loads: Vec<(&str, usize)> =
                workers.iter().map(|w| (w.url(), w.load())).collect();
            debug!(
                "Load balancing triggered | max: {} | min: {} | workers: {:?}",
                max_load, min_load, worker_loads
            );
        }

        RouterMetrics::record_load_balancing_event();
        RouterMetrics::set_load_range(max_load, min_load);

        // Use shortest queue when imbalanced (lingjing 0006: never onto a saturated replica
        // while an unsaturated one exists)
        let min_load_idx = self.pick_new_session(workers, healthy_indices, placement, |idx| {
            workers[idx].load()
        })?;

        // Even in imbalanced mode, update the tree to maintain cache state
        if let Some(text) = request_text {
            // Get the tree reference without locking the entire HashMap
            // DashMap only locks the specific shard containing this key
            let tree = self.trees.get(model_id).map(|entry| entry.value().clone());

            if let Some(tree) = tree {
                let worker_url = workers[min_load_idx].url();
                tree.insert(text, worker_url);
                crate::lingjing_affinity::publish(model_id, text, worker_url);
            } else {
                debug!(
                    "Warning: No tree found for model '{}', skipping cache update",
                    model_id
                );
            }
        }

        // Increment processed counter
        workers[min_load_idx].increment_processed();
        RouterMetrics::record_processed_request(workers[min_load_idx].url());
        RouterMetrics::record_policy_decision(self.name(), workers[min_load_idx].url());

        Some(min_load_idx)
    }
}

impl CacheAwarePolicy {
    fn select_worker_inner(
        &self,
        workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
        headers: Option<&RequestHeaders>,
    ) -> Option<usize> {
        let healthy_indices = get_healthy_worker_indices(workers);

        if healthy_indices.is_empty() {
            return None;
        }

        // Determine the model for this set of workers (router pre-filters by model)
        // All workers should be from the same model
        let model_id = normalize_model_key(workers[healthy_indices[0]].model_id());

        // Base the load decision on workers that are eligible for selection.
        // Unhealthy workers must not affect whether healthy workers use cache
        // affinity or load balancing.
        let (min_load, max_load) = healthy_indices
            .iter()
            .map(|&idx| workers[idx].load())
            .fold((usize::MAX, 0usize), |(min, max), load| {
                (min.min(load), max.max(load))
            });
        let min_load = if min_load == usize::MAX { 0 } else { min_load };

        // Check if load is imbalanced
        let is_imbalanced = max_load.saturating_sub(min_load) > self.config.balance_abs_threshold
            && (max_load as f32) > (min_load as f32 * self.config.balance_rel_threshold);

        debug!(
            "Load status for model: max_load={}, min_load={}, is_imbalanced={}",
            max_load, min_load, is_imbalanced
        );

        let warmup = self.lingjing_warmup();
        let mut placement = warmup.map_or_else(Default::default, |state| {
            let count = state.cfg.warmup_mode != crate::lingjing_warmup::Mode::Off;
            crate::lingjing_warmup::Placement::from_headers(headers, count)
        });
        let mut kept_match = None;
        let mut would_false_hit = false;
        if is_imbalanced {
            // lingjing 0008: a hit on a usable, unsaturated worker stays. The match is only
            // computed with a feature on: it refreshes tenant timestamps, and off means 0007.
            if let Some(state) = warmup {
                use crate::lingjing_warmup::{record_imbalance, Mode};
                let text = request_text.unwrap_or("");
                let tree = self.trees.get(model_id).map(|entry| entry.value().clone());
                let matched = tree.as_ref().map(|tree| {
                    let result = tree.prefix_match_with_counts(text);
                    let mut hit = result.input_char_count > 0
                        && result.matched_char_count as f32 / result.input_char_count as f32
                            > self.config.cache_threshold;
                    if hit && self.lingjing_is_false_hit(text, &result, Mode::On) {
                        crate::lingjing_warmup::record_session_match("false_hit_avoided");
                        hit = false;
                    } else if hit && self.lingjing_is_false_hit(text, &result, Mode::Shadow) {
                        // Counted in this branch only when the hit path below will not count it.
                        would_false_hit = true;
                    }
                    let target = hit
                        .then(|| self.lingjing_hit_target(tree, text, &result.tenant, workers))
                        .flatten();
                    (result, hit, target)
                });
                match matched {
                    Some((result, true, Some(idx))) => {
                        let url = workers[idx].url();
                        let saturated =
                            self.lingjing_load().is_some_and(|load| load.is_saturated(url));
                        match (saturated, state.cfg.keep_hit) {
                            (false, Mode::On) => {
                                record_imbalance("kept_hit", url);
                                kept_match = Some(result);
                            }
                            (false, Mode::Shadow) => record_imbalance("would_keep_hit", url),
                            _ => {
                                record_imbalance("moved_hit", url);
                                placement.moved_hit = true;
                            }
                        }
                    }
                    Some((result, true, None)) => record_imbalance("moved_hit", &result.tenant),
                    _ => record_imbalance("moved_new", "-"),
                }
            }
            if kept_match.is_none() {
                if would_false_hit {
                    crate::lingjing_warmup::record_session_match("would_false_hit");
                }
                return self.select_worker_min_load(
                    workers,
                    request_text,
                    &healthy_indices,
                    model_id,
                    max_load,
                    min_load,
                    placement,
                );
            }
        }

        // Use cache-aware routing when balanced
        let text = request_text.unwrap_or("");

        // Get the tree reference without locking the entire HashMap
        // DashMap only locks the specific shard containing this key
        let tree = self.trees.get(model_id).map(|entry| entry.value().clone());

        let keys: Vec<_> = self.trees.iter().map(|entry| entry.key().clone()).collect();
        debug!("Available tree keys: {:?}", keys);

        let Some(tree) = tree else {
            // No tree for this model, log warning and use random selection
            debug!(
                "Warning: No tree found for model '{}', using random worker selection",
                model_id
            );
            // Return a random healthy worker
            let mut rng = rand::rng();
            let random_idx = rng.random_range(0..healthy_indices.len());
            let selected_idx = healthy_indices[random_idx];

            workers[selected_idx].increment_processed();
            RouterMetrics::record_processed_request(workers[selected_idx].url());
            RouterMetrics::record_policy_decision(self.name(), workers[selected_idx].url());

            return Some(selected_idx);
        };
        debug!("Using cache-aware routing for model '{}'", model_id);
        // Now we work with the tree without holding the HashMap lock
        // Use prefix_match_with_counts to avoid redundant chars().count() calls
        let result = kept_match.unwrap_or_else(|| tree.prefix_match_with_counts(text));
        let match_rate = if result.input_char_count == 0 {
            0.0
        } else {
            result.matched_char_count as f32 / result.input_char_count as f32
        };

        debug!(
            "Cache match for model '{}': matched_chars={}, input_chars={}, match_rate={:.2}",
            model_id, result.matched_char_count, result.input_char_count, match_rate
        );
        // Low cache match: use worker with minimum load. lingjing patch: break
        // load ties by the least text held in this worker's tree, so a lightly
        // loaded fleet (every load == 0) spreads new conversations instead of
        // always picking the first healthy worker.
        // lingjing 0006: a new conversation is not placed on a saturated replica while an
        // unsaturated one exists; the cache-hit branch above keeps its affinity regardless.
        // lingjing 0008: a warming worker's text counts with its credit once warmup is on.
        let warmup_on = warmup.filter(|s| s.cfg.warmup_mode == crate::lingjing_warmup::Mode::On);
        let uncached_on = self
            .lingjing_uncached()
            .filter(|u| u.cfg.mode == crate::lingjing_warmup::Mode::On);
        let new_session_key = |idx: usize, credit: usize| {
            let held = tree
                .tenant_char_count
                .get(workers[idx].url())
                .map(|count| *count)
                .unwrap_or(0);
            let wait = uncached_on.map_or(0, |u| u.wait_ms(workers[idx].url(), 0));
            (wait, workers[idx].load(), held + credit)
        };
        let pick_new = |placement: crate::lingjing_warmup::Placement| {
            self.pick_new_session(workers, &healthy_indices, placement, |idx| {
                new_session_key(idx, warmup_on.map_or(0, |s| s.credit(workers[idx].url())))
            })
        };

        // lingjing 0008: a session hash either matches whole or is a new session.
        let threshold_hit = match_rate > self.config.cache_threshold;
        let false_hit = threshold_hit
            && self.lingjing_is_false_hit(text, &result, crate::lingjing_warmup::Mode::On);
        if warmup.is_some_and(|s| s.cfg.exact_session != crate::lingjing_warmup::Mode::Off)
            && crate::lingjing_warmup::is_session_hash(text)
        {
            crate::lingjing_warmup::record_session_match(if false_hit {
                "false_hit_avoided"
            } else if !threshold_hit {
                "new"
            } else if result.matched_char_count < result.input_char_count {
                "would_false_hit"
            } else {
                "exact"
            });
        }
        let is_hit = threshold_hit && !false_hit;

        // Select worker without String allocation
        let keep_on = warmup
            .is_some_and(|s| s.cfg.keep_affinity == crate::lingjing_warmup::Mode::On);
        let selected_idx = if is_hit {
            // Cache hit path: find worker by URL (compare &str directly, no allocation)
            let target = self.lingjing_hit_target(&tree, text, &result.tenant, workers);
            let body = placement.body_bytes;
            if target.is_some_and(|idx| {
                self.lingjing_hit_should_move(text, idx, workers, &healthy_indices, body)
            }) {
                pick_new(crate::lingjing_warmup::Placement { moved_hit: true, ..placement })
            } else if target.is_none() && keep_on {
                // lingjing 0008: the worker is away and nobody else holds this session.
                crate::lingjing_warmup::record_keep_affinity("new_session", &result.tenant);
                pick_new(placement)
            } else {
                target
            }
        } else {
            let chosen = pick_new(placement);
            if let Some(state) =
                warmup.filter(|s| s.cfg.warmup_mode == crate::lingjing_warmup::Mode::Shadow)
            {
                // Shadow: would the credit alone have sent this session elsewhere?
                let credited = healthy_indices
                    .iter()
                    .copied()
                    .min_by_key(|&idx| new_session_key(idx, state.credit(workers[idx].url())));
                if let (Some(actual), Some(would)) = (chosen, credited) {
                    if actual != would && state.has_warming(workers, &[actual]) {
                        crate::lingjing_warmup::record_warmup(
                            "would_shift_credit",
                            workers[actual].url(),
                            placement.body_bytes,
                        );
                    }
                }
            }
            chosen
        };

        if let Some(idx) = selected_idx {
            // Update the tree with this request (use worker URL directly, no allocation)
            tree.insert(text, workers[idx].url());
            crate::lingjing_affinity::publish(model_id, text, workers[idx].url());

            // Increment processed counter
            workers[idx].increment_processed();
            RouterMetrics::record_processed_request(workers[idx].url());
            RouterMetrics::record_policy_decision(self.name(), workers[idx].url());

            return Some(idx);
        }

        // Selected worker no longer exists or unhealthy, remove stale tenant from tree
        if is_hit {
            let tenant_url: &str = &result.tenant;
            if warmup.is_some_and(|s| s.cfg.keep_affinity == crate::lingjing_warmup::Mode::Shadow) {
                crate::lingjing_warmup::record_keep_affinity("would_retain", tenant_url);
            }
            tree.remove_tenant(tenant_url);
            debug!("Removed stale worker {} from cache tree", tenant_url);
        }

        // Fallback to first healthy worker (lingjing 0008: not a warming one while there are others)
        let settled = warmup_on.and_then(|state| {
            healthy_indices
                .iter()
                .copied()
                .find(|&idx| !state.is_warming(workers[idx].url()))
        });
        if let Some(idx) = settled.or_else(|| healthy_indices.first().copied()) {
            workers[idx].increment_processed();
            RouterMetrics::record_processed_request(workers[idx].url());
            RouterMetrics::record_policy_decision(self.name(), workers[idx].url());

            Some(idx)
        } else {
            None
        }
    }

}

impl LoadBalancingPolicy for CacheAwarePolicy {
    fn select_worker_with_headers(
        &self,
        workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
        headers: Option<&RequestHeaders>,
    ) -> Option<usize> {
        let chosen = self.select_worker_inner(workers, request_text, headers);
        if let (Some(state), Some(idx), Some(text)) = (self.lingjing_warmup(), chosen, request_text) {
            let body = crate::lingjing_warmup::Placement::from_headers(headers, false).body_bytes;
            state.observe_route(text, workers[idx].url(), body);
        }
        chosen
    }

    fn name(&self) -> &'static str {
        "cache_aware"
    }

    fn needs_request_text(&self) -> bool {
        true // Cache-aware policy needs request text for cache affinity
    }

    fn on_request_complete(&self, worker_url: &str, success: bool) {
        // Could track success rates per worker for more intelligent routing
        if !success {
            // Optionally reduce affinity for failed requests
            tracing::debug!(
                "Request to {} completed with success={}",
                worker_url,
                success
            );
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn select_worker_pair_with_headers(
        &self,
        prefill_workers: &[Arc<dyn Worker>],
        decode_workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
        headers: Option<&RequestHeaders>,
    ) -> Option<(usize, usize)> {
        // DEPRECATED: This method is no longer used when separate policies are configured.
        // The PD router now uses separate policies for prefill and decode selection.
        // This implementation remains for backward compatibility when a single policy is used.

        // In PD mode with single policy:
        // - Prefill: Use cache-aware routing for better cache utilization
        // - Decode: Use least-load routing for better load distribution

        // Select prefill worker using cache-aware logic
        let prefill_idx =
            self.select_worker_with_headers(prefill_workers, request_text, headers)?;

        // Select decode worker using least-load logic
        let healthy_decode = get_healthy_worker_indices(decode_workers);
        if healthy_decode.is_empty() {
            return None;
        }

        let decode_idx = healthy_decode
            .iter()
            .min_by_key(|&&idx| decode_workers[idx].load())
            .copied()?;

        Some((prefill_idx, decode_idx))
    }

    fn requires_initialization(&self) -> bool {
        true // Cache-aware policy requires init_workers() to set up trees
    }

    fn init_workers(&self, workers: &[Arc<dyn Worker>]) {
        // Group workers by model
        info!(
            "Initializing workers for cache-aware policy: {}",
            workers
                .iter()
                .map(|w| w.url())
                .collect::<Vec<_>>()
                .join(", ")
        );
        let mut model_workers: HashMap<String, Vec<&Arc<dyn Worker>>> = HashMap::new();
        for worker in workers {
            let tree_key = normalize_model_key(worker.model_id());
            model_workers
                .entry(tree_key.to_string())
                .or_default()
                .push(worker);
        }

        // Initialize tree for each model
        for (tree_key, model_workers) in model_workers {
            info!(
                "Creating tree for model key: '{}' with {} workers",
                tree_key,
                model_workers.len()
            );
            let tree = self
                .trees
                .entry(tree_key)
                .or_insert_with(|| Arc::new(Tree::new()))
                .clone();
            for worker in model_workers {
                tree.insert("", worker.url());
            }
        }
    }
}

impl Default for CacheAwarePolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for CacheAwarePolicy {
    fn drop(&mut self) {
        // Note: We can't properly stop the eviction thread since it's in an infinite loop
        // In a production system, we'd use a channel or atomic flag to signal shutdown
        if let Some(handle) = self.eviction_handle.take() {
            // The thread will continue running until the program exits
            // This is acceptable for now since the router typically runs for the lifetime of the program
            drop(handle);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorker, WorkerType};

    #[test]
    fn unhealthy_worker_load_does_not_disable_cache_affinity() {
        let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
            cache_threshold: 0.5,
            balance_abs_threshold: 0,
            balance_rel_threshold: 1.0,
            eviction_interval_secs: 0,
            max_tree_size: 100,
        });
        let workers: Vec<Arc<dyn Worker>> = (0..3)
            .map(|index| {
                Arc::new(BasicWorker::new(
                    format!("http://worker-{index}:8000"),
                    WorkerType::Regular,
                )) as Arc<dyn Worker>
            })
            .collect();
        workers[0].set_healthy(false);
        workers[2].set_healthy(false);
        policy.init_workers(&workers);
        for _ in 0..5 {
            workers[0].increment_load();
            workers[1].increment_load();
        }

        // Establish affinity while worker 1 is the only healthy worker.
        assert_eq!(policy.select_worker(&workers, Some("sticky")), Some(1));
        workers[0].set_healthy(true);

        // Both eligible workers have equal load, so the unhealthy zero-load
        // worker must not switch the policy into load-balancing mode.
        assert_eq!(policy.select_worker(&workers, Some("sticky")), Some(1));
    }

    fn lingjing_fleet(avoid: bool) -> (CacheAwarePolicy, Vec<Arc<dyn Worker>>, Arc<crate::lingjing_load::LoadState>) {
        let load = Arc::new(crate::lingjing_load::LoadState::new(crate::lingjing_load::Config {
            avoid_saturated: avoid,
            ..Default::default()
        }));
        let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        })
        .with_lingjing_load(load.clone());
        let workers: Vec<Arc<dyn Worker>> = (0..3)
            .map(|index| {
                Arc::new(BasicWorker::new(format!("http://lj0006-{index}:8000"), WorkerType::Regular))
                    as Arc<dyn Worker>
            })
            .collect();
        policy.init_workers(&workers);
        // Worker 0 carries one request, so the ordinary choice for a new session is worker 1
        // (never the index-0 fallback, which would hide "fell through" behind "chose 0").
        workers[0].increment_load();
        (policy, workers, load)
    }

    #[test]
    fn new_sessions_avoid_saturated_replicas() {
        let (policy, workers, load) = lingjing_fleet(true);
        assert_eq!(policy.select_worker(&workers, Some("baseline newcomer")), Some(1));
        load.force(workers[1].url(), true);
        assert_eq!(policy.select_worker(&workers, Some("second newcomer")), Some(2));
        // Imbalanced (min-load) path: loads 41 / 0 / 10 (abs threshold 32). Worker 1 is the
        // least loaded but saturated; worker 2 is within the burst guard, so it takes the session.
        for _ in 0..40 {
            workers[0].increment_load();
        }
        for _ in 0..10 {
            workers[2].increment_load();
        }
        assert_eq!(policy.select_worker(&workers, Some("imbalanced newcomer")), Some(2));
    }

    #[test]
    fn avoidance_off_never_redirects() {
        let (policy, workers, load) = lingjing_fleet(false);
        load.force(workers[1].url(), true);
        for turn in 0..4 {
            assert_eq!(
                policy.select_worker(&workers, Some(&format!("shadow newcomer {turn}"))),
                Some(1),
                "with avoidance off the saturated ordinary choice must stand"
            );
            workers[1].decrement_load();
        }
    }

    #[test]
    fn cache_hit_keeps_affinity_on_saturated_replica() {
        let (policy, workers, load) = lingjing_fleet(true);
        let first = policy.select_worker(&workers, Some("long running session prefix"));
        assert_eq!(first, Some(1));
        load.force(workers[1].url(), true);
        assert_eq!(
            policy.select_worker(&workers, Some("long running session prefix, next turn")),
            Some(1),
            "an existing session must not be migrated off its KV"
        );
    }

    #[test]
    fn nothing_is_avoided_when_every_replica_is_saturated() {
        let (policy, workers, load) = lingjing_fleet(true);
        for worker in &workers {
            load.force(worker.url(), true);
        }
        // The ordinary choice (worker 1), not the index-0 fallback.
        assert_eq!(policy.select_worker(&workers, Some("everyone is busy")), Some(1));
    }

    #[test]
    fn redirect_is_refused_when_the_alternative_is_near_overload() {
        // max_num_seqs guard
        let load = Arc::new(crate::lingjing_load::LoadState::new(crate::lingjing_load::Config {
            avoid_saturated: true,
            max_num_seqs: 4,
            ..Default::default()
        }));
        let workers: Vec<Arc<dyn Worker>> = (0..2)
            .map(|index| {
                Arc::new(BasicWorker::new(format!("http://lj0006-guard-{index}:8000"), WorkerType::Regular))
                    as Arc<dyn Worker>
            })
            .collect();
        load.force(workers[0].url(), true);
        for _ in 0..4 {
            workers[1].increment_load();
        }
        assert_eq!(
            crate::lingjing_load::place_new_session(Some(&load), &workers, &[0, 1], 8),
            Some(0),
            "alternative at max_num_seqs must not take the session"
        );
        workers[1].decrement_load();
        assert_eq!(crate::lingjing_load::place_new_session(Some(&load), &workers, &[0, 1], 8), Some(1));
        // abs-threshold guard: alternative already >= best + abs
        let load = Arc::new(crate::lingjing_load::LoadState::new(crate::lingjing_load::Config {
            avoid_saturated: true,
            ..Default::default()
        }));
        load.force(workers[0].url(), true);
        for _ in 0..5 {
            workers[1].increment_load();
        }
        // workers[1].load() == 8, workers[0].load() == 0
        assert_eq!(crate::lingjing_load::place_new_session(Some(&load), &workers, &[0, 1], 8), Some(0));
        assert_eq!(crate::lingjing_load::place_new_session(Some(&load), &workers, &[0, 1], 9), Some(1));
    }

    // ── 0008: decision trace shared with the 0007 golden file ──

    /// Deterministic scripted traffic over the whole selection surface: new and returning
    /// sessions, long texts, load swings past the imbalance thresholds, saturation flips,
    /// scale-out, eviction + re-add, a replica going unhealthy, a stale tenant. One line per
    /// decision, plus each tree's per-worker text size at the end of every phase.
    fn lingjing_decision_trace(
        policy: &CacheAwarePolicy,
        load: &crate::lingjing_load::LoadState,
        on_add: &dyn Fn(&str, usize),
    ) -> String {
        let mut rng = 0x2545_f491_4f6c_dd1du64;
        let mut next = move |bound: u64| {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (rng >> 33) % bound
        };
        let pool: Vec<Arc<dyn Worker>> = (0..4)
            .map(|i| {
                Arc::new(BasicWorker::new(format!("http://trace-{i}:8000"), WorkerType::Regular))
                    as Arc<dyn Worker>
            })
            .collect();
        let mut active: Vec<Arc<dyn Worker>> = pool[..2].to_vec();
        policy.init_workers(&active);
        let mut out = String::new();
        let mut sessions: Vec<String> = Vec::new();
        let mut held: Vec<Arc<dyn Worker>> = Vec::new();
        let mut placed: HashMap<String, std::collections::HashSet<String>> = HashMap::new();
        let mut serial = 0x4e00u32;
        let mut fresh = move |r: u64| {
            serial += 1;
            format!("{}{r:015x}", char::from_u32(serial).unwrap())
        };
        for (phase, ops) in [300u32, 300, 200, 200, 200, 200].into_iter().enumerate() {
            match phase {
                1 => {
                    on_add(pool[2].url(), active.len());
                    active.push(pool[2].clone());
                    policy.init_workers(&active);
                }
                2 => {
                    policy.remove_worker_by_url(pool[1].url());
                    active.remove(1);
                }
                3 => {
                    on_add(pool[1].url(), active.len());
                    active.insert(1, pool[1].clone());
                    policy.init_workers(&active);
                }
                4 => pool[2].set_healthy(false),
                5 => {
                    // Tenant left in the tree, worker gone from the list (open breaker).
                    pool[2].set_healthy(true);
                    active.retain(|w| w.url() != pool[2].url());
                }
                _ => {}
            }
            for op in 0..ops {
                match next(9) {
                    0..=2 => {
                        let w = active[next(active.len() as u64) as usize].clone();
                        for _ in 0..=next(5) {
                            w.increment_load();
                            held.push(w.clone());
                        }
                    }
                    3..=4 => {
                        for _ in 0..next(8) {
                            if !held.is_empty() {
                                held.swap_remove(next(held.len() as u64) as usize).decrement_load();
                            }
                        }
                    }
                    5 => load.force(active[next(active.len() as u64) as usize].url(), next(2) == 0),
                    _ => {}
                }
                let text = match next(5) {
                    0 | 1 if !sessions.is_empty() => {
                        let again = sessions[next(sessions.len() as u64) as usize].clone();
                        // Once tenants have been removed, a node holding several tenants picks
                        // its survivor by DashMap order (random per process, in 0007 too):
                        // sessions placed on more than one worker are not replayed then, and
                        // every session starts with its own character so nodes are not shared.
                        if phase >= 2 && placed.get(&again).is_some_and(|w| w.len() > 1) {
                            fresh(next(u64::MAX))
                        } else {
                            again
                        }
                    }
                    2 => {
                        let s = fresh(next(u64::MAX));
                        sessions.push(s.clone());
                        s
                    }
                    3 => {
                        let s = fresh(next(u64::MAX)).repeat(12);
                        sessions.push(s.clone());
                        s
                    }
                    _ => fresh(next(u64::MAX)),
                };
                let headers: Option<crate::policies::RequestHeaders> = match next(3) {
                    0 => None,
                    1 => Some([("content-length".to_string(), "1024".to_string())].into()),
                    _ => Some([("content-length".to_string(), "4194304".to_string())].into()),
                };
                let picked = policy
                    .select_worker_with_headers(&active, Some(&text), headers.as_ref())
                    .map(|idx| active[idx].url().to_string());
                if let Some(url) = &picked {
                    placed.entry(text).or_default().insert(url.clone());
                }
                out.push_str(&format!("{phase}.{op} {}\n", picked.as_deref().unwrap_or("none")));
            }
            let tree = policy.trees.get("default").unwrap().value().clone();
            let mut sizes: Vec<_> = tree.get_used_size_per_tenant().into_iter().collect();
            sizes.sort();
            out.push_str(&format!("phase {phase} tree {sizes:?}\n"));
        }
        out
    }

    /// Session hashes only, drawn from a few leading prefixes so that most new sessions collide
    /// with an older one past the 0.3 threshold (5 of 16 chars and more), under load swings and a
    /// scale-out. The 0007 side of the exact-match switch. No worker leaves: survivor choice on
    /// multi-tenant nodes is random per process in 0007 (see `lingjing_decision_trace`).
    fn lingjing_hash_trace(
        policy: &CacheAwarePolicy,
        load: &crate::lingjing_load::LoadState,
    ) -> String {
        let mut rng = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move |bound: u64| {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (rng >> 33) % bound
        };
        let pool: Vec<Arc<dyn Worker>> = (0..3)
            .map(|i| {
                Arc::new(BasicWorker::new(format!("http://hash-{i}:8000"), WorkerType::Regular))
                    as Arc<dyn Worker>
            })
            .collect();
        let mut active: Vec<Arc<dyn Worker>> = pool[..2].to_vec();
        policy.init_workers(&active);
        let mut out = String::new();
        let mut sessions: Vec<String> = Vec::new();
        let mut held: Vec<Arc<dyn Worker>> = Vec::new();
        for op in 0..1200u32 {
            if op == 600 {
                active.push(pool[2].clone());
                policy.init_workers(&active);
            }
            match next(6) {
                0 => {
                    let w = active[next(active.len() as u64) as usize].clone();
                    for _ in 0..=next(12) {
                        w.increment_load();
                        held.push(w.clone());
                    }
                }
                1 | 2 => {
                    for _ in 0..next(10) {
                        if !held.is_empty() {
                            held.swap_remove(next(held.len() as u64) as usize).decrement_load();
                        }
                    }
                }
                3 => load.force(active[next(active.len() as u64) as usize].url(), next(3) == 0),
                _ => {}
            }
            let text = if next(3) == 0 && !sessions.is_empty() {
                sessions[next(sessions.len() as u64) as usize].clone()
            } else {
                let shared = 5 + next(6) as usize;
                let lead = format!("{:05x}", next(4) * 0x11111);
                let tail = format!("{:016x}", next(u64::MAX).wrapping_mul(0x2545_f491_4f6c_dd1d));
                let s = format!("{lead}{}", &tail[..11]);
                let s = format!("{}{}", &s[..shared.min(16)], &tail[..16 - shared.min(16)]);
                sessions.push(s.clone());
                s
            };
            let headers: Option<crate::policies::RequestHeaders> =
                Some([("content-length".to_string(), "1024".to_string())].into());
            let picked = policy
                .select_worker_with_headers(&active, Some(&text), headers.as_ref())
                .map(|idx| active[idx].url().to_string());
            out.push_str(&format!("{op} {}\n", picked.as_deref().unwrap_or("none")));
        }
        let tree = policy.trees.get("default").unwrap().value().clone();
        let mut sizes: Vec<_> = tree.get_used_size_per_tenant().into_iter().collect();
        sizes.sort();
        out.push_str(&format!("tree {sizes:?}\n"));
        out
    }

    fn trace_policy() -> (CacheAwarePolicy, Arc<crate::lingjing_load::LoadState>) {
        let load = Arc::new(crate::lingjing_load::LoadState::new(crate::lingjing_load::Config {
            avoid_saturated: true,
            ..Default::default()
        }));
        let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
            cache_threshold: 0.3,
            balance_abs_threshold: 8,
            balance_rel_threshold: 1.2,
            eviction_interval_secs: 0,
            max_tree_size: 1 << 24,
        })
        .with_lingjing_load(load.clone());
        (policy, load)
    }

    const GOLDEN_0007: &str = include_str!("../../tests/fixtures/lingjing_0007_decisions.txt");

    fn assert_same_as_0007(trace: String) {
        for (line, (got, want)) in trace.lines().zip(GOLDEN_0007.lines()).enumerate() {
            assert_eq!(got, want, "first difference from 0007 at line {}", line + 1);
        }
        assert_eq!(trace.lines().count(), GOLDEN_0007.lines().count());
    }

    const GOLDEN_0007_HASHES: &str =
        include_str!("../../tests/fixtures/lingjing_0007_hash_decisions.txt");

    fn assert_same_hash_trace_as_0007(trace: String) {
        for (line, (got, want)) in trace.lines().zip(GOLDEN_0007_HASHES.lines()).enumerate() {
            assert_eq!(got, want, "first difference from 0007 at line {}", line + 1);
        }
        assert_eq!(trace.lines().count(), GOLDEN_0007_HASHES.lines().count());
    }

    #[test]
    fn lingjing_all_off_decides_exactly_like_0007() {
        let (policy, load) = trace_policy();
        assert_same_as_0007(lingjing_decision_trace(&policy, &load, &|_, _| {}));
    }

    #[test]
    fn lingjing_all_off_decides_colliding_hashes_exactly_like_0007() {
        let (policy, load) = trace_policy();
        assert_same_hash_trace_as_0007(lingjing_hash_trace(&policy, &load));
    }

    #[test]
    fn lingjing_all_shadow_decides_colliding_hashes_exactly_like_0007() {
        use crate::lingjing_warmup::{Config, Mode, State};
        let state = Arc::new(State::new(Config {
            warmup_mode: Mode::Shadow,
            keep_hit: Mode::Shadow,
            keep_affinity: Mode::Shadow,
            exact_session: Mode::Shadow,
            route_quality: Mode::On,
            ..Config::default()
        }));
        let (policy, load) = trace_policy();
        let policy = policy.with_lingjing_warmup(state);
        let trace = counted(|| lingjing_hash_trace(&policy, &load));
        assert!(trace.1.get("vllm_router_lingjing_session_match_total:would_false_hit").copied().unwrap_or(0) > 100, "{:?}", trace.1);
        assert_same_hash_trace_as_0007(trace.0);
    }

    #[test]
    fn lingjing_all_shadow_decides_exactly_like_0007() {
        use crate::lingjing_warmup::{Config, Mode, State};
        let state = Arc::new(State::new(Config {
            warmup_mode: Mode::Shadow,
            keep_hit: Mode::Shadow,
            keep_affinity: Mode::Shadow,
            exact_session: Mode::Shadow,
            route_quality: Mode::On,
            startup_grace_secs: 0,
            ..Config::default()
        }));
        let (policy, load) = trace_policy();
        let policy = policy.with_lingjing_warmup(state.clone());
        let on_add = |url: &str, current: usize| {
            state.on_worker_adding("default", url, current);
            state.set_credit(url, 1 << 20);
        };
        assert_same_as_0007(lingjing_decision_trace(&policy, &load, &on_add));
        assert!(state.is_warming("http://trace-2:8000"), "shadow really saw the scale-out");
    }

    // ── 0008: slow start, hits stay, eviction keeps affinity ──

    use crate::lingjing_warmup::{Config as Lj8Config, Mode as Lj8Mode, State as Lj8State};
    use std::collections::{BTreeMap, BTreeSet};
    use std::sync::Mutex;

    const KIB256: u64 = 262_144;

    struct Lj8 {
        policy: CacheAwarePolicy,
        workers: Vec<Arc<dyn Worker>>,
        load: Arc<crate::lingjing_load::LoadState>,
        state: Arc<Lj8State>,
    }

    fn lj8(cfg: Lj8Config, replicas: usize) -> Lj8 {
        let state = Arc::new(Lj8State::new(Lj8Config { startup_grace_secs: 0, ..cfg }));
        let load = Arc::new(crate::lingjing_load::LoadState::new(crate::lingjing_load::Config {
            avoid_saturated: true,
            ..Default::default()
        }));
        let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
            cache_threshold: 0.3,
            balance_abs_threshold: 8,
            balance_rel_threshold: 1.2,
            eviction_interval_secs: 0,
            max_tree_size: 1 << 24,
        })
        .with_lingjing_load(load.clone())
        .with_lingjing_warmup(state.clone());
        let workers: Vec<Arc<dyn Worker>> = (0..replicas)
            .map(|i| {
                Arc::new(BasicWorker::new(format!("http://lj8-{i}:8000"), WorkerType::Regular))
                    as Arc<dyn Worker>
            })
            .collect();
        policy.init_workers(&workers);
        // Baseline: this count is the fleet, not a scale-out.
        state.on_worker_removed("default", "baseline", replicas, false);
        Lj8 { policy, workers, load, state }
    }

    impl Lj8 {
        /// What `Router::add_worker` does.
        fn scale_out(&mut self) -> usize {
            let url = format!("http://lj8-{}:8000", self.workers.len());
            let warming = self.state.on_worker_adding("default", &url, self.workers.len());
            self.workers
                .push(Arc::new(BasicWorker::new(url.clone(), WorkerType::Regular)));
            self.policy.init_workers(&self.workers);
            if warming {
                self.policy.lingjing_seed_credit(&url, &self.workers);
            }
            self.workers.len() - 1
        }

        fn send(&self, session: &str, bytes: Option<u64>) -> usize {
            self.send_to(&self.workers, session, bytes)
        }

        fn send_to(&self, workers: &[Arc<dyn Worker>], session: &str, bytes: Option<u64>) -> usize {
            let headers: Option<crate::policies::RequestHeaders> =
                bytes.map(|b| [("content-length".to_string(), b.to_string())].into());
            self.policy
                .select_worker_with_headers(workers, Some(session), headers.as_ref())
                .unwrap()
        }

        fn load(&self, idx: usize, requests: usize) {
            for _ in 0..requests {
                self.workers[idx].increment_load();
            }
        }

        fn without(&self, gone: usize) -> Vec<Arc<dyn Worker>> {
            let mut left = self.workers.clone();
            left.remove(gone);
            left
        }

        fn tenants(&self, session: &str) -> usize {
            self.policy.trees.get("default").unwrap().matched_tenants(session).len()
        }
    }

    fn on(warmup: Lj8Mode, keep_hit: Lj8Mode, keep_affinity: Lj8Mode) -> Lj8Config {
        Lj8Config { warmup_mode: warmup, keep_hit, keep_affinity, ..Lj8Config::default() }
    }

    /// A session hash like the gateway's: 16 hex chars sharing no meaningful prefix.
    fn sid(tag: u64, i: u64) -> String {
        format!("{:016x}", (tag << 32 | i).wrapping_mul(0x9e37_79b9_7f4a_7c15).rotate_left(23))
    }

    /// Spread `n` sessions over a fleet, each one small.
    fn seed_sessions(f: &Lj8, n: usize) -> BTreeMap<String, usize> {
        (0..n)
            .map(|i| {
                let session = sid(1, i as u64);
                let placed = f.send(&session, Some(1024));
                (session, placed)
            })
            .collect()
    }

    #[derive(Clone, Default)]
    struct Counts(Arc<Mutex<BTreeMap<String, u64>>>);

    impl metrics::Recorder for Counts {
        fn describe_counter(&self, _: metrics::KeyName, _: Option<metrics::Unit>, _: metrics::SharedString) {}
        fn describe_gauge(&self, _: metrics::KeyName, _: Option<metrics::Unit>, _: metrics::SharedString) {}
        fn describe_histogram(&self, _: metrics::KeyName, _: Option<metrics::Unit>, _: metrics::SharedString) {}

        fn register_counter(&self, key: &metrics::Key, _: &metrics::Metadata<'_>) -> metrics::Counter {
            let label = key.labels().map(|l| l.value().to_string()).collect::<Vec<_>>().join(",");
            metrics::Counter::from_arc(Arc::new(CountHandle(
                self.0.clone(),
                format!("{}:{label}", key.name()),
            )))
        }

        fn register_gauge(&self, _: &metrics::Key, _: &metrics::Metadata<'_>) -> metrics::Gauge {
            metrics::Gauge::noop()
        }

        fn register_histogram(&self, _: &metrics::Key, _: &metrics::Metadata<'_>) -> metrics::Histogram {
            metrics::Histogram::noop()
        }
    }

    struct CountHandle(Arc<Mutex<BTreeMap<String, u64>>>, String);

    impl metrics::CounterFn for CountHandle {
        fn increment(&self, value: u64) {
            *self.0.lock().unwrap().entry(self.1.clone()).or_default() += value;
        }

        fn absolute(&self, _: u64) {}
    }

    fn counted<T>(run: impl FnOnce() -> T) -> (T, BTreeMap<String, u64>) {
        let counts = Counts::default();
        let out = metrics::with_local_recorder(&counts, run);
        let seen = counts.0.lock().unwrap().clone();
        (out, seen)
    }

    fn warmup_count(seen: &BTreeMap<String, u64>, result: &str) -> u64 {
        seen.get(&format!("vllm_router_lingjing_warmup_total:{result}")).copied().unwrap_or(0)
    }

    fn imbalance_count(seen: &BTreeMap<String, u64>, action: &str) -> u64 {
        seen.get(&format!("vllm_router_lingjing_imbalance_total:{action}")).copied().unwrap_or(0)
    }

    // Design §2.3 probes, reversed where the design wants them reversed.

    #[test]
    fn probe_known_sessions_never_move_on_scale_out() {
        let mut f = lj8(on(Lj8Mode::On, Lj8Mode::On, Lj8Mode::On), 2);
        let sessions = seed_sessions(&f, 2000);
        f.scale_out();
        let moved = sessions.iter().filter(|(s, placed)| f.send(s, Some(4 << 20)) != **placed).count();
        assert_eq!(moved, 0, "of 2000 known sessions, even with large bodies");
    }

    #[test]
    fn probe_new_sessions_used_to_all_land_on_the_empty_replica() {
        let mut f = lj8(on(Lj8Mode::Off, Lj8Mode::Off, Lj8Mode::Off), 2);
        seed_sessions(&f, 2000);
        let fresh = f.scale_out();
        let landed = (0..1000).filter(|&i| f.send(&sid(2, i), Some(1024)) == fresh).count();
        // A few random hashes share five characters with a seeded one and count as hits.
        assert!(landed >= 990, "the §2.3 behaviour this patch exists to change: {landed}");
    }

    #[test]
    fn probe_slow_start_shares_new_sessions_instead_of_flooding() {
        let mut f = lj8(on(Lj8Mode::On, Lj8Mode::Off, Lj8Mode::Off), 2);
        seed_sessions(&f, 2000);
        let fresh = f.scale_out();
        let mut landed = [0usize; 3];
        for i in 0..3000 {
            landed[f.send(&sid(2, i), Some(1024))] += 1;
        }
        assert!(
            landed.iter().all(|n| (900..=1100).contains(n)),
            "small new sessions spread roughly evenly: {landed:?}"
        );
        assert!(landed[fresh] > 0);
    }

    #[test]
    fn probe_imbalance_used_to_move_hits_and_keep_hit_stops_it() {
        for (keep, stays) in [(Lj8Mode::Off, false), (Lj8Mode::On, true)] {
            let mut f = lj8(on(Lj8Mode::Off, keep, Lj8Mode::Off), 2);
            let home = f.send("session-a-000000", Some(1024));
            let other = 1 - home;
            let fresh = f.scale_out();
            f.load(home, 9);
            f.load(other, 9);
            let picked = f.send("session-a-000000", Some(1024));
            assert_eq!(picked == home, stays, "{keep:?}");
            assert_eq!(picked == fresh, !stays, "{keep:?}");
        }
    }

    #[test]
    fn slow_start_credit_is_the_mean_of_the_other_trees() {
        // credit = mean(others) - own (design §4.1). The largest tree would starve the new replica
        // until every old one caught up with it; the mean lets it share with the smaller one.
        let mut f = lj8(on(Lj8Mode::On, Lj8Mode::Off, Lj8Mode::Off), 2);
        let tree = f.policy.trees.get("default").unwrap().clone();
        for i in 0..3000 {
            tree.insert(&sid(11, i), f.workers[0].url());
        }
        for i in 0..1000 {
            tree.insert(&sid(12, i), f.workers[1].url());
        }
        let held = tree.get_used_size_per_tenant();
        let (large, small) = (held[f.workers[0].url()], held[f.workers[1].url()]);
        let fresh = f.scale_out();
        assert_eq!(f.state.credit(f.workers[fresh].url()), (large + small) / 2);
        let mut landed = [0usize; 3];
        for i in 0..1500 {
            landed[f.send(&sid(13, i), Some(1024))] += 1;
        }
        assert!(landed[0] <= 15, "only 5-char collisions reach the largest tree: {landed:?}");
        assert!((1100..=1300).contains(&landed[1]), "the smaller tree catches up first: {landed:?}");
        assert!((200..=350).contains(&landed[fresh]), "then the two share: {landed:?}");
    }

    #[test]
    fn slow_start_new_replica_keeps_its_share_after_the_window() {
        // A window's worth of small sessions must not push the replica behind for good.
        let mut f = lj8(
            Lj8Config { warmup_secs: 10, ..on(Lj8Mode::On, Lj8Mode::Off, Lj8Mode::Off) },
            2,
        );
        seed_sessions(&f, 2000);
        let fresh = f.scale_out();
        for i in 0..600 {
            f.send(&sid(2, i), Some(1024));
        }
        f.state.advance(std::time::Duration::from_secs(11));
        assert!(!f.state.is_warming(f.workers[fresh].url()));
        let landed = (0..1200).filter(|&i| f.send(&sid(3, i), Some(1024)) == fresh).count();
        assert!((340..=460).contains(&landed), "about a third after the window: {landed}");
    }

    // ── 0008: session hashes match whole ──

    fn hex_session(i: u64) -> String {
        let mut x = i.wrapping_add(0x9e37_79b9_7f4a_7c15);
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        format!("{:016x}", x ^ (x >> 31))
    }

    /// The share of brand-new sessions that clear the threshold against a tree of `n` hashes.
    fn false_hit_share(n: u64, threshold: f32) -> (f64, f64) {
        let tree = Tree::new();
        let mut cumulative = 0u64;
        let hit = |tree: &Tree, s: &str| {
            let r = tree.prefix_match_with_counts(s);
            r.matched_char_count as f32 / r.input_char_count as f32 > threshold
        };
        for i in 0..n {
            let s = hex_session(i);
            cumulative += u64::from(hit(&tree, &s));
            tree.insert(&s, "http://w:8000");
        }
        let probes = 100_000u64;
        let at_n = (0..probes).filter(|i| hit(&tree, &hex_session(n + i))).count();
        (cumulative as f64 / n as f64, at_n as f64 / probes as f64)
    }

    #[test]
    fn false_hits_are_measured_not_estimated() {
        let (cumulative, at_100k) = false_hit_share(100_000, 0.3);
        println!("false hits: {cumulative:.4} of the first 100k, {at_100k:.4} at 100k");
        // 1 - exp(-100000 / 16^5) = 0.0910 for the next session, about half that on average.
        assert!((0.085..0.097).contains(&at_100k), "{at_100k}");
        assert!((0.042..0.050).contains(&cumulative), "{cumulative}");
    }

    const SETTLED: &str = "abcde00000000001";
    const COLLIDES: &str = "abcde11111111111";

    fn collision_fleet(exact: Lj8Mode) -> Lj8 {
        let f = lj8(Lj8Config { exact_session: exact, ..Lj8Config::default() }, 2);
        // Worker 0 holds the older session and more text, so the new-session path picks worker 1.
        let tree = f.policy.trees.get("default").unwrap().clone();
        tree.insert(SETTLED, f.workers[0].url());
        tree.insert("ffff000000000000", f.workers[0].url());
        f
    }

    #[test]
    fn exact_match_decides_where_a_colliding_new_session_goes() {
        let session_match = |seen: &BTreeMap<String, u64>, result: &str| {
            seen.get(&format!("vllm_router_lingjing_session_match_total:{result}")).copied().unwrap_or(0)
        };
        let f = collision_fleet(Lj8Mode::Off);
        let (landed, seen) = counted(|| f.send(COLLIDES, Some(1024)));
        assert_eq!(landed, 0, "0007: five shared chars are a hit");
        assert_eq!(session_match(&seen, "would_false_hit"), 0);

        let f = collision_fleet(Lj8Mode::Shadow);
        let (landed, seen) = counted(|| f.send(COLLIDES, Some(1024)));
        assert_eq!(landed, 0, "shadow routes like off");
        assert_eq!(session_match(&seen, "would_false_hit"), 1);

        let f = collision_fleet(Lj8Mode::On);
        let (landed, seen) = counted(|| f.send(COLLIDES, Some(1024)));
        assert_eq!(landed, 1, "a new session takes the new-session path");
        assert_eq!(session_match(&seen, "false_hit_avoided"), 1);
        let (again, seen) = counted(|| f.send(COLLIDES, Some(1024)));
        assert_eq!(again, 1, "and its next turn matches whole");
        assert_eq!(session_match(&seen, "exact"), 1);
        let (unseen, seen) = counted(|| f.send("9999000000000000", Some(1024)));
        assert_eq!(unseen, 1);
        assert_eq!(session_match(&seen, "new"), 1, "the denominator for the shadow ratio");
    }

    #[test]
    fn exact_match_takes_fifteen_of_sixteen_as_a_new_session_too() {
        let f = collision_fleet(Lj8Mode::On);
        assert_eq!(f.send("abcde00000000002", Some(1024)), 1);
    }

    #[test]
    fn exact_match_leaves_prefix_texts_on_the_threshold() {
        let f = lj8(Lj8Config { exact_session: Lj8Mode::On, ..Lj8Config::default() }, 2);
        let home = f.send("session-a-000000", Some(1024));
        assert_eq!(f.send("session-a-111111", Some(1024)), home, "16 chars, not a hash");
        let long = "x".repeat(400);
        let home = f.send(&long, Some(1024));
        assert_eq!(f.send(&format!("{long}{}", "y".repeat(300)), Some(1024)), home);
    }

    #[test]
    fn exact_match_applies_while_imbalanced_and_to_the_admission_check() {
        let f = lj8(
            Lj8Config {
                exact_session: Lj8Mode::On,
                keep_hit: Lj8Mode::On,
                ..Lj8Config::default()
            },
            2,
        );
        let policy = &f.policy;
        let tree = policy.trees.get("default").unwrap().clone();
        tree.insert(SETTLED, f.workers[0].url());
        assert!(policy.has_cache_hit(&f.workers, SETTLED));
        assert!(!policy.has_cache_hit(&f.workers, COLLIDES), "a collision is not a known session");
        for _ in 0..12 {
            f.workers[0].increment_load();
        }
        let headers: crate::policies::RequestHeaders =
            [("content-length".to_string(), "1024".to_string())].into();
        let (landed, seen) = counted(|| {
            policy.select_worker_with_headers(&f.workers, Some(COLLIDES), Some(&headers))
        });
        assert_eq!(landed, Some(1), "not kept on worker 0 as a hit");
        assert_eq!(imbalance_count(&seen, "moved_new"), 1, "{seen:?}");
        assert_eq!(
            policy.select_worker_with_headers(&f.workers, Some(SETTLED), Some(&headers)),
            Some(0),
            "a whole match still is one"
        );
    }

    // ── 0009: uncached token load and the time gate ──

    use crate::lingjing_uncached::{Config as Lj9Config, State as Lj9State};

    fn lj9_cfg(mode: Lj8Mode) -> Lj9Config {
        Lj9Config {
            mode,
            peak_tps: 1000.0,
            bytes_per_token: 10.0,
            max_penalty_ms: 2000,
            sat_wait_ms: 5000,
            max_hold_secs: 60,
            peer_interval_ms: 1000,
        }
    }

    /// Two replicas and a settled session on one of them.
    fn lj9(mode: Lj8Mode) -> (CacheAwarePolicy, Vec<Arc<dyn Worker>>, Arc<Lj9State>, usize) {
        let warmup = Arc::new(Lj8State::new(Lj8Config {
            track_sessions: true,
            startup_grace_secs: 0,
            ..Lj8Config::default()
        }));
        let uncached = Arc::new(Lj9State::new(lj9_cfg(mode)));
        let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
            cache_threshold: 0.3,
            balance_abs_threshold: 8,
            balance_rel_threshold: 1.2,
            eviction_interval_secs: 0,
            max_tree_size: 1 << 24,
        })
        .with_lingjing_warmup(warmup)
        .with_lingjing_uncached(uncached.clone());
        let workers: Vec<Arc<dyn Worker>> = (0..2)
            .map(|i| {
                Arc::new(BasicWorker::new(format!("http://lj9-{i}:8000"), WorkerType::Regular))
                    as Arc<dyn Worker>
            })
            .collect();
        policy.init_workers(&workers);
        let home = lj9_send(&policy, &workers, SETTLED, 10_000);
        (policy, workers, uncached, home)
    }

    fn lj9_send(policy: &CacheAwarePolicy, workers: &[Arc<dyn Worker>], s: &str, bytes: u64) -> usize {
        let headers: crate::policies::RequestHeaders =
            [("content-length".to_string(), bytes.to_string())].into();
        policy.select_worker_with_headers(workers, Some(s), Some(&headers)).unwrap()
    }

    #[test]
    fn uncached_new_sessions_rank_by_wait() {
        let (policy, workers, uncached, home) = lj9(Lj8Mode::On);
        let other = 1 - home;
        let _busy = uncached.charge(workers[other].url(), 500);
        assert_eq!(lj9_send(&policy, &workers, "1111000000000000", 1024), home, "the lower wait wins");
    }

    #[test]
    fn uncached_hit_stays_inside_the_penalty_and_counts_its_delta_only() {
        let (policy, workers, uncached, home) = lj9(Lj8Mode::On);
        // Home waits 2s; moving costs the whole 100s body, so the gate must not open.
        let _busy = uncached.charge(workers[home].url(), 2_000);
        assert_eq!(lj9_send(&policy, &workers, SETTLED, 1_000_000), home);
    }

    #[test]
    fn uncached_hit_moves_when_staying_is_slower_by_more_than_the_penalty() {
        let (policy, workers, uncached, home) = lj9(Lj8Mode::On);
        // Home waits 10s, moving costs 1s of prefill: 10s > 1s + 2s.
        let _busy = uncached.charge(workers[home].url(), 10_000);
        let (landed, seen) = counted(|| lj9_send(&policy, &workers, SETTLED, 10_000));
        assert_eq!(landed, 1 - home);
        assert_eq!(seen.get("vllm_router_lingjing_uncached_total:moved_hit"), Some(&1));
    }

    #[test]
    fn uncached_shadow_counts_the_move_and_stays() {
        let (policy, workers, uncached, home) = lj9(Lj8Mode::Shadow);
        let _busy = uncached.charge(workers[home].url(), 10_000);
        let (landed, seen) = counted(|| lj9_send(&policy, &workers, SETTLED, 10_000));
        assert_eq!(landed, home);
        assert_eq!(seen.get("vllm_router_lingjing_uncached_total:would_move_hit"), Some(&1));
    }

    #[test]
    fn lingjing_uncached_shadow_decides_like_0007() {
        let warmup = Arc::new(Lj8State::new(Lj8Config {
            track_sessions: true,
            startup_grace_secs: 0,
            ..Lj8Config::default()
        }));
        let (policy, load) = trace_policy();
        let policy = policy
            .with_lingjing_warmup(warmup)
            .with_lingjing_uncached(Arc::new(Lj9State::new(lj9_cfg(Lj8Mode::Shadow))));
        assert_same_hash_trace_as_0007(lingjing_hash_trace(&policy, &load));
    }

    // ── 0008: route quality ──

    #[test]
    fn route_quality_counts_turns_that_moved() {
        let f = lj8(Lj8Config { route_quality: Lj8Mode::On, ..Lj8Config::default() }, 2);
        let quality = |seen: &BTreeMap<String, u64>, name: &str, result: &str| {
            seen.get(&format!("vllm_router_lingjing_route_quality{name}_total:{result}"))
                .copied()
                .unwrap_or(0)
        };
        let s = "0123456789abcdef";
        let (home, seen) = counted(|| f.send(s, Some(1000)));
        assert_eq!((quality(&seen, "", "first"), quality(&seen, "_bytes", "first")), (1, 1000));
        let (_, seen) = counted(|| f.send(s, Some(1400)));
        assert_eq!((quality(&seen, "", "same"), quality(&seen, "_bytes", "same")), (1, 400), "the new bytes");
        let (_, seen) = counted(|| f.send(s, Some(900)));
        assert_eq!(quality(&seen, "_bytes", "same"), 0, "a shorter body adds nothing");
        // An imbalanced fleet moves the hit: the whole body is cold on the other worker.
        f.load(home, 20);
        let (moved, seen) = counted(|| f.send(s, Some(5000)));
        assert_ne!(moved, home);
        assert_eq!((quality(&seen, "", "moved"), quality(&seen, "_bytes", "moved")), (1, 5000));
    }

    #[test]
    fn route_quality_ignores_prefix_texts_and_stays_quiet_when_off() {
        let f = lj8(Lj8Config { route_quality: Lj8Mode::On, ..Lj8Config::default() }, 2);
        let quiet = |seen: BTreeMap<String, u64>| !seen.keys().any(|k| k.contains("route_quality"));
        let (_, seen) = counted(|| f.send(&"x".repeat(100), Some(1000)));
        assert!(quiet(seen));
        let f = lj8(Lj8Config::default(), 2);
        let (_, seen) = counted(|| f.send("0123456789abcdef", Some(1000)));
        assert!(quiet(seen));
    }

    #[test]
    fn warmup_gate_is_exactly_at_the_threshold() {
        let mut f = lj8(on(Lj8Mode::On, Lj8Mode::Off, Lj8Mode::Off), 2);
        let fresh = f.scale_out();
        f.state.set_credit(f.workers[fresh].url(), 0); // empty trees: fresh wins the tie
        f.load(0, 1);
        f.load(1, 1);
        assert_eq!(f.send(&sid(8, 1), Some(KIB256 - 1)), fresh);
        assert_ne!(f.send(&sid(8, 2), Some(KIB256)), fresh);
    }

    #[test]
    fn warmup_unknown_length_counts_as_large() {
        let mut f = lj8(on(Lj8Mode::On, Lj8Mode::Off, Lj8Mode::Off), 2);
        let fresh = f.scale_out();
        f.state.set_credit(f.workers[fresh].url(), 0);
        f.load(0, 1);
        f.load(1, 1);
        let (_, seen) = counted(|| {
            assert_ne!(f.send(&sid(3, 1), None), fresh);
            let headers: crate::policies::RequestHeaders =
                [("content-length".to_string(), "garbage".to_string())].into();
            let picked = f.policy.select_worker_with_headers(
                &f.workers,
                Some(&sid(3, 2)),
                Some(&headers),
            );
            assert_ne!(picked, Some(fresh));
        });
        assert_eq!(warmup_count(&seen, "no_length"), 2);
        assert_eq!(warmup_count(&seen, "blocked_long"), 2);
    }

    #[test]
    fn warmup_large_sessions_never_reach_the_warming_replica() {
        let mut f = lj8(on(Lj8Mode::On, Lj8Mode::Off, Lj8Mode::Off), 2);
        seed_sessions(&f, 200);
        let fresh = f.scale_out();
        for i in 0..300 {
            let body = KIB256 + i * 4096;
            assert_ne!(f.send(&sid(4, i), Some(body)), fresh);
        }
    }

    #[test]
    fn warmup_opens_only_when_every_other_replica_is_saturated() {
        let mut f = lj8(on(Lj8Mode::On, Lj8Mode::Off, Lj8Mode::Off), 2);
        let fresh = f.scale_out();
        f.load(0, 1);
        f.load(1, 1);
        f.state.set_credit(f.workers[fresh].url(), 0);
        f.load.force(f.workers[0].url(), true);
        assert_eq!(f.send(&sid(9, 1), Some(4 << 20)), 1, "one other replica is free");
        f.workers[1].decrement_load();
        f.load.force(f.workers[1].url(), true);
        let (picked, seen) = counted(|| f.send(&sid(9, 2), Some(4 << 20)));
        assert_eq!(picked, fresh, "scale-out must help once the old replicas are full");
        assert_eq!(warmup_count(&seen, "bypass_saturated"), 1);
    }

    #[test]
    fn warmup_ends_and_the_credit_keeps_preventing_a_flood() {
        let mut f = lj8(
            Lj8Config { warmup_secs: 60, ..on(Lj8Mode::On, Lj8Mode::Off, Lj8Mode::Off) },
            2,
        );
        seed_sessions(&f, 500);
        let fresh = f.scale_out();
        f.state.advance(std::time::Duration::from_secs(61));
        assert!(!f.state.is_warming(f.workers[fresh].url()));
        let mut landed = [0usize; 3];
        for i in 0..600 {
            landed[f.send(&sid(5, i), Some(4 << 20))] += 1;
        }
        assert!(landed[fresh] > 0, "large sessions may enter after the window: {landed:?}");
        assert!(landed[fresh] < 300, "no flood when the window opens: {landed:?}");
    }

    #[test]
    fn warmup_shadow_counts_but_routes_like_off() {
        let mut f = lj8(on(Lj8Mode::Shadow, Lj8Mode::Off, Lj8Mode::Off), 2);
        seed_sessions(&f, 100);
        let fresh = f.scale_out();
        let (picked, seen) = counted(|| {
            (f.send(&sid(6, 1), Some(4 << 20)), f.send(&sid(6, 2), Some(1024)))
        });
        assert_eq!(picked, (fresh, fresh), "shadow keeps the 0007 flood");
        assert_eq!(warmup_count(&seen, "would_block_long"), 1);
        assert_eq!(warmup_count(&seen, "allowed_short"), 1);
        assert_eq!(warmup_count(&seen, "would_shift_credit"), 2);
        assert_eq!(warmup_count(&seen, "blocked_long"), 0);
    }

    #[test]
    fn warmup_hits_keep_going_to_a_warming_replica() {
        let mut f = lj8(on(Lj8Mode::On, Lj8Mode::Off, Lj8Mode::Off), 2);
        let fresh = f.scale_out();
        f.state.set_credit(f.workers[fresh].url(), 0);
        f.load(0, 1);
        f.load(1, 1);
        assert_eq!(f.send("small-then-large", Some(1024)), fresh);
        assert_eq!(f.send("small-then-large", Some(8 << 20)), fresh);
    }

    #[test]
    fn warmup_rolling_restart_is_not_a_scale_out() {
        let mut f = lj8(on(Lj8Mode::On, Lj8Mode::Off, Lj8Mode::Off), 3);
        let gone = f.workers.remove(2);
        f.state.on_worker_removed("default", gone.url(), 3, false);
        f.policy.remove_worker_by_url(gone.url());
        let replacement = f.scale_out();
        assert!(!f.state.is_warming(f.workers[replacement].url()));
        assert_eq!(f.state.credit(f.workers[replacement].url()), 0);
    }

    #[test]
    fn warmup_scale_in_forgets_the_replica() {
        let mut f = lj8(on(Lj8Mode::On, Lj8Mode::On, Lj8Mode::On), 2);
        let fresh = f.scale_out();
        let url = f.workers[fresh].url().to_string();
        f.state.on_worker_removed("default", &url, 3, false);
        assert!(!f.state.is_warming(&url));
        assert_eq!(f.state.credit(&url), 0);
    }

    #[test]
    fn two_routers_reach_the_same_gate_decisions() {
        // Each router has its own tree and clock; the gate depends on neither.
        let mut a = lj8(on(Lj8Mode::On, Lj8Mode::On, Lj8Mode::On), 2);
        let mut b = lj8(on(Lj8Mode::On, Lj8Mode::On, Lj8Mode::On), 2);
        // Sessions are pinned to one router, so their trees differ.
        seed_sessions(&a, 40);
        let fresh_a = a.scale_out();
        b.state.advance(std::time::Duration::from_secs(5));
        let fresh_b = b.scale_out();
        for i in 0..200u64 {
            let session = sid(7, i);
            let body = if i % 3 == 0 { KIB256 + i } else { 2048 };
            let (pa, pb) = (a.send(&session, Some(body)), b.send(&session, Some(body)));
            if body >= KIB256 {
                assert!(pa != fresh_a && pb != fresh_b, "request {i}");
            }
        }
        assert_eq!(a.state.is_warming(a.workers[fresh_a].url()), b.state.is_warming(b.workers[fresh_b].url()));
    }

    #[test]
    fn restarted_router_adopts_the_window_and_the_credit() {
        let mut a = lj8(on(Lj8Mode::On, Lj8Mode::Off, Lj8Mode::Off), 2);
        seed_sessions(&a, 200);
        let fresh = a.scale_out();
        let b = lj8(on(Lj8Mode::On, Lj8Mode::Off, Lj8Mode::Off), 3);
        assert_eq!(b.state.apply_peer(&a.state.snapshot()), 1);
        assert_ne!(b.send("restart-large-01", Some(4 << 20)), fresh);
        assert!(b.state.credit(b.workers[fresh].url()) > 0);
    }

    #[test]
    fn keep_hit_only_on_a_usable_unsaturated_worker() {
        let mut f = lj8(on(Lj8Mode::Off, Lj8Mode::On, Lj8Mode::Off), 2);
        let home = f.send("session-b-000000", Some(1024));
        let fresh = f.scale_out();
        f.load(home, 9);
        let other = 1 - home;
        f.load(other, 9);

        let (picked, seen) = counted(|| f.send("session-b-000000", Some(1024)));
        assert_eq!(picked, home);
        assert_eq!(imbalance_count(&seen, "kept_hit"), 1);

        f.load.force(f.workers[home].url(), true);
        let (picked, seen) = counted(|| f.send("session-b-000000", Some(1024)));
        assert_eq!(picked, fresh, "a saturated target moves as before");
        assert_eq!(imbalance_count(&seen, "moved_hit"), 1);

        let (picked, seen) = counted(|| f.send(&sid(10, 1), Some(1024)));
        assert_eq!(picked, fresh);
        assert_eq!(imbalance_count(&seen, "moved_new"), 1);
    }

    #[test]
    fn keep_hit_shadow_counts_and_still_moves() {
        let mut f = lj8(on(Lj8Mode::Off, Lj8Mode::Shadow, Lj8Mode::Off), 2);
        let home = f.send("session-c-000000", Some(1024));
        let fresh = f.scale_out();
        f.load(0, 9);
        f.load(1, 9);
        let (picked, seen) = counted(|| f.send("session-c-000000", Some(1024)));
        assert_eq!(picked, fresh);
        assert_ne!(picked, home);
        assert_eq!(imbalance_count(&seen, "would_keep_hit"), 1);
        assert_eq!(imbalance_count(&seen, "kept_hit"), 0);
    }

    #[test]
    fn moved_hit_prefers_a_replica_that_is_not_warming() {
        let mut f = lj8(on(Lj8Mode::On, Lj8Mode::On, Lj8Mode::Off), 3);
        let home = f.send("session-d-000000", Some(1024));
        let fresh = f.scale_out();
        let settled = (0..3).find(|&i| i != home).unwrap();
        for i in 0..3 {
            f.load(i, if i == settled { 9 } else { 10 });
        }
        f.load.force(f.workers[home].url(), true);
        assert_eq!(
            f.send("session-d-000000", Some(1024)),
            settled,
            "the least loaded worker is warming, but a small moved hit stays off it too"
        );
        assert_eq!(f.workers[fresh].load(), 0);
    }

    #[test]
    fn evicted_worker_gets_its_sessions_back() {
        let f = lj8(on(Lj8Mode::Off, Lj8Mode::Off, Lj8Mode::On), 3);
        let home = f.send("session-e-000000", Some(1024));
        let left = f.without(home);
        let during = f.send_to(&left, "session-e-000000", Some(1024));
        for _ in 0..10 {
            assert_eq!(f.send_to(&left, "session-e-000000", Some(1024)), during, "no hopping");
        }
        assert!(f.state.is_retained(f.workers[home].url()));
        assert_eq!(f.tenants("session-e-000000"), 2, "the evicted tenant is still in the tree");
        assert_eq!(f.send("session-e-000000", Some(1024)), home, "back on its KV");
    }

    #[test]
    fn eviction_off_and_shadow_delete_the_tenant_like_0007() {
        for mode in [Lj8Mode::Off, Lj8Mode::Shadow] {
            let f = lj8(on(Lj8Mode::Off, Lj8Mode::Off, mode), 3);
            let home = f.send("session-f-000000", Some(1024));
            let left = f.without(home);
            let _ = f.send_to(&left, "session-f-000000", Some(1024));
            assert!(!f.state.is_retained(f.workers[home].url()), "{mode:?}");
            let held = f.policy.trees.get("default").unwrap().get_used_size_per_tenant();
            assert!(!held.contains_key(f.workers[home].url()), "{mode:?}");
        }
    }

    #[test]
    fn eviction_ttl_removes_tenants_of_a_worker_that_never_returns() {
        let f = lj8(on(Lj8Mode::Off, Lj8Mode::Off, Lj8Mode::On), 3);
        let home = f.send("session-g-000000", Some(1024));
        let url = f.workers[home].url().to_string();
        let left = f.without(home);
        let during = f.send_to(&left, "session-g-000000", Some(1024));
        f.state.advance(std::time::Duration::from_secs(600));
        f.state.purge_retained(|_| false, |gone| f.policy.remove_worker_by_url(gone));
        let tree = f.policy.trees.get("default").unwrap().clone();
        assert!(!tree.get_used_size_per_tenant().contains_key(&url));
        assert_eq!(f.send_to(&left, "session-g-000000", Some(1024)), during);
    }

    #[test]
    fn eviction_with_kept_affinity_does_not_bring_large_sessions_to_a_warming_replica() {
        let mut f = lj8(on(Lj8Mode::On, Lj8Mode::Off, Lj8Mode::On), 2);
        let home = f.send("session-h-000000", Some(8 << 20));
        let fresh = f.scale_out();
        let left = f.without(home);
        assert_ne!(f.send_to(&left, "session-h-000000", Some(8 << 20)), left.len());
        let picked = left[f.send_to(&left, "session-h-000000", Some(8 << 20))].url().to_string();
        assert_ne!(picked, f.workers[fresh].url());
    }

    #[test]
    fn stale_tenant_fallback_avoids_a_warming_replica_without_kept_affinity() {
        let mut f = lj8(on(Lj8Mode::On, Lj8Mode::Off, Lj8Mode::Off), 2);
        let home = f.send("session-i-000000", Some(1024));
        let fresh = f.scale_out();
        // Warming worker first in the list: 0007's fallback would pick it.
        let mut list = vec![f.workers[fresh].clone()];
        list.push(f.workers[1 - home].clone());
        assert_eq!(f.send_to(&list, "session-i-000000", Some(8 << 20)), 1);
    }

    #[test]
    fn matched_tenants_are_ordered_by_recency_then_url() {
        let f = lj8(on(Lj8Mode::Off, Lj8Mode::Off, Lj8Mode::Off), 1);
        let tree = f.policy.trees.get("default").unwrap().clone();
        tree.insert("shared session text", "http://b");
        tree.insert("shared session text", "http://a");
        let order: BTreeSet<String> = tree.matched_tenants("shared session text").iter().map(|t| t.to_string()).collect();
        assert_eq!(order.len(), 2);
        assert_eq!(tree.matched_tenants("shared session text")[0].as_ref(), "http://a");
    }

    #[test]
    fn idle_fleet_spreads_new_conversations_by_tree_size() {
        // lingjing patch: with every worker at load 0, unrelated new conversations
        // must not all land on the first worker; follow-up turns keep affinity.
        let config = CacheAwareConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        };
        let policy = CacheAwarePolicy::with_config(config);
        let workers: Vec<Arc<dyn Worker>> = (1..=2)
            .map(|i| {
                Arc::new(BasicWorker::new(format!("http://w{i}:8000"), WorkerType::Regular))
                    as Arc<dyn Worker>
            })
            .collect();
        policy.init_workers(&workers);

        let a = policy.select_worker(&workers, Some("system:alpha conversation one")).unwrap();
        let b = policy.select_worker(&workers, Some("system:beta conversation two")).unwrap();
        assert_ne!(a, b, "second unrelated conversation must go to the emptier worker");
        let a2 = policy
            .select_worker(&workers, Some("system:alpha conversation one\nuser:next turn"))
            .unwrap();
        assert_eq!(a, a2, "follow-up turn keeps its worker");
    }

    #[test]
    fn test_cache_aware_with_balanced_load() {
        // Create policy without eviction thread for testing
        let config = CacheAwareConfig {
            eviction_interval_secs: 0, // Disable eviction thread
            ..Default::default()
        };
        let policy = CacheAwarePolicy::with_config(config);
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(BasicWorker::new(
                "http://w1:8000".to_string(),
                WorkerType::Regular,
            )),
            Arc::new(BasicWorker::new(
                "http://w2:8000".to_string(),
                WorkerType::Regular,
            )),
        ];

        // Initialize the policy with workers
        policy.init_workers(&workers);

        // First request should be distributed
        let idx1 = policy.select_worker(&workers, Some("hello world")).unwrap();

        // Same request should go to same worker (cache hit)
        let idx2 = policy.select_worker(&workers, Some("hello world")).unwrap();
        assert_eq!(idx1, idx2);

        // Similar request should also go to same worker
        let idx3 = policy.select_worker(&workers, Some("hello")).unwrap();
        assert_eq!(idx1, idx3);
    }

    #[test]
    fn test_cache_aware_with_imbalanced_load() {
        let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
            cache_threshold: 0.5,
            balance_abs_threshold: 5,
            balance_rel_threshold: 2.0,
            eviction_interval_secs: 0, // Disable eviction thread
            max_tree_size: 10000,
        });

        let worker1 = BasicWorker::new("http://w1:8000".to_string(), WorkerType::Regular);
        let worker2 = BasicWorker::new("http://w2:8000".to_string(), WorkerType::Regular);

        // Create significant load imbalance
        for _ in 0..20 {
            worker1.increment_load();
        }
        // worker2 has load 0

        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(worker1), Arc::new(worker2)];
        policy.init_workers(&workers);

        // Should select worker2 (lower load) despite cache affinity
        for _ in 0..5 {
            let idx = policy.select_worker(&workers, Some("test")).unwrap();
            assert_eq!(idx, 1); // Should always pick worker2
        }
    }

    #[test]
    fn test_cache_aware_worker_removal() {
        let config = CacheAwareConfig {
            eviction_interval_secs: 0, // Disable eviction thread
            ..Default::default()
        };
        let policy = CacheAwarePolicy::with_config(config);
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(BasicWorker::new(
                "http://w1:8000".to_string(),
                WorkerType::Regular,
            )),
            Arc::new(BasicWorker::new(
                "http://w2:8000".to_string(),
                WorkerType::Regular,
            )),
        ];

        policy.init_workers(&workers);

        // Route some requests
        policy.select_worker(&workers, Some("test1"));
        policy.select_worker(&workers, Some("test2"));

        // Remove a worker
        policy.remove_worker_by_url("http://w1:8000");
        workers[0].set_healthy(false);

        // All requests should now go to worker2
        let idx = policy.select_worker(&workers, Some("test1")).unwrap();
        assert_eq!(idx, 1);
    }

    #[test]
    fn initial_placement_combines_affinity_and_headroom_without_mutation() {
        let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        });
        let candidates = [
            CacheAwareCandidate {
                target_id: "rank-0",
                kv_pressure: 0.2,
            },
            CacheAwareCandidate {
                target_id: "rank-1",
                kv_pressure: 0.1,
            },
        ];
        assert_eq!(
            policy
                .evaluate_initial_placement("model", "shared suffix", &candidates)
                .unwrap()
                .target_id,
            "rank-1"
        );
        policy.commit_placement("model", "shared first", "rank-0");
        let equal_pressure = [
            CacheAwareCandidate {
                target_id: "rank-0",
                kv_pressure: 0.1,
            },
            CacheAwareCandidate {
                target_id: "rank-1",
                kv_pressure: 0.1,
            },
        ];
        assert_eq!(
            policy
                .evaluate_initial_placement("model", "shared second", &equal_pressure)
                .unwrap()
                .target_id,
            "rank-0"
        );
    }
}
