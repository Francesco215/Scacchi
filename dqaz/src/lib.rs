#![allow(dead_code)]

use numpy::PyArrayDyn;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule};
use rand::distributions::WeightedIndex;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, Gamma};
use std::collections::{HashMap, HashSet};
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

type TreeId = u64;
type NodeId = u32;
type RequestId = u64;
type BatchToken = u64;
type Action = u32;

const DUMMY_ALPHA: [f32; 3] = [1.0, 1.0, 1.0];
const TARGET_PAD: i8 = 0;
const TARGET_DIRICHLET: i8 = 1;
const TARGET_CATEGORICAL: i8 = 2;
const OUTCOME_LOSS: i8 = 0;
const OUTCOME_DRAW: i8 = 1;
const OUTCOME_WIN: i8 = 2;
const NO_OUTCOME: i8 = -1;
const NO_DISTANCE: i32 = -1;
const CATEGORICAL_EPSILON: f32 = 1e-6;
const JAX_BACKUP_BLOCK_DEPTH: usize = 32;

#[pyclass(module = "dqaz")]
#[derive(Clone)]
struct SearchConfig {
    #[pyo3(get)]
    action_size: usize,
    #[pyo3(get)]
    observation_shape: Vec<usize>,
    #[pyo3(get)]
    simulations_per_root: u32,
    #[pyo3(get)]
    posterior_best_samples: u32,
    #[pyo3(get)]
    kappa_n: f64,
    #[pyo3(get)]
    seed: u64,
    #[pyo3(get)]
    debug: bool,
    #[pyo3(get)]
    solve_categorical: bool,
}

#[pymethods]
impl SearchConfig {
    #[new]
    #[pyo3(signature = (
        action_size,
        observation_shape,
        simulations_per_root = 64,
        posterior_best_samples = 128,
        kappa_n = 32.0,
        seed = 0,
        debug = false,
        solve_categorical = false
    ))]
    fn new(
        action_size: usize,
        observation_shape: Vec<usize>,
        simulations_per_root: u32,
        posterior_best_samples: u32,
        kappa_n: f64,
        seed: u64,
        debug: bool,
        solve_categorical: bool,
    ) -> PyResult<Self> {
        if action_size == 0 {
            return Err(PyValueError::new_err("action_size must be positive"));
        }
        if action_size > Action::MAX as usize {
            return Err(PyValueError::new_err("action_size must fit in u32"));
        }
        if observation_shape.is_empty() || observation_shape.iter().any(|dim| *dim == 0) {
            return Err(PyValueError::new_err(
                "observation_shape must contain positive dimensions",
            ));
        }
        if simulations_per_root == 0 {
            return Err(PyValueError::new_err(
                "simulations_per_root must be positive",
            ));
        }
        if posterior_best_samples == 0 {
            return Err(PyValueError::new_err(
                "posterior_best_samples must be positive",
            ));
        }
        if !kappa_n.is_finite() || kappa_n < 0.0 {
            return Err(PyValueError::new_err(
                "kappa_n must be a finite nonnegative value",
            ));
        }

        Ok(Self {
            action_size,
            observation_shape,
            simulations_per_root,
            posterior_best_samples,
            kappa_n,
            seed,
            debug,
            solve_categorical,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "SearchConfig(action_size={}, observation_shape={:?}, \
             simulations_per_root={}, posterior_best_samples={}, \
             kappa_n={}, seed={}, debug={}, solve_categorical={})",
            self.action_size,
            self.observation_shape,
            self.simulations_per_root,
            self.posterior_best_samples,
            self.kappa_n,
            self.seed,
            self.debug,
            self.solve_categorical,
        )
    }
}

impl SearchConfig {
    fn observation_len(&self) -> usize {
        self.observation_shape.iter().product()
    }
}

struct Node {
    id: NodeId,
    generation: u32,
    parent: Option<NodeId>,
    parent_link: Option<ParentLink>,
    depth: u32,
    state: PyObject,
    current_player: i32,
    kind: NodeKind,
    c_v: Option<[f32; 3]>,
    cached_pi: Option<Vec<f32>>,
    n_down: u32,
    cache_version: u32,
    cat_outcome: i8,
    cat_distance: i32,
    cat_action: Option<Action>,
}

enum NodeKind {
    Decision(DecisionData),
    Terminal(TerminalData),
}

#[derive(Clone, Copy)]
enum ParentLink {
    DecisionAction { action: Action },
}

#[derive(Debug)]
struct DecisionData {
    observation: Vec<f32>,
    legal_actions: Vec<Action>,
    policy_logits: Vec<f32>,
    value_alpha: [f32; 3],
    q_alpha: Vec<[f32; 3]>,
    edges: Vec<DecisionEdge>,
}

#[derive(Default)]
struct SubmitProfile {
    enabled: bool,
    active_rows: usize,
    padded_rows: usize,
    parsed_rows: usize,
    rows_seen: usize,
    terminal_rows: usize,
    new_decision_nodes: usize,
    existing_decision_nodes: usize,
    skipped_rows: usize,
    total_legal_actions: usize,
    backup_path_steps: usize,
    recompute_calls: usize,
    posterior_policy_calls: usize,
    posterior_policy_action_visits: usize,
    total: Duration,
    parse: Duration,
    loop_total: Duration,
    batch_item: Duration,
    decision_data: Duration,
    decision_key: Duration,
    decision_lookup: Duration,
    node_insert: Duration,
    parent_update: Duration,
    backup: Duration,
    publish_child: Duration,
    align: Duration,
    edge_write: Duration,
    recompute: Duration,
    recompute_categorize: Duration,
    posterior_policy: Duration,
    propagate: Duration,
}

impl SubmitProfile {
    fn new(active_rows: usize, padded_rows: usize) -> Self {
        Self {
            enabled: std::env::var_os("DQAZ_PROFILE_SUBMIT").is_some(),
            active_rows,
            padded_rows,
            ..Self::default()
        }
    }

    fn start(&self) -> Option<Instant> {
        self.enabled.then(Instant::now)
    }

    fn print(&self) {
        if !self.enabled {
            return;
        }
        let accounted = self.parse
            + self.batch_item
            + self.decision_data
            + self.decision_key
            + self.decision_lookup
            + self.node_insert
            + self.parent_update
            + self.backup;
        eprintln!(
            concat!(
                "DQAZ_PROFILE_SUBMIT",
                " active_rows={} padded_rows={} parsed_rows={} rows_seen={} skipped_rows={}",
                " terminal_rows={} new_decision_nodes={} existing_decision_nodes={}",
                " total_legal_actions={} backup_path_steps={} recompute_calls={}",
                " posterior_policy_calls={} posterior_policy_action_visits={}",
                " total_ms={:.3} parse_ms={:.3} loop_ms={:.3}",
                " batch_item_ms={:.3} decision_data_ms={:.3}",
                " decision_key_ms={:.3} decision_lookup_ms={:.3}",
                " node_insert_ms={:.3} parent_update_ms={:.3} backup_ms={:.3}",
                " publish_child_ms={:.3} align_ms={:.3} edge_write_ms={:.3}",
                " recompute_ms={:.3} recompute_categorize_ms={:.3}",
                " posterior_policy_ms={:.3} propagate_ms={:.3}",
                " accounted_ms={:.3}"
            ),
            self.active_rows,
            self.padded_rows,
            self.parsed_rows,
            self.rows_seen,
            self.skipped_rows,
            self.terminal_rows,
            self.new_decision_nodes,
            self.existing_decision_nodes,
            self.total_legal_actions,
            self.backup_path_steps,
            self.recompute_calls,
            self.posterior_policy_calls,
            self.posterior_policy_action_visits,
            duration_ms(self.total),
            duration_ms(self.parse),
            duration_ms(self.loop_total),
            duration_ms(self.batch_item),
            duration_ms(self.decision_data),
            duration_ms(self.decision_key),
            duration_ms(self.decision_lookup),
            duration_ms(self.node_insert),
            duration_ms(self.parent_update),
            duration_ms(self.backup),
            duration_ms(self.publish_child),
            duration_ms(self.align),
            duration_ms(self.edge_write),
            duration_ms(self.recompute),
            duration_ms(self.recompute_categorize),
            duration_ms(self.posterior_policy),
            duration_ms(self.propagate),
            duration_ms(accounted),
        );
    }
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

impl DecisionData {
    fn new(
        action_size: usize,
        observation: Vec<f32>,
        legal_actions: Vec<Action>,
        policy_logits: Vec<f32>,
        value_alpha: [f32; 3],
        q_alpha: Vec<[f32; 3]>,
    ) -> PyResult<Self> {
        let valid_count = legal_actions.len();
        if valid_count == 0 {
            return Err(PyValueError::new_err(
                "decision nodes must have at least one legal action",
            ));
        }
        if policy_logits.len() != valid_count {
            return Err(PyValueError::new_err(
                "policy_logits length must match legal_actions length",
            ));
        }
        if q_alpha.len() != valid_count {
            return Err(PyValueError::new_err(
                "q_alpha length must match legal_actions length",
            ));
        }
        if value_alpha.iter().any(|alpha| !alpha.is_finite() || *alpha <= 0.0) {
            return Err(PyValueError::new_err("value_alpha must be strictly positive"));
        }
        if q_alpha
            .iter()
            .flatten()
            .any(|alpha| !alpha.is_finite() || *alpha <= 0.0)
        {
            return Err(PyValueError::new_err("q_alpha must be strictly positive"));
        }

        let mut seen = HashSet::with_capacity(valid_count);
        for action in &legal_actions {
            if (*action as usize) >= action_size {
                return Err(PyValueError::new_err("legal action outside action_size"));
            }
            if !seen.insert(*action) {
                return Err(PyValueError::new_err("duplicate legal action"));
            }
        }

        Ok(Self {
            observation,
            legal_actions,
            policy_logits,
            value_alpha,
            q_alpha,
            edges: (0..valid_count).map(|_| DecisionEdge::new()).collect(),
        })
    }

    fn edge_index_for_action(&self, action: Action) -> Option<usize> {
        self.legal_actions
            .iter()
            .position(|candidate| *candidate == action)
    }
}

#[derive(Debug)]
struct DecisionEdge {
    child: Option<NodeId>,
    completed: bool,
    pending: bool,
    b: [f32; 3],
    r_count: u32,
    child_cache_version: Option<u32>,
    cat_outcome: i8,
    cat_distance: i32,
}

impl DecisionEdge {
    fn new() -> Self {
        Self {
            child: None,
            completed: false,
            pending: false,
            b: DUMMY_ALPHA,
            r_count: 0,
            child_cache_version: None,
            cat_outcome: NO_OUTCOME,
            cat_distance: NO_DISTANCE,
        }
    }
}

struct TerminalData {
    alpha: [f32; 3],
    outcome: i8,
}

#[derive(Clone)]
struct PathStep {
    node_id: NodeId,
    edge_index: usize,
    action: Action,
}

#[derive(Clone)]
struct RequestRecord {
    request_id: RequestId,
    tree_id: TreeId,
    tree_generation: u32,
    node_id: NodeId,
    node_generation: u32,
    action: Action,
    path: Vec<PathStep>,
}

struct PendingBatch {
    records: Vec<RequestRecord>,
    padded_size: usize,
}

struct PreparedJaxBackup {
    tree_id: TreeId,
    path: Vec<PathStep>,
    leaf_alpha: [f32; 3],
    leaf_player: i32,
    categorical_found: bool,
}

#[derive(Clone)]
struct CategoricalTouch {
    tree_id: TreeId,
    path: Vec<PathStep>,
}

struct Tree {
    id: TreeId,
    generation: u32,
    nodes: Vec<Node>,
    root: NodeId,
    pending_requests: HashSet<RequestId>,
    rng: ChaCha20Rng,
    decision_table: HashMap<DecisionKey, NodeId>,
}

#[derive(Clone, Eq, Hash, PartialEq)]
struct DecisionKey {
    current_player: i32,
    observation_bits: Vec<u32>,
}

impl Tree {
    fn node(&self, node_id: NodeId) -> PyResult<&Node> {
        self.nodes
            .get(node_id as usize)
            .ok_or_else(|| PyRuntimeError::new_err("node id out of range"))
    }

    fn node_mut(&mut self, node_id: NodeId) -> PyResult<&mut Node> {
        self.nodes
            .get_mut(node_id as usize)
            .ok_or_else(|| PyRuntimeError::new_err("node id out of range"))
    }

    fn is_done(&self, config: &SearchConfig) -> bool {
        if !self.pending_requests.is_empty() {
            return false;
        }
        let Ok(root) = self.node(self.root) else {
            return true;
        };
        match root.kind {
            NodeKind::Decision(_) => {
                root.cat_outcome != NO_OUTCOME || root.n_down >= config.simulations_per_root
            }
            NodeKind::Terminal(_) => true,
        }
    }
}

struct Forest {
    config: SearchConfig,
    trees: Vec<Option<Tree>>,
    free_tree_slots: Vec<usize>,
    next_tree_id: TreeId,
    next_request_id: RequestId,
    next_batch_token: BatchToken,
    request_table: HashMap<BatchToken, PendingBatch>,
    pending_categorical_touches: Vec<CategoricalTouch>,
    round_robin_cursor: usize,
}

impl Forest {
    fn new(config: SearchConfig) -> Self {
        Self {
            config,
            trees: Vec::new(),
            free_tree_slots: Vec::new(),
            next_tree_id: 1,
            next_request_id: 1,
            next_batch_token: 1,
            request_table: HashMap::new(),
            pending_categorical_touches: Vec::new(),
            round_robin_cursor: 0,
        }
    }

    fn insert_tree(&mut self, tree: Tree) {
        if let Some(slot) = self.free_tree_slots.pop() {
            self.trees[slot] = Some(tree);
        } else {
            self.trees.push(Some(tree));
        }
    }

    fn tree_index(&self, tree_id: TreeId) -> PyResult<usize> {
        self.trees
            .iter()
            .position(|tree| tree.as_ref().is_some_and(|tree| tree.id == tree_id))
            .ok_or_else(|| PyValueError::new_err(format!("unknown tree id {tree_id}")))
    }

    fn tree(&self, tree_id: TreeId) -> PyResult<&Tree> {
        let index = self.tree_index(tree_id)?;
        Ok(self.trees[index].as_ref().expect("tree slot must be occupied"))
    }

    fn tree_mut(&mut self, tree_id: TreeId) -> PyResult<&mut Tree> {
        let index = self.tree_index(tree_id)?;
        Ok(self.trees[index].as_mut().expect("tree slot must be occupied"))
    }

    fn active_tree_ids(&self) -> Vec<TreeId> {
        self.trees
            .iter()
            .filter_map(|tree| tree.as_ref().map(|tree| tree.id))
            .collect()
    }
}

#[pyclass(module = "dqaz")]
struct TransitionBatch {
    #[pyo3(get)]
    token: BatchToken,
    #[pyo3(get)]
    size: usize,
    #[pyo3(get)]
    padded_size: usize,
    #[pyo3(get)]
    parent_states: PyObject,
    #[pyo3(get)]
    actions: PyObject,
    #[pyo3(get)]
    active_mask: PyObject,
    #[pyo3(get)]
    tree_ids: PyObject,
    #[pyo3(get)]
    parent_node_ids: PyObject,
    #[pyo3(get)]
    request_ids: PyObject,
    #[pyo3(get)]
    tree_generations: PyObject,
}

#[pymethods]
impl TransitionBatch {
    fn __repr__(&self) -> String {
        format!(
            "TransitionBatch(token={}, size={}, padded_size={})",
            self.token, self.size, self.padded_size
        )
    }
}

#[pyclass(module = "dqaz")]
struct SearchResults {
    #[pyo3(get)]
    tree_ids: PyObject,
    #[pyo3(get)]
    actions: PyObject,
    #[pyo3(get)]
    action_offsets: PyObject,
    #[pyo3(get)]
    legal_actions: PyObject,
    #[pyo3(get)]
    pi_search: PyObject,
    #[pyo3(get)]
    root_alpha: PyObject,
    #[pyo3(get)]
    root_q_mean: PyObject,
    #[pyo3(get)]
    beta_v: PyObject,
    #[pyo3(get)]
    q_target_kind: PyObject,
    #[pyo3(get)]
    q_target_weight: PyObject,
    #[pyo3(get)]
    q_target_outcome: PyObject,
    #[pyo3(get)]
    q_target_distance: PyObject,
    #[pyo3(get)]
    v_target_kind: PyObject,
    #[pyo3(get)]
    v_target_weight: PyObject,
    #[pyo3(get)]
    v_target_outcome: PyObject,
    #[pyo3(get)]
    v_target_distance: PyObject,
}

#[pyclass(module = "dqaz")]
struct SearchTargets {
    #[pyo3(get)]
    observations: PyObject,
    #[pyo3(get)]
    action_offsets: PyObject,
    #[pyo3(get)]
    legal_actions: PyObject,
    #[pyo3(get)]
    policy_target: PyObject,
    #[pyo3(get)]
    q_target_alpha: PyObject,
    #[pyo3(get)]
    q_loss_weight: PyObject,
    #[pyo3(get)]
    v_target_alpha: PyObject,
    #[pyo3(get)]
    q_target_kind: PyObject,
    #[pyo3(get)]
    q_target_weight: PyObject,
    #[pyo3(get)]
    q_target_outcome: PyObject,
    #[pyo3(get)]
    q_target_distance: PyObject,
    #[pyo3(get)]
    v_target_kind: PyObject,
    #[pyo3(get)]
    v_target_weight: PyObject,
    #[pyo3(get)]
    v_target_outcome: PyObject,
    #[pyo3(get)]
    v_target_distance: PyObject,
    #[pyo3(get)]
    row_mask: PyObject,
    #[pyo3(get)]
    tree_ids: PyObject,
    #[pyo3(get)]
    node_ids: PyObject,
    #[pyo3(get)]
    depths: PyObject,
}

#[pyclass(module = "dqaz")]
struct JaxBackupBatch {
    #[pyo3(get)]
    used_jax: bool,
    #[pyo3(get)]
    node_count: usize,
    #[pyo3(get)]
    path_count: usize,
    #[pyo3(get)]
    max_depth: usize,
    #[pyo3(get)]
    path_depth: usize,
    #[pyo3(get)]
    tree_ids: PyObject,
    #[pyo3(get)]
    node_ids: PyObject,
    #[pyo3(get)]
    edge_b: PyObject,
    #[pyo3(get)]
    edge_completed: PyObject,
    #[pyo3(get)]
    edge_r_count: PyObject,
    #[pyo3(get)]
    q_alpha: PyObject,
    #[pyo3(get)]
    value_alpha: PyObject,
    #[pyo3(get)]
    legal_mask: PyObject,
    #[pyo3(get)]
    node_players: PyObject,
    #[pyo3(get)]
    path_nodes: PyObject,
    #[pyo3(get)]
    path_edges: PyObject,
    #[pyo3(get)]
    path_mask: PyObject,
    #[pyo3(get)]
    leaf_alpha: PyObject,
    #[pyo3(get)]
    leaf_players: PyObject,
}

#[pyclass(module = "dqaz")]
struct SearchEngine {
    inner: Mutex<Forest>,
}

#[pymethods]
impl SearchEngine {
    #[new]
    fn new(config: SearchConfig) -> Self {
        Self {
            inner: Mutex::new(Forest::new(config)),
        }
    }

    #[pyo3(signature = (
        root_states,
        observations,
        action_offsets,
        legal_actions,
        current_players,
        policy_logits,
        value_alpha,
        q_alpha,
        terminated=None,
        terminal_alpha=None
    ))]
    fn add_roots(
        &self,
        py: Python<'_>,
        root_states: &PyAny,
        observations: &PyAny,
        action_offsets: &PyAny,
        legal_actions: &PyAny,
        current_players: &PyAny,
        policy_logits: &PyAny,
        value_alpha: &PyAny,
        q_alpha: &PyAny,
        terminated: Option<&PyAny>,
        terminal_alpha: Option<&PyAny>,
    ) -> PyResult<PyObject> {
        let mut inner = self.lock()?;
        let batch = ParsedNodeBatch::parse(
            py,
            &inner.config,
            observations,
            action_offsets,
            legal_actions,
            current_players,
            policy_logits,
            value_alpha,
        q_alpha,
        terminated,
        terminal_alpha,
        None,
    )?;

        let mut tree_ids = Vec::with_capacity(batch.len);
        for row in 0..batch.len {
            let state = batch_item(py, root_states, row)?;
            let current_player = batch.current_players[row];
            let node = if batch.terminated[row] {
                let alpha = batch.terminal_alpha[row].ok_or_else(|| {
                    PyValueError::new_err("terminal row missing positive terminal_alpha")
                })?;
                terminal_node(0, state, current_player, alpha)
            } else {
                let decision = batch.decision_data_for_row(&inner.config, row)?;
                decision_node(0, state, current_player, decision)
            };

            let id = inner.next_tree_id;
            inner.next_tree_id += 1;
            let rng = ChaCha20Rng::seed_from_u64(inner.config.seed ^ id.rotate_left(17));
            let mut decision_table = HashMap::new();
            if let NodeKind::Decision(data) = &node.kind {
                decision_table.insert(decision_key(current_player, &data.observation), 0);
            }
            let tree = Tree {
                id,
                generation: 0,
                nodes: vec![node],
                root: 0,
                pending_requests: HashSet::new(),
                rng,
                decision_table,
            };
            inner.insert_tree(tree);
            tree_ids.push(id);
        }

        np_array(py, tree_ids, "uint64")
    }

    #[pyo3(signature = (max_batch_size, pad_to=None))]
    fn request_transitions(
        &self,
        py: Python<'_>,
        max_batch_size: usize,
        pad_to: Option<usize>,
    ) -> PyResult<Py<TransitionBatch>> {
        if max_batch_size == 0 {
            return Err(PyValueError::new_err("max_batch_size must be positive"));
        }

        let mut inner = self.lock()?;
        let batch = request_transitions(py, &mut inner, max_batch_size, pad_to)?;
        Py::new(py, batch)
    }

    #[pyo3(signature = (
        token,
        child_states,
        observations,
        action_offsets,
        legal_actions,
        current_players,
        terminated,
        terminal_alpha,
        policy_logits,
        value_alpha,
        q_alpha
    ))]
    fn submit_transitions(
        &self,
        py: Python<'_>,
        token: BatchToken,
        child_states: &PyAny,
        observations: &PyAny,
        action_offsets: &PyAny,
        legal_actions: &PyAny,
        current_players: &PyAny,
        terminated: &PyAny,
        terminal_alpha: &PyAny,
        policy_logits: &PyAny,
        value_alpha: &PyAny,
        q_alpha: &PyAny,
    ) -> PyResult<()> {
        let total_start = Instant::now();
        let mut inner = self.lock()?;
        let pending = inner
            .request_table
            .remove(&token)
            .ok_or_else(|| PyValueError::new_err("wrong transition batch token"))?;
        let mut profile = SubmitProfile::new(pending.records.len(), pending.padded_size);

        let parse_start = profile.start();
        let batch = ParsedNodeBatch::parse(
            py,
            &inner.config,
            observations,
            action_offsets,
            legal_actions,
            current_players,
            policy_logits,
            value_alpha,
            q_alpha,
            Some(terminated),
            Some(terminal_alpha),
            Some(pending.records.len()),
        )?;
        if let Some(start) = parse_start {
            profile.parse = start.elapsed();
        }
        profile.parsed_rows = batch.len;
        if batch.len != pending.padded_size {
            return Err(PyValueError::new_err("transition output shape mismatch"));
        }

        let loop_start = profile.start();
        for (row, record) in pending.records.iter().enumerate() {
            submit_one_transition(
                py,
                &mut inner,
                record,
                &batch,
                child_states,
                row,
                &mut profile,
            )?;
        }
        if let Some(start) = loop_start {
            profile.loop_total = start.elapsed();
        }
        if profile.enabled {
            profile.total = total_start.elapsed();
            profile.print();
        }
        Ok(())
    }

    #[pyo3(signature = (
        token,
        child_states,
        observations,
        action_offsets,
        legal_actions,
        current_players,
        terminated,
        terminal_alpha,
        policy_logits,
        value_alpha,
        q_alpha
    ))]
    fn submit_transitions_jax_prepare(
        &self,
        py: Python<'_>,
        token: BatchToken,
        child_states: &PyAny,
        observations: &PyAny,
        action_offsets: &PyAny,
        legal_actions: &PyAny,
        current_players: &PyAny,
        terminated: &PyAny,
        terminal_alpha: &PyAny,
        policy_logits: &PyAny,
        value_alpha: &PyAny,
        q_alpha: &PyAny,
    ) -> PyResult<Py<JaxBackupBatch>> {
        let mut inner = self.lock()?;
        let pending = inner
            .request_table
            .remove(&token)
            .ok_or_else(|| PyValueError::new_err("wrong transition batch token"))?;
        let mut profile = SubmitProfile::new(pending.records.len(), pending.padded_size);

        let batch = ParsedNodeBatch::parse(
            py,
            &inner.config,
            observations,
            action_offsets,
            legal_actions,
            current_players,
            policy_logits,
            value_alpha,
            q_alpha,
            Some(terminated),
            Some(terminal_alpha),
            Some(pending.records.len()),
        )?;
        profile.parsed_rows = batch.len;
        if batch.len != pending.padded_size {
            return Err(PyValueError::new_err("transition output shape mismatch"));
        }

        let mut prepared = Vec::with_capacity(pending.records.len());
        let mut categorical_touches = Vec::new();
        for (row, record) in pending.records.iter().enumerate() {
            if let Some(item) = submit_one_transition_prepare_jax(
                py,
                &mut inner,
                record,
                &batch,
                child_states,
                row,
                &mut profile,
            )? {
                if item.categorical_found {
                    categorical_touches.push(CategoricalTouch {
                        tree_id: item.tree_id,
                        path: item.path.clone(),
                    });
                }
                prepared.push(item);
            }
        }

        let batch = build_jax_backup_batch(py, &inner, &prepared, pending.padded_size)?;
        inner
            .pending_categorical_touches
            .extend(categorical_touches);
        Ok(batch)
    }

    #[pyo3(signature = (
        tree_ids,
        node_ids,
        edge_b,
        edge_completed,
        edge_r_count,
        c_v,
        n_down,
        policy,
        node_count=None
    ))]
    fn apply_jax_backup(
        &self,
        py: Python<'_>,
        tree_ids: &PyAny,
        node_ids: &PyAny,
        edge_b: &PyAny,
        edge_completed: &PyAny,
        edge_r_count: &PyAny,
        c_v: &PyAny,
        n_down: &PyAny,
        policy: &PyAny,
        node_count: Option<usize>,
    ) -> PyResult<()> {
        let tree_ids = array_flat_u64(py, tree_ids)?;
        let node_ids = array_flat_i64(py, node_ids)?;
        let edge_b = array_flat_f32(py, edge_b)?;
        let edge_completed = array_flat_bool(py, edge_completed)?;
        let edge_r_count = array_flat_i64(py, edge_r_count)?;
        let c_v = array_flat_f32(py, c_v)?;
        let n_down = array_flat_i64(py, n_down)?;
        let policy = array_flat_f32(py, policy)?;

        let mut inner = self.lock()?;
        let node_count = node_count.unwrap_or(tree_ids.len());
        apply_jax_backup_result(
            &mut inner,
            &tree_ids,
            &node_ids,
            &edge_b,
            &edge_completed,
            &edge_r_count,
            &c_v,
            &n_down,
            &policy,
            node_count,
        )?;
        apply_pending_categorical_touches(&mut inner)
    }

    #[pyo3(signature = (tree_ids=None))]
    fn is_done(&self, tree_ids: Option<&PyAny>) -> PyResult<bool> {
        let inner = self.lock()?;
        let ids = selected_tree_ids(&inner, tree_ids)?;
        Ok(ids
            .iter()
            .all(|tree_id| inner.tree(*tree_id).is_ok_and(|tree| tree.is_done(&inner.config))))
    }

    fn stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        let inner = self.lock()?;
        let stats = PyDict::new(py);
        let active = inner.trees.iter().filter(|tree| tree.is_some()).count();
        let pending = inner.request_table.len();
        stats.set_item("trees", active)?;
        stats.set_item("pending_batches", pending)?;
        stats.set_item("next_tree_id", inner.next_tree_id)?;
        Ok(stats.into_py(py))
    }

    #[pyo3(signature = (tree_ids=None, commit="posterior_sample"))]
    fn finish(
        &self,
        py: Python<'_>,
        tree_ids: Option<&PyAny>,
        commit: &str,
    ) -> PyResult<Py<SearchResults>> {
        validate_commit(commit)?;
        let mut inner = self.lock()?;
        let ids = selected_tree_ids(&inner, tree_ids)?;
        let results = finish_trees(py, &mut inner, &ids, commit)?;
        Py::new(py, results)
    }

    #[pyo3(signature = (tree_ids=None))]
    fn export_targets(
        &self,
        py: Python<'_>,
        tree_ids: Option<&PyAny>,
    ) -> PyResult<Py<SearchTargets>> {
        let inner = self.lock()?;
        let ids = selected_tree_ids(&inner, tree_ids)?;
        let targets = export_targets(py, &inner, &ids)?;
        Py::new(py, targets)
    }

    fn advance_roots(&self, tree_ids: &PyAny, actions: &PyAny) -> PyResult<()> {
        let ids = array_flat_u64(tree_ids.py(), tree_ids)?;
        let actions = array_flat_i64(actions.py(), actions)?;
        if ids.len() != actions.len() {
            return Err(PyValueError::new_err("tree_ids and actions length mismatch"));
        }

        let mut inner = self.lock()?;
        for (tree_id, action) in ids.iter().zip(actions.iter()) {
            if *action < 0 || (*action as usize) >= inner.config.action_size {
                return Err(PyValueError::new_err("invalid legal action in advance_roots"));
            }
            advance_one_root(&mut inner, *tree_id, *action as Action)?;
        }
        Ok(())
    }

    #[pyo3(signature = (tree_ids=None))]
    fn clear(&self, tree_ids: Option<&PyAny>) -> PyResult<()> {
        let mut inner = self.lock()?;
        let ids = selected_tree_ids(&inner, tree_ids)?;
        let id_set = ids.iter().copied().collect::<HashSet<_>>();
        let mut canceled = Vec::new();
        inner.request_table.retain(|_, batch| {
            if batch.records.iter().any(|record| id_set.contains(&record.tree_id)) {
                canceled.extend(batch.records.iter().map(|record| {
                    (record.tree_id, record.request_id)
                }));
                false
            } else {
                true
            }
        });
        for (tree_id, request_id) in canceled {
            if let Ok(tree) = inner.tree_mut(tree_id) {
                tree.pending_requests.remove(&request_id);
            }
        }
        inner
            .pending_categorical_touches
            .retain(|touch| !id_set.contains(&touch.tree_id));
        for tree_id in ids {
            let index = inner.tree_index(tree_id)?;
            inner.trees[index] = None;
            inner.free_tree_slots.push(index);
        }
        Ok(())
    }

    fn clear_all(&self) -> PyResult<()> {
        let mut inner = self.lock()?;
        inner.trees.clear();
        inner.free_tree_slots.clear();
        inner.request_table.clear();
        inner.pending_categorical_touches.clear();
        inner.round_robin_cursor = 0;
        Ok(())
    }
}

impl SearchEngine {
    fn lock(&self) -> PyResult<MutexGuard<'_, Forest>> {
        self.inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("SearchEngine mutex is poisoned"))
    }
}

struct ParsedNodeBatch {
    len: usize,
    observations: Vec<f32>,
    action_offsets: Vec<usize>,
    legal_actions: Vec<Action>,
    current_players: Vec<i32>,
    policy_logits: Vec<f32>,
    value_alpha: Vec<[f32; 3]>,
    q_alpha: Vec<[f32; 3]>,
    terminated: Vec<bool>,
    terminal_alpha: Vec<Option<[f32; 3]>>,
}

impl ParsedNodeBatch {
    #[allow(clippy::too_many_arguments)]
    fn parse(
        py: Python<'_>,
        config: &SearchConfig,
        observations: &PyAny,
        action_offsets: &PyAny,
        legal_actions: &PyAny,
        current_players: &PyAny,
        policy_logits: &PyAny,
        value_alpha: &PyAny,
        q_alpha: &PyAny,
        terminated: Option<&PyAny>,
        terminal_alpha: Option<&PyAny>,
        active_rows: Option<usize>,
    ) -> PyResult<Self> {
        let current_players = array_flat_i64(py, current_players)?
            .into_iter()
            .map(|value| value as i32)
            .collect::<Vec<_>>();
        let len = current_players.len();
        let obs_len = config.observation_len();
        let observations = array_flat_f32(py, observations)?;
        if observations.len() != len * obs_len {
            return Err(PyValueError::new_err("transition output shape mismatch"));
        }

        let raw_offsets = array_flat_i64(py, action_offsets)?;
        if raw_offsets.len() != len + 1 {
            return Err(PyValueError::new_err(
                "invalid action_offsets shape or monotonicity",
            ));
        }
        if raw_offsets.first() != Some(&0) || raw_offsets.iter().any(|offset| *offset < 0) {
            return Err(PyValueError::new_err(
                "invalid action_offsets shape or monotonicity",
            ));
        }
        let mut action_offsets = Vec::with_capacity(raw_offsets.len());
        let mut last = 0usize;
        for raw in raw_offsets {
            let offset = raw as usize;
            if offset < last {
                return Err(PyValueError::new_err(
                    "invalid action_offsets shape or monotonicity",
                ));
            }
            action_offsets.push(offset);
            last = offset;
        }
        let total_actions = *action_offsets.last().unwrap_or(&0);

        let legal_actions = array_flat_i64(py, legal_actions)?
            .into_iter()
            .map(|action| {
                if action < 0 || action as usize >= config.action_size {
                    Err(PyValueError::new_err("legal action outside 0..action_size"))
                } else {
                    Ok(action as Action)
                }
            })
            .collect::<PyResult<Vec<_>>>()?;
        if legal_actions.len() != total_actions {
            return Err(PyValueError::new_err(
                "legal_actions length mismatch with action_offsets",
            ));
        }

        let policy_logits = array_flat_f32(py, policy_logits)?;
        if policy_logits.len() != total_actions {
            return Err(PyValueError::new_err(
                "policy_logits length mismatch with legal_actions",
            ));
        }

        let q_flat = array_flat_f32(py, q_alpha)?;
        if q_flat.len() != total_actions * 3 {
            return Err(PyValueError::new_err(
                "q_alpha length mismatch with legal_actions",
            ));
        }
        let q_alpha = q_flat
            .chunks_exact(3)
            .map(|chunk| [chunk[0], chunk[1], chunk[2]])
            .collect::<Vec<_>>();

        let value_flat = array_flat_f32(py, value_alpha)?;
        if value_flat.len() != len * 3 {
            return Err(PyValueError::new_err("value_alpha shape mismatch"));
        }
        let value_alpha = value_flat
            .chunks_exact(3)
            .map(|chunk| [chunk[0], chunk[1], chunk[2]])
            .collect::<Vec<_>>();

        let terminated = match terminated {
            Some(value) => {
                let values = array_flat_bool(py, value)?;
                if values.len() != len {
                    return Err(PyValueError::new_err("terminated shape mismatch"));
                }
                values
            }
            None => vec![false; len],
        };

        let terminal_alpha = match terminal_alpha {
            Some(value) => {
                let flat = array_flat_f32(py, value)?;
                if flat.len() != len * 3 {
                    return Err(PyValueError::new_err("terminal alpha invalid"));
                }
                flat.chunks_exact(3)
                    .map(|chunk| Some([chunk[0], chunk[1], chunk[2]]))
                    .collect::<Vec<_>>()
            }
            None => vec![None; len],
        };

        let active_rows = active_rows.unwrap_or(len);
        if active_rows > len {
            return Err(PyValueError::new_err("transition output shape mismatch"));
        }
        for row in 0..active_rows {
            let start = action_offsets[row];
            let end = action_offsets[row + 1];
            if terminated[row] {
                let alpha = terminal_alpha[row].ok_or_else(|| {
                    PyValueError::new_err("terminal row missing positive terminal_alpha")
                })?;
                validate_alpha(alpha, "terminal alpha invalid")?;
                continue;
            }
            if start == end {
                return Err(PyValueError::new_err(
                    "decision nodes must have at least one legal action",
                ));
            }
            validate_alpha(value_alpha[row], "non-terminal row missing positive value_alpha")?;
            let mut seen = HashSet::with_capacity(end - start);
            for index in start..end {
                if !seen.insert(legal_actions[index]) {
                    return Err(PyValueError::new_err("duplicate legal action within one row"));
                }
                validate_alpha(q_alpha[index], "non-terminal row missing positive q_alpha")?;
            }
        }

        Ok(Self {
            len,
            observations,
            action_offsets,
            legal_actions,
            current_players,
            policy_logits,
            value_alpha,
            q_alpha,
            terminated,
            terminal_alpha,
        })
    }

    fn decision_data_for_row(&self, config: &SearchConfig, row: usize) -> PyResult<DecisionData> {
        let obs_len = config.observation_len();
        let obs_start = row * obs_len;
        let action_start = self.action_offsets[row];
        let action_end = self.action_offsets[row + 1];
        DecisionData::new(
            config.action_size,
            self.observations[obs_start..obs_start + obs_len].to_vec(),
            self.legal_actions[action_start..action_end].to_vec(),
            self.policy_logits[action_start..action_end].to_vec(),
            self.value_alpha[row],
            self.q_alpha[action_start..action_end].to_vec(),
        )
    }
}

fn validate_alpha(alpha: [f32; 3], message: &str) -> PyResult<()> {
    if alpha.iter().all(|value| value.is_finite() && *value > 0.0) {
        Ok(())
    } else {
        Err(PyValueError::new_err(message.to_string()))
    }
}

fn decision_node(
    id: NodeId,
    state: PyObject,
    current_player: i32,
    decision: DecisionData,
) -> Node {
    Node {
        id,
        generation: 0,
        parent: None,
        parent_link: None,
        depth: 0,
        state,
        current_player,
        c_v: Some(decision.value_alpha),
        cached_pi: None,
        n_down: 0,
        cache_version: 0,
        cat_outcome: NO_OUTCOME,
        cat_distance: NO_DISTANCE,
        cat_action: None,
        kind: NodeKind::Decision(decision),
    }
}

fn terminal_node(id: NodeId, state: PyObject, current_player: i32, alpha: [f32; 3]) -> Node {
    let outcome = outcome_from_alpha(alpha);
    Node {
        id,
        generation: 0,
        parent: None,
        parent_link: None,
        depth: 0,
        state,
        current_player,
        kind: NodeKind::Terminal(TerminalData { alpha, outcome }),
        c_v: Some(alpha),
        cached_pi: None,
        n_down: 1,
        cache_version: 0,
        cat_outcome: outcome,
        cat_distance: 0,
        cat_action: None,
    }
}

fn request_transitions(
    py: Python<'_>,
    forest: &mut Forest,
    max_batch_size: usize,
    pad_to: Option<usize>,
) -> PyResult<TransitionBatch> {
    let mut records = Vec::new();
    let tree_slots = forest.trees.len();
    if tree_slots == 0 {
        return empty_transition_batch(py);
    }

    loop {
        let mut made_progress = false;
        for offset in 0..tree_slots {
            if records.len() == max_batch_size {
                break;
            }
            let index = (forest.round_robin_cursor + offset) % tree_slots;
            let Some(tree) = forest.trees[index].as_mut() else {
                continue;
            };
            if (!forest.config.solve_categorical && !tree.pending_requests.is_empty())
                || tree.is_done(&forest.config)
            {
                continue;
            }
            match next_transition_request(tree, &forest.config, &mut forest.next_request_id)? {
                NextRequestResult::TransitionRequest(record) => {
                    records.push(record);
                    made_progress = true;
                }
                NextRequestResult::CompletedOneSimulation => {
                    made_progress = true;
                }
                NextRequestResult::BlockedByPendingRequest
                | NextRequestResult::TreeDone
                | NextRequestResult::NoProgress => {}
            }
        }
        forest.round_robin_cursor = (forest.round_robin_cursor + 1) % tree_slots;
        if records.len() == max_batch_size || !made_progress {
            break;
        }
    }

    if records.is_empty() {
        return empty_transition_batch(py);
    }

    let size = records.len();
    let padded_size = pad_to.unwrap_or(size);
    if padded_size < size {
        return Err(PyValueError::new_err(
            "pad_to smaller than collected active rows",
        ));
    }

    let mut parent_states = Vec::with_capacity(padded_size);
    let mut actions = Vec::with_capacity(padded_size);
    let mut active_mask = Vec::with_capacity(padded_size);
    let mut tree_ids = Vec::with_capacity(padded_size);
    let mut parent_node_ids = Vec::with_capacity(padded_size);
    let mut request_ids = Vec::with_capacity(padded_size);
    let mut tree_generations = Vec::with_capacity(padded_size);

    for record in &records {
        let tree = forest.tree(record.tree_id)?;
        let node = tree.node(record.node_id)?;
        parent_states.push(node.state.clone_ref(py));
        actions.push(record.action as i32);
        active_mask.push(true);
        tree_ids.push(record.tree_id);
        parent_node_ids.push(record.node_id);
        request_ids.push(record.request_id);
        tree_generations.push(record.tree_generation);
    }
    for _ in size..padded_size {
        parent_states.push(parent_states[0].clone_ref(py));
        actions.push(actions[0]);
        active_mask.push(false);
        tree_ids.push(tree_ids[0]);
        parent_node_ids.push(parent_node_ids[0]);
        request_ids.push(request_ids[0]);
        tree_generations.push(tree_generations[0]);
    }

    let token = forest.next_batch_token;
    forest.next_batch_token += 1;
    forest.request_table.insert(
        token,
        PendingBatch {
            records,
            padded_size,
        },
    );

    Ok(TransitionBatch {
        token,
        size,
        padded_size,
        parent_states: PyList::new(py, parent_states).into_py(py),
        actions: np_array(py, actions, "int32")?,
        active_mask: np_array(py, active_mask, "bool_")?,
        tree_ids: np_array(py, tree_ids, "uint64")?,
        parent_node_ids: np_array(py, parent_node_ids, "uint32")?,
        request_ids: np_array(py, request_ids, "uint64")?,
        tree_generations: np_array(py, tree_generations, "uint32")?,
    })
}

fn empty_transition_batch(py: Python<'_>) -> PyResult<TransitionBatch> {
    Ok(TransitionBatch {
        token: 0,
        size: 0,
        padded_size: 0,
        parent_states: PyList::empty(py).into_py(py),
        actions: np_array(py, Vec::<i32>::new(), "int32")?,
        active_mask: np_array(py, Vec::<bool>::new(), "bool_")?,
        tree_ids: np_array(py, Vec::<TreeId>::new(), "uint64")?,
        parent_node_ids: np_array(py, Vec::<NodeId>::new(), "uint32")?,
        request_ids: np_array(py, Vec::<RequestId>::new(), "uint64")?,
        tree_generations: np_array(py, Vec::<u32>::new(), "uint32")?,
    })
}

enum NextRequestResult {
    TransitionRequest(RequestRecord),
    CompletedOneSimulation,
    BlockedByPendingRequest,
    TreeDone,
    NoProgress,
}

fn next_transition_request(
    tree: &mut Tree,
    config: &SearchConfig,
    next_request_id: &mut RequestId,
) -> PyResult<NextRequestResult> {
    if config.solve_categorical {
        return next_solve_transition_request(tree, config, next_request_id);
    }
    if !config.solve_categorical && !tree.pending_requests.is_empty() {
        return Ok(NextRequestResult::BlockedByPendingRequest);
    }
    if tree.is_done(config) {
        return Ok(NextRequestResult::TreeDone);
    }

    let mut node_id = tree.root;
    let mut path = Vec::new();
    loop {
        propagate_categorical(tree, node_id, config)?;
        if tree.node(node_id)?.cat_outcome != NO_OUTCOME {
            return Ok(NextRequestResult::CompletedOneSimulation);
        }
        match &tree.node(node_id)?.kind {
            NodeKind::Terminal(data) => {
                backup_terminal(tree, &path, data.alpha, data.outcome, config)?;
                return Ok(NextRequestResult::CompletedOneSimulation);
            }
            NodeKind::Decision(_) => {
                let edge_index = match thompson_select(tree, node_id, config) {
                    Ok(edge_index) => edge_index,
                    Err(_) if config.solve_categorical => {
                        return Ok(NextRequestResult::NoProgress);
                    }
                    Err(err) => return Err(err),
                };
                let action = decision(tree, node_id)?.legal_actions[edge_index];
                if let Some(child_id) = decision(tree, node_id)?.edges[edge_index].child {
                    path.push(PathStep {
                        node_id,
                        edge_index,
                        action,
                    });
                    node_id = child_id;
                    continue;
                }

                let request_id = *next_request_id;
                *next_request_id += 1;
                tree.pending_requests.insert(request_id);
                {
                    let parent = tree.node_mut(node_id)?;
                    let NodeKind::Decision(data) = &mut parent.kind else {
                        return Err(PyRuntimeError::new_err("request parent is terminal"));
                    };
                    data.edges[edge_index].pending = true;
                }
                path.push(PathStep {
                    node_id,
                    edge_index,
                    action,
                });
                let record = RequestRecord {
                    request_id,
                    tree_id: tree.id,
                    tree_generation: tree.generation,
                    node_id,
                    node_generation: tree.node(node_id)?.generation,
                    action,
                    path,
                };
                return Ok(NextRequestResult::TransitionRequest(record));
            }
        }
    }
}

fn next_solve_transition_request(
    tree: &mut Tree,
    config: &SearchConfig,
    next_request_id: &mut RequestId,
) -> PyResult<NextRequestResult> {
    if tree.is_done(config) {
        return Ok(NextRequestResult::TreeDone);
    }
    let mut path = Vec::new();
    find_solve_transition_request(tree, tree.root, config, next_request_id, &mut path)
}

fn find_solve_transition_request(
    tree: &mut Tree,
    node_id: NodeId,
    config: &SearchConfig,
    next_request_id: &mut RequestId,
    path: &mut Vec<PathStep>,
) -> PyResult<NextRequestResult> {
    propagate_categorical(tree, node_id, config)?;
    if tree.node(node_id)?.cat_outcome != NO_OUTCOME {
        return Ok(NextRequestResult::CompletedOneSimulation);
    }
    if let NodeKind::Terminal(data) = &tree.node(node_id)?.kind {
        backup_terminal(tree, path, data.alpha, data.outcome, config)?;
        return Ok(NextRequestResult::CompletedOneSimulation);
    }

    let action_count = decision(tree, node_id)?.legal_actions.len();
    for edge_index in 0..action_count {
        let action = decision(tree, node_id)?.legal_actions[edge_index];
        let edge_snapshot = {
            let edge = &decision(tree, node_id)?.edges[edge_index];
            (edge.child, edge.pending, edge.cat_outcome)
        };
        if edge_snapshot.2 != NO_OUTCOME || edge_snapshot.1 {
            continue;
        }
        if let Some(child_id) = edge_snapshot.0 {
            path.push(PathStep {
                node_id,
                edge_index,
                action,
            });
            match find_solve_transition_request(tree, child_id, config, next_request_id, path)? {
                NextRequestResult::NoProgress => {
                    path.pop();
                    continue;
                }
                other => return Ok(other),
            }
        }

        let request_id = *next_request_id;
        *next_request_id += 1;
        tree.pending_requests.insert(request_id);
        {
            let parent = tree.node_mut(node_id)?;
            let NodeKind::Decision(data) = &mut parent.kind else {
                return Err(PyRuntimeError::new_err("request parent is terminal"));
            };
            data.edges[edge_index].pending = true;
        }
        path.push(PathStep {
            node_id,
            edge_index,
            action,
        });
        let record = RequestRecord {
            request_id,
            tree_id: tree.id,
            tree_generation: tree.generation,
            node_id,
            node_generation: tree.node(node_id)?.generation,
            action,
            path: path.clone(),
        };
        path.pop();
        return Ok(NextRequestResult::TransitionRequest(record));
    }
    Ok(NextRequestResult::NoProgress)
}

fn submit_one_transition(
    py: Python<'_>,
    forest: &mut Forest,
    record: &RequestRecord,
    batch: &ParsedNodeBatch,
    child_states: &PyAny,
    row: usize,
    profile: &mut SubmitProfile,
) -> PyResult<()> {
    profile.rows_seen += 1;
    profile.backup_path_steps += record.path.len();
    let config = forest.config.clone();
    let Ok(tree) = forest.tree_mut(record.tree_id) else {
        profile.skipped_rows += 1;
        return Ok(());
    };
    if tree.generation != record.tree_generation {
        profile.skipped_rows += 1;
        return Ok(());
    }
    let Ok(parent) = tree.node(record.node_id) else {
        profile.skipped_rows += 1;
        return Ok(());
    };
    if parent.generation != record.node_generation {
        profile.skipped_rows += 1;
        return Ok(());
    }
    if !tree.pending_requests.contains(&record.request_id) {
        profile.skipped_rows += 1;
        return Ok(());
    }

    let batch_item_start = profile.start();
    let child_state = batch_item(py, child_states, row)?;
    if let Some(start) = batch_item_start {
        profile.batch_item += start.elapsed();
    }
    let current_player = batch.current_players[row];
    let (child_id, is_new_child) = if batch.terminated[row] {
        profile.terminal_rows += 1;
        let node_insert_start = profile.start();
        let child_id = tree.nodes.len() as NodeId;
        let alpha = batch.terminal_alpha[row].ok_or_else(|| {
            PyValueError::new_err("terminal row missing positive terminal_alpha")
        })?;
        let mut child = terminal_node(child_id, child_state, current_player, alpha);
        child.parent = Some(record.node_id);
        child.parent_link = Some(ParentLink::DecisionAction {
            action: record.action,
        });
        child.depth = tree.node(record.node_id)?.depth + 1;
        tree.nodes.push(child);
        if let Some(start) = node_insert_start {
            profile.node_insert += start.elapsed();
        }
        (child_id, true)
    } else {
        let action_start = batch.action_offsets[row];
        let action_end = batch.action_offsets[row + 1];
        profile.total_legal_actions += action_end - action_start;

        let decision_data_start = profile.start();
        let decision = batch.decision_data_for_row(&config, row)?;
        if let Some(start) = decision_data_start {
            profile.decision_data += start.elapsed();
        }
        let decision_key_start = profile.start();
        let key = decision_key(current_player, &decision.observation);
        if let Some(start) = decision_key_start {
            profile.decision_key += start.elapsed();
        }
        let lookup_start = profile.start();
        let existing = tree.decision_table.get(&key).copied();
        if let Some(start) = lookup_start {
            profile.decision_lookup += start.elapsed();
        }
        if let Some(existing) = existing {
            profile.existing_decision_nodes += 1;
            (existing, false)
        } else {
            profile.new_decision_nodes += 1;
            let node_insert_start = profile.start();
            let child_id = tree.nodes.len() as NodeId;
            let mut child = decision_node(child_id, child_state, current_player, decision);
            child.parent = Some(record.node_id);
            child.parent_link = Some(ParentLink::DecisionAction {
                action: record.action,
            });
            child.depth = tree.node(record.node_id)?.depth + 1;
            tree.nodes.push(child);
            tree.decision_table.insert(key, child_id);
            if let Some(start) = node_insert_start {
                profile.node_insert += start.elapsed();
            }
            (child_id, true)
        }
    };

    let parent_update_start = profile.start();
    let parent = tree.node_mut(record.node_id)?;
    parent.cached_pi = None;
    let data = match &mut parent.kind {
        NodeKind::Decision(data) => data,
        NodeKind::Terminal(_) => return Err(PyRuntimeError::new_err("request parent is terminal")),
    };
    if data.edges[record.path.last().unwrap().edge_index].child.is_some() {
        return Err(PyRuntimeError::new_err("request edge already has a child"));
    }
    let edge = &mut data.edges[record.path.last().unwrap().edge_index];
    edge.child = Some(child_id);
    edge.pending = false;
    tree.pending_requests.remove(&record.request_id);
    if let Some(start) = parent_update_start {
        profile.parent_update += start.elapsed();
    }

    let _ = is_new_child;
    let backup_start = profile.start();
    let result = match &tree.node(child_id)?.kind {
        NodeKind::Decision(data) => backup(tree, &record.path, data.value_alpha, &config, profile),
        NodeKind::Terminal(data) => {
            backup_terminal_profile(tree, &record.path, data.alpha, data.outcome, &config, profile)
        }
    };
    if let Some(start) = backup_start {
        profile.backup += start.elapsed();
    }
    result
}

fn submit_one_transition_prepare_jax(
    py: Python<'_>,
    forest: &mut Forest,
    record: &RequestRecord,
    batch: &ParsedNodeBatch,
    child_states: &PyAny,
    row: usize,
    profile: &mut SubmitProfile,
) -> PyResult<Option<PreparedJaxBackup>> {
    profile.rows_seen += 1;
    profile.backup_path_steps += record.path.len();
    let config = forest.config.clone();
    let Ok(tree) = forest.tree_mut(record.tree_id) else {
        profile.skipped_rows += 1;
        return Ok(None);
    };
    if tree.generation != record.tree_generation {
        profile.skipped_rows += 1;
        return Ok(None);
    }
    let Ok(parent) = tree.node(record.node_id) else {
        profile.skipped_rows += 1;
        return Ok(None);
    };
    if parent.generation != record.node_generation {
        profile.skipped_rows += 1;
        return Ok(None);
    }
    if !tree.pending_requests.contains(&record.request_id) {
        profile.skipped_rows += 1;
        return Ok(None);
    }
    let batch_item_start = profile.start();
    let child_state = batch_item(py, child_states, row)?;
    if let Some(start) = batch_item_start {
        profile.batch_item += start.elapsed();
    }
    let current_player = batch.current_players[row];

    let child_id = if batch.terminated[row] {
        profile.terminal_rows += 1;
        let node_insert_start = profile.start();
        let child_id = tree.nodes.len() as NodeId;
        let alpha = batch.terminal_alpha[row].ok_or_else(|| {
            PyValueError::new_err("terminal row missing positive terminal_alpha")
        })?;
        let mut child = terminal_node(child_id, child_state, current_player, alpha);
        child.parent = Some(record.node_id);
        child.parent_link = Some(ParentLink::DecisionAction {
            action: record.action,
        });
        child.depth = tree.node(record.node_id)?.depth + 1;
        tree.nodes.push(child);
        if let Some(start) = node_insert_start {
            profile.node_insert += start.elapsed();
        }
        child_id
    } else {
        let action_start = batch.action_offsets[row];
        let action_end = batch.action_offsets[row + 1];
        profile.total_legal_actions += action_end - action_start;

        let decision_data_start = profile.start();
        let decision_data = batch.decision_data_for_row(&config, row)?;
        if let Some(start) = decision_data_start {
            profile.decision_data += start.elapsed();
        }
        let decision_key_start = profile.start();
        let key = decision_key(current_player, &decision_data.observation);
        if let Some(start) = decision_key_start {
            profile.decision_key += start.elapsed();
        }
        let lookup_start = profile.start();
        let existing = tree.decision_table.get(&key).copied();
        if let Some(start) = lookup_start {
            profile.decision_lookup += start.elapsed();
        }
        if let Some(existing) = existing {
            profile.existing_decision_nodes += 1;
            existing
        } else {
            profile.new_decision_nodes += 1;
            let node_insert_start = profile.start();
            let child_id = tree.nodes.len() as NodeId;
            let mut child = decision_node(child_id, child_state, current_player, decision_data);
            child.parent = Some(record.node_id);
            child.parent_link = Some(ParentLink::DecisionAction {
                action: record.action,
            });
            child.depth = tree.node(record.node_id)?.depth + 1;
            tree.nodes.push(child);
            tree.decision_table.insert(key, child_id);
            if let Some(start) = node_insert_start {
                profile.node_insert += start.elapsed();
            }
            child_id
        }
    };
    let child = tree.node(child_id)?;
    let leaf_alpha = if child.cat_outcome != NO_OUTCOME {
        child.c_v.unwrap_or(categorical_proxy(child.cat_outcome))
    } else {
        match &child.kind {
            NodeKind::Decision(data) => data.value_alpha,
            NodeKind::Terminal(data) => data.alpha,
        }
    };

    let parent_update_start = profile.start();
    let parent = tree.node_mut(record.node_id)?;
    parent.cached_pi = None;
    let data = match &mut parent.kind {
        NodeKind::Decision(data) => data,
        NodeKind::Terminal(_) => return Err(PyRuntimeError::new_err("request parent is terminal")),
    };
    if data.edges[record.path.last().unwrap().edge_index].child.is_some() {
        return Err(PyRuntimeError::new_err("request edge already has a child"));
    }
    let edge = &mut data.edges[record.path.last().unwrap().edge_index];
    edge.child = Some(child_id);
    edge.pending = false;
    tree.pending_requests.remove(&record.request_id);
    if let Some(start) = parent_update_start {
        profile.parent_update += start.elapsed();
    }

    let categorical_found = tree.node(child_id)?.cat_outcome != NO_OUTCOME;

    Ok(Some(PreparedJaxBackup {
        tree_id: record.tree_id,
        path: record.path.clone(),
        leaf_alpha,
        leaf_player: current_player,
        categorical_found,
    }))
}

fn advance_one_root(forest: &mut Forest, tree_id: TreeId, action: Action) -> PyResult<()> {
    let tree = forest.tree_mut(tree_id)?;
    let root = tree.root;
    let edge_index = decision(tree, root)?
        .edge_index_for_action(action)
        .ok_or_else(|| PyValueError::new_err("invalid legal action in advance_roots"))?;
    let child = decision(tree, root)?.edges[edge_index]
        .child
        .ok_or_else(|| PyValueError::new_err("advance_roots called for an action without an existing child"))?;

    tree.root = child;
    tree.generation += 1;
    tree.pending_requests.clear();
    clear_pending_edges(tree);
    let root_node = tree.node_mut(child)?;
    root_node.parent = None;
    root_node.parent_link = None;
    root_node.depth = 0;
    recompute_depths_from_root(tree)?;
    Ok(())
}

fn recompute_depths_from_root(tree: &mut Tree) -> PyResult<()> {
    let mut stack = vec![(tree.root, 0u32)];
    while let Some((node_id, depth)) = stack.pop() {
        let children = {
            let node = tree.node_mut(node_id)?;
            node.depth = depth;
            match &node.kind {
                NodeKind::Decision(data) => data
                    .edges
                    .iter()
                    .filter_map(|edge| edge.child)
                    .collect::<Vec<_>>(),
                NodeKind::Terminal(_) => Vec::new(),
            }
        };
        for child in children {
            stack.push((child, depth + 1));
        }
    }
    Ok(())
}

fn clear_pending_edges(tree: &mut Tree) {
    for node in &mut tree.nodes {
        if let NodeKind::Decision(data) = &mut node.kind {
            for edge in &mut data.edges {
                edge.pending = false;
            }
        }
    }
}

fn cached_policy_target(tree: &Tree, node_id: NodeId) -> PyResult<Option<Vec<f32>>> {
    let action_count = decision(tree, node_id)?.legal_actions.len();
    let Some(pi) = tree.node(node_id)?.cached_pi.as_ref() else {
        return Ok(None);
    };
    if pi.len() != action_count {
        return Ok(None);
    }
    if !pi.iter().all(|value| value.is_finite() && *value >= 0.0) {
        return Err(PyRuntimeError::new_err("cached policy contains invalid value"));
    }
    if pi.iter().sum::<f32>() <= 0.0 {
        return Ok(None);
    }
    Ok(Some(pi.clone()))
}

fn decision_has_categorical_edge(tree: &Tree, node_id: NodeId) -> PyResult<bool> {
    Ok(decision(tree, node_id)?
        .edges
        .iter()
        .any(|edge| edge.cat_outcome != NO_OUTCOME))
}

fn finish_trees(
    py: Python<'_>,
    forest: &mut Forest,
    tree_ids: &[TreeId],
    commit: &str,
) -> PyResult<SearchResults> {
    let mut out_tree_ids = Vec::with_capacity(tree_ids.len());
    let mut actions = Vec::with_capacity(tree_ids.len());
    let mut action_offsets = Vec::with_capacity(tree_ids.len() + 1);
    let mut legal_actions = Vec::new();
    let mut pi_search = Vec::new();
    let mut root_alpha = Vec::new();
    let mut root_q_mean = Vec::new();
    let mut beta_v = Vec::new();
    let mut q_target_kind = Vec::new();
    let mut q_target_weight = Vec::new();
    let mut q_target_outcome = Vec::new();
    let mut q_target_distance = Vec::new();
    let mut v_target_kind = Vec::new();
    let mut v_target_weight = Vec::new();
    let mut v_target_outcome = Vec::new();
    let mut v_target_distance = Vec::new();
    action_offsets.push(0i64);

    for tree_id in tree_ids {
        let config = forest.config.clone();
        let tree = forest.tree_mut(*tree_id)?;
        if !tree.pending_requests.is_empty() {
            return Err(PyValueError::new_err("finish called with pending request"));
        }
        if !matches!(tree.node(tree.root)?.kind, NodeKind::Decision(_)) {
            return Err(PyValueError::new_err("finish called on non-decision root"));
        }
        if tree.node(tree.root)?.cat_outcome == NO_OUTCOME
            && tree.node(tree.root)?.n_down < config.simulations_per_root
        {
            return Err(PyValueError::new_err(
                "finish called before simulations_per_root reached",
            ));
        }

        let root = tree.root;
        let legal_actions_for_root = decision(tree, root)?.legal_actions.clone();
        propagate_categorical(tree, root, &config)?;
        let root_is_categorical = tree.node(root)?.cat_outcome != NO_OUTCOME;
        let has_categorical_edge = !root_is_categorical && decision_has_categorical_edge(tree, root)?;
        if root_is_categorical || has_categorical_edge || cached_policy_target(tree, root)?.is_none() {
            refresh_categorical_edges(tree, root, &config)?;
        }
        let pi = if tree.node(root)?.cat_outcome != NO_OUTCOME {
            solved_policy_target(tree, root)?
        } else if let Some(pi) = cached_policy_target(tree, root)? {
            pi
        } else {
            posterior_best_policy_target(tree, root, &config)?
        };
        let mut alpha_rows = Vec::with_capacity(legal_actions_for_root.len());
        let mut q_rows = Vec::with_capacity(legal_actions_for_root.len());
        for edge_index in 0..legal_actions_for_root.len() {
            let alpha = decision_edge_posterior_ref(tree, root, edge_index, &config)?;
            alpha_rows.push(alpha);
            q_rows.push((alpha[2] - alpha[0]) / alpha.iter().sum::<f32>());
            let (kind, weight, outcome, distance) = q_native_field(tree, root, edge_index)?;
            q_target_kind.push(kind);
            q_target_weight.push(weight);
            q_target_outcome.push(outcome);
            q_target_distance.push(distance);
        }
        let selected_edge = if tree.node(root)?.cat_outcome != NO_OUTCOME {
            solved_edge_index(tree, root)?
        } else {
            match commit {
                "posterior_argmax" => argmax(&pi),
                "mean_utility_argmax" => argmax(&q_rows),
                "posterior_sample" => sample_index(&pi, &mut tree.rng)?,
                _ => unreachable!("commit mode already validated"),
            }
        };
        let (v_kind, v_weight, v_outcome, v_distance) = v_native_field(tree, root)?;
        beta_v.push(tree.node(root)?.c_v.unwrap_or(DUMMY_ALPHA));
        v_target_kind.push(v_kind);
        v_target_weight.push(v_weight);
        v_target_outcome.push(v_outcome);
        v_target_distance.push(v_distance);

        out_tree_ids.push(*tree_id);
        actions.push(legal_actions_for_root[selected_edge] as i32);
        legal_actions.extend(legal_actions_for_root.iter().map(|action| *action as i32));
        pi_search.extend(pi);
        root_alpha.extend(alpha_rows);
        root_q_mean.extend(q_rows);
        action_offsets.push(legal_actions.len() as i64);
    }

    Ok(SearchResults {
        tree_ids: np_array(py, out_tree_ids, "uint64")?,
        actions: np_array(py, actions, "int32")?,
        action_offsets: np_array(py, action_offsets, "int64")?,
        legal_actions: np_array(py, legal_actions, "int32")?,
        pi_search: np_array(py, pi_search, "float32")?,
        root_alpha: np_array_reshape(
            py,
            alpha_rows_to_flat(&root_alpha),
            vec![root_alpha.len(), 3],
            "float32",
        )?,
        root_q_mean: np_array(py, root_q_mean, "float32")?,
        beta_v: np_array_reshape(py, alpha_rows_to_flat(&beta_v), vec![beta_v.len(), 3], "float32")?,
        q_target_kind: np_array(py, q_target_kind, "int8")?,
        q_target_weight: np_array(py, q_target_weight, "float32")?,
        q_target_outcome: np_array(py, q_target_outcome, "int8")?,
        q_target_distance: np_array(py, q_target_distance, "int32")?,
        v_target_kind: np_array(py, v_target_kind, "int8")?,
        v_target_weight: np_array(py, v_target_weight, "float32")?,
        v_target_outcome: np_array(py, v_target_outcome, "int8")?,
        v_target_distance: np_array(py, v_target_distance, "int32")?,
    })
}

fn export_targets(py: Python<'_>, forest: &Forest, tree_ids: &[TreeId]) -> PyResult<SearchTargets> {
    for tree_id in tree_ids {
        if !forest.tree(*tree_id)?.pending_requests.is_empty() {
            return Err(PyValueError::new_err("export called with pending request"));
        }
    }

    let mut observations: Vec<f32> = Vec::new();
    let mut action_offsets = vec![0i64];
    let mut legal_actions: Vec<i32> = Vec::new();
    let mut policy_target: Vec<f32> = Vec::new();
    let mut q_target_alpha: Vec<[f32; 3]> = Vec::new();
    let mut q_loss_weight: Vec<f32> = Vec::new();
    let mut v_target_alpha: Vec<[f32; 3]> = Vec::new();
    let mut q_target_kind: Vec<i8> = Vec::new();
    let mut q_target_weight: Vec<f32> = Vec::new();
    let mut q_target_outcome: Vec<i8> = Vec::new();
    let mut q_target_distance: Vec<i32> = Vec::new();
    let mut v_target_kind: Vec<i8> = Vec::new();
    let mut v_target_weight: Vec<f32> = Vec::new();
    let mut v_target_outcome: Vec<i8> = Vec::new();
    let mut v_target_distance: Vec<i32> = Vec::new();
    let mut row_mask: Vec<bool> = Vec::new();
    let mut out_tree_ids: Vec<TreeId> = Vec::new();
    let mut node_ids: Vec<NodeId> = Vec::new();
    let mut depths: Vec<u32> = Vec::new();

    for tree_id in tree_ids {
        let tree = forest.tree(*tree_id)?;
        let mut stack = vec![tree.root];
        while let Some(node_id) = stack.pop() {
            let node = tree.node(node_id)?;
            let NodeKind::Decision(data) = &node.kind else {
                continue;
            };
            for edge in &data.edges {
                if let Some(child) = edge.child {
                    stack.push(child);
                }
            }
            if node.n_down == 0 || node.c_v.is_none() {
                continue;
            }
            let pi = posterior_best_policy_target_ref(tree, node_id, &forest.config)?;
            observations.extend(&data.observation);
            legal_actions.extend(data.legal_actions.iter().map(|action| *action as i32));
            policy_target.extend(&pi);
            q_loss_weight.extend(&pi);
            for edge_index in 0..data.legal_actions.len() {
                q_target_alpha.push(decision_edge_posterior_ref(
                    tree,
                    node_id,
                    edge_index,
                    &forest.config,
                )?);
                let (kind, weight, outcome, distance) =
                    q_native_field(tree, node_id, edge_index)?;
                q_target_kind.push(kind);
                q_target_weight.push(weight);
                q_target_outcome.push(outcome);
                q_target_distance.push(distance);
            }
            v_target_alpha.push(node.c_v.unwrap());
            let (kind, weight, outcome, distance) = v_native_field(tree, node_id)?;
            v_target_kind.push(kind);
            v_target_weight.push(weight);
            v_target_outcome.push(outcome);
            v_target_distance.push(distance);
            row_mask.push(true);
            out_tree_ids.push(*tree_id);
            node_ids.push(node_id);
            depths.push(node.depth);
            action_offsets.push(legal_actions.len() as i64);
        }
    }

    let rows = row_mask.len();
    Ok(SearchTargets {
        observations: np_array_reshape(
            py,
            observations,
            std::iter::once(rows)
                .chain(forest.config.observation_shape.iter().copied())
                .collect(),
            "float32",
        )?,
        action_offsets: np_array(py, action_offsets, "int64")?,
        legal_actions: np_array(py, legal_actions, "int32")?,
        policy_target: np_array(py, policy_target, "float32")?,
        q_target_alpha: np_array_reshape(
            py,
            alpha_rows_to_flat(&q_target_alpha),
            vec![q_target_alpha.len(), 3],
            "float32",
        )?,
        q_loss_weight: np_array(py, q_loss_weight, "float32")?,
        v_target_alpha: np_array_reshape(
            py,
            alpha_rows_to_flat(&v_target_alpha),
            vec![v_target_alpha.len(), 3],
            "float32",
        )?,
        q_target_kind: np_array(py, q_target_kind, "int8")?,
        q_target_weight: np_array(py, q_target_weight, "float32")?,
        q_target_outcome: np_array(py, q_target_outcome, "int8")?,
        q_target_distance: np_array(py, q_target_distance, "int32")?,
        v_target_kind: np_array(py, v_target_kind, "int8")?,
        v_target_weight: np_array(py, v_target_weight, "float32")?,
        v_target_outcome: np_array(py, v_target_outcome, "int8")?,
        v_target_distance: np_array(py, v_target_distance, "int32")?,
        row_mask: np_array(py, row_mask, "bool_")?,
        tree_ids: np_array(py, out_tree_ids, "uint64")?,
        node_ids: np_array(py, node_ids, "uint32")?,
        depths: np_array(py, depths, "uint32")?,
    })
}

fn empty_jax_backup_batch(py: Python<'_>) -> PyResult<Py<JaxBackupBatch>> {
    Py::new(
        py,
        JaxBackupBatch {
            used_jax: false,
            node_count: 0,
            path_count: 0,
            max_depth: 0,
            path_depth: 0,
            tree_ids: np_array(py, Vec::<TreeId>::new(), "uint64")?,
            node_ids: np_array(py, Vec::<NodeId>::new(), "uint32")?,
            edge_b: np_array_reshape(py, Vec::<f32>::new(), vec![0, 0, 3], "float32")?,
            edge_completed: np_array_reshape(py, Vec::<bool>::new(), vec![0, 0], "bool_")?,
            edge_r_count: np_array_reshape(py, Vec::<i32>::new(), vec![0, 0], "int32")?,
            q_alpha: np_array_reshape(py, Vec::<f32>::new(), vec![0, 0, 3], "float32")?,
            value_alpha: np_array_reshape(py, Vec::<f32>::new(), vec![0, 3], "float32")?,
            legal_mask: np_array_reshape(py, Vec::<bool>::new(), vec![0, 0], "bool_")?,
            node_players: np_array(py, Vec::<i32>::new(), "int32")?,
            path_nodes: np_array_reshape(py, Vec::<i32>::new(), vec![0, 0], "int32")?,
            path_edges: np_array_reshape(py, Vec::<i32>::new(), vec![0, 0], "int32")?,
            path_mask: np_array_reshape(py, Vec::<bool>::new(), vec![0, 0], "bool_")?,
            leaf_alpha: np_array_reshape(py, Vec::<f32>::new(), vec![0, 3], "float32")?,
            leaf_players: np_array(py, Vec::<i32>::new(), "int32")?,
        },
    )
}

fn round_up_to(value: usize, multiple: usize) -> usize {
    if value == 0 {
        multiple
    } else {
        value.div_ceil(multiple) * multiple
    }
}

fn jax_backup_node_capacity(forest: &Forest, node_count: usize) -> PyResult<usize> {
    let active_trees = forest.trees.iter().filter(|tree| tree.is_some()).count();
    let per_tree = usize::try_from(forest.config.simulations_per_root)
        .map_err(|_| PyRuntimeError::new_err("simulations_per_root does not fit in usize"))?
        .saturating_add(1);
    Ok(active_trees.saturating_mul(per_tree).max(node_count).max(1))
}

fn build_jax_backup_batch(
    py: Python<'_>,
    forest: &Forest,
    prepared: &[PreparedJaxBackup],
    path_capacity: usize,
) -> PyResult<Py<JaxBackupBatch>> {
    if prepared.is_empty() {
        return empty_jax_backup_batch(py);
    }

    let mut node_map: HashMap<(TreeId, NodeId), usize> = HashMap::new();
    let mut node_refs: Vec<(TreeId, NodeId)> = Vec::new();
    for item in prepared {
        for step in &item.path {
            let key = (item.tree_id, step.node_id);
            if !node_map.contains_key(&key) {
                node_map.insert(key, node_refs.len());
                node_refs.push(key);
            }
        }
    }
    if node_refs.is_empty() {
        return empty_jax_backup_batch(py);
    }

    let node_count = node_refs.len();
    let node_capacity = jax_backup_node_capacity(forest, node_count)?;
    let max_actions = forest.config.action_size;
    let max_depth = prepared
        .iter()
        .map(|item| item.path.len())
        .max()
        .unwrap_or(0);
    let configured_depth = usize::try_from(forest.config.simulations_per_root)
        .map_err(|_| PyRuntimeError::new_err("simulations_per_root does not fit in usize"))?;
    let path_depth = round_up_to(max_depth.max(configured_depth).max(1), JAX_BACKUP_BLOCK_DEPTH);
    let path_capacity = path_capacity.max(prepared.len()).max(1);

    let mut out_tree_ids = Vec::with_capacity(node_capacity);
    let mut out_node_ids = Vec::with_capacity(node_capacity);
    let mut edge_b: Vec<[f32; 3]> = Vec::with_capacity(node_capacity * max_actions);
    let mut edge_completed: Vec<bool> = Vec::with_capacity(node_capacity * max_actions);
    let mut edge_r_count: Vec<i32> = Vec::with_capacity(node_capacity * max_actions);
    let mut q_alpha: Vec<[f32; 3]> = Vec::with_capacity(node_capacity * max_actions);
    let mut value_alpha: Vec<[f32; 3]> = Vec::with_capacity(node_capacity);
    let mut legal_mask: Vec<bool> = Vec::with_capacity(node_capacity * max_actions);
    let mut node_players: Vec<i32> = Vec::with_capacity(node_capacity);

    for (tree_id, node_id) in &node_refs {
        let tree = forest.tree(*tree_id)?;
        let node = tree.node(*node_id)?;
        let NodeKind::Decision(data) = &node.kind else {
            return Err(PyRuntimeError::new_err("jax backup node is not a decision node"));
        };
        if node.cat_outcome != NO_OUTCOME {
            return Err(PyRuntimeError::new_err(
                "jax backup export encountered a categorical node",
            ));
        }
        out_tree_ids.push(*tree_id);
        out_node_ids.push(*node_id);
        value_alpha.push(data.value_alpha);
        node_players.push(node.current_player);
        for edge_index in 0..max_actions {
            if edge_index < data.edges.len() {
                let edge = &data.edges[edge_index];
                if edge.cat_outcome != NO_OUTCOME {
                    return Err(PyRuntimeError::new_err(
                        "jax backup export encountered a categorical edge",
                    ));
                }
                edge_b.push(edge.b);
                edge_completed.push(edge.completed);
                edge_r_count.push(i32::try_from(edge.r_count).map_err(|_| {
                    PyRuntimeError::new_err("edge visit count does not fit in int32")
                })?);
                q_alpha.push(data.q_alpha[edge_index]);
                legal_mask.push(true);
            } else {
                edge_b.push(DUMMY_ALPHA);
                edge_completed.push(false);
                edge_r_count.push(0);
                q_alpha.push(DUMMY_ALPHA);
                legal_mask.push(false);
            }
        }
    }
    let dummy_tree_id = out_tree_ids.first().copied().unwrap_or(0);
    for _ in node_count..node_capacity {
        out_tree_ids.push(dummy_tree_id);
        out_node_ids.push(0);
        value_alpha.push(DUMMY_ALPHA);
        node_players.push(0);
        for _ in 0..max_actions {
            edge_b.push(DUMMY_ALPHA);
            edge_completed.push(false);
            edge_r_count.push(0);
            q_alpha.push(DUMMY_ALPHA);
            legal_mask.push(false);
        }
    }

    let mut path_nodes = vec![0i32; path_capacity * path_depth];
    let mut path_edges = vec![0i32; path_capacity * path_depth];
    let mut path_mask = vec![false; path_capacity * path_depth];
    let mut leaf_alpha = vec![DUMMY_ALPHA; path_capacity];
    let mut leaf_players = vec![0i32; path_capacity];
    for (row, item) in prepared.iter().enumerate() {
        leaf_alpha[row] = item.leaf_alpha;
        leaf_players[row] = item.leaf_player;
        for (depth, step) in item.path.iter().enumerate() {
            let node_row = *node_map
                .get(&(item.tree_id, step.node_id))
                .ok_or_else(|| PyRuntimeError::new_err("jax backup node mapping missing"))?;
            let offset = row * path_depth + depth;
            path_nodes[offset] = i32::try_from(node_row).map_err(|_| {
                PyRuntimeError::new_err("jax backup node row does not fit in int32")
            })?;
            path_edges[offset] = i32::try_from(step.edge_index).map_err(|_| {
                PyRuntimeError::new_err("jax backup edge index does not fit in int32")
            })?;
            path_mask[offset] = true;
        }
    }

    Py::new(
        py,
        JaxBackupBatch {
            used_jax: true,
            node_count,
            path_count: prepared.len(),
            max_depth,
            path_depth,
            tree_ids: np_array(py, out_tree_ids, "uint64")?,
            node_ids: np_array(py, out_node_ids, "uint32")?,
            edge_b: np_array_reshape(
                py,
                alpha_rows_to_flat(&edge_b),
                vec![node_capacity, max_actions, 3],
                "float32",
            )?,
            edge_completed: np_array_reshape(
                py,
                edge_completed,
                vec![node_capacity, max_actions],
                "bool_",
            )?,
            edge_r_count: np_array_reshape(
                py,
                edge_r_count,
                vec![node_capacity, max_actions],
                "int32",
            )?,
            q_alpha: np_array_reshape(
                py,
                alpha_rows_to_flat(&q_alpha),
                vec![node_capacity, max_actions, 3],
                "float32",
            )?,
            value_alpha: np_array_reshape(
                py,
                alpha_rows_to_flat(&value_alpha),
                vec![node_capacity, 3],
                "float32",
            )?,
            legal_mask: np_array_reshape(py, legal_mask, vec![node_capacity, max_actions], "bool_")?,
            node_players: np_array(py, node_players, "int32")?,
            path_nodes: np_array_reshape(
                py,
                path_nodes,
                vec![path_capacity, path_depth],
                "int32",
            )?,
            path_edges: np_array_reshape(
                py,
                path_edges,
                vec![path_capacity, path_depth],
                "int32",
            )?,
            path_mask: np_array_reshape(
                py,
                path_mask,
                vec![path_capacity, path_depth],
                "bool_",
            )?,
            leaf_alpha: np_array_reshape(
                py,
                alpha_rows_to_flat(&leaf_alpha),
                vec![path_capacity, 3],
                "float32",
            )?,
            leaf_players: np_array(py, leaf_players, "int32")?,
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn apply_jax_backup_result(
    forest: &mut Forest,
    tree_ids: &[TreeId],
    node_ids: &[i64],
    edge_b: &[f32],
    edge_completed: &[bool],
    edge_r_count: &[i64],
    c_v: &[f32],
    n_down: &[i64],
    policy: &[f32],
    node_count: usize,
) -> PyResult<()> {
    let node_capacity = tree_ids.len();
    if node_count > node_capacity {
        return Err(PyValueError::new_err("jax backup node_count exceeds array capacity"));
    }
    if node_ids.len() != node_capacity
        || n_down.len() != node_capacity
        || c_v.len() != node_capacity * 3
    {
        return Err(PyValueError::new_err("jax backup node array shape mismatch"));
    }
    let max_actions = forest.config.action_size;
    if edge_b.len() != node_capacity * max_actions * 3
        || edge_completed.len() != node_capacity * max_actions
        || edge_r_count.len() != node_capacity * max_actions
        || policy.len() != node_capacity * max_actions
    {
        return Err(PyValueError::new_err("jax backup edge array shape mismatch"));
    }

    for row in 0..node_count {
        if node_ids[row] < 0 {
            return Err(PyValueError::new_err("jax backup node id must be nonnegative"));
        }
        if n_down[row] < 0 || n_down[row] > u32::MAX as i64 {
            return Err(PyValueError::new_err("jax backup n_down out of range"));
        }
        let node_id = node_ids[row] as NodeId;
        let tree = forest.tree_mut(tree_ids[row])?;
        let node = tree.node_mut(node_id)?;
        if node.cat_outcome != NO_OUTCOME {
            return Err(PyRuntimeError::new_err(
                "jax backup apply encountered a categorical node",
            ));
        }
        let NodeKind::Decision(data) = &mut node.kind else {
            return Err(PyRuntimeError::new_err("jax backup apply node is not a decision node"));
        };
        let mut cached_pi = Vec::with_capacity(data.edges.len());
        for edge_index in 0..data.edges.len() {
            let edge_offset = row * max_actions + edge_index;
            let alpha_offset = edge_offset * 3;
            let count = edge_r_count[edge_offset];
            if count < 0 || count > u32::MAX as i64 {
                return Err(PyValueError::new_err("jax backup edge count out of range"));
            }
            let pi = policy[edge_offset];
            if !pi.is_finite() || pi < 0.0 {
                return Err(PyValueError::new_err("jax backup policy contains invalid value"));
            }
            let edge = &mut data.edges[edge_index];
            if edge.cat_outcome != NO_OUTCOME {
                return Err(PyRuntimeError::new_err(
                    "jax backup apply encountered a categorical edge",
                ));
            }
            edge.b = [
                edge_b[alpha_offset],
                edge_b[alpha_offset + 1],
                edge_b[alpha_offset + 2],
            ];
            edge.completed = edge_completed[edge_offset];
            edge.r_count = count as u32;
            edge.cat_outcome = NO_OUTCOME;
            edge.cat_distance = NO_DISTANCE;
            cached_pi.push(pi);
        }
        node.c_v = Some([c_v[row * 3], c_v[row * 3 + 1], c_v[row * 3 + 2]]);
        node.cached_pi = Some(cached_pi);
        node.n_down = n_down[row] as u32;
        node.cache_version = node.cache_version.wrapping_add(1);
    }
    Ok(())
}

fn apply_pending_categorical_touches(forest: &mut Forest) -> PyResult<()> {
    let touches = std::mem::take(&mut forest.pending_categorical_touches);
    for touch in touches {
        let tree = forest.tree_mut(touch.tree_id)?;
        apply_categorical_touch(tree, &touch.path)?;
    }
    Ok(())
}

fn apply_categorical_touch(tree: &mut Tree, path: &[PathStep]) -> PyResult<()> {
    let Some(final_step) = path.last() else {
        return Ok(());
    };
    let mut child_id = decision(tree, final_step.node_id)?.edges[final_step.edge_index]
        .child
        .ok_or_else(|| PyRuntimeError::new_err("categorical touch edge has no child"))?;
    for step in path.iter().rev() {
        let changed = publish_categorical_edge_from_child_metadata(
            tree,
            step.node_id,
            step.edge_index,
            child_id,
        )?;
        if !changed && tree.node(step.node_id)?.cat_outcome == NO_OUTCOME {
            break;
        }
        try_categorize_node_from_known_edges(tree, step.node_id)?;
        if tree.node(step.node_id)?.cat_outcome == NO_OUTCOME {
            break;
        }
        child_id = step.node_id;
    }
    Ok(())
}

fn backup(
    tree: &mut Tree,
    path: &[PathStep],
    beta_leaf: [f32; 3],
    config: &SearchConfig,
    profile: &mut SubmitProfile,
) -> PyResult<()> {
    let mut beta = beta_leaf;
    for step in path.iter().rev() {
        let child_id = decision(tree, step.node_id)?.edges[step.edge_index]
            .child
            .ok_or_else(|| PyRuntimeError::new_err("backup edge has no child"))?;
        let publish_start = profile.start();
        let published =
            publish_categorical_edge_from_child(tree, step.node_id, step.edge_index, child_id, false)?;
        if let Some(start) = publish_start {
            profile.publish_child += start.elapsed();
        }
        if published {
            recompute_node_profile(tree, step.node_id, config, profile)?;
            let propagate_start = profile.start();
            propagate_categorical(tree, step.node_id, config)?;
            if let Some(start) = propagate_start {
                profile.propagate += start.elapsed();
            }
            beta = tree.node(step.node_id)?.c_v.unwrap_or(categorical_proxy(
                tree.node(step.node_id)?.cat_outcome,
            ));
            continue;
        }
        let align_start = profile.start();
        beta = align_child_to_parent(tree, child_id, step.node_id, beta)?;
        if let Some(start) = align_start {
            profile.align += start.elapsed();
        }
        let edge_write_start = profile.start();
        {
            let parent = tree.node_mut(step.node_id)?;
            let NodeKind::Decision(data) = &mut parent.kind else {
                return Err(PyRuntimeError::new_err("backup parent is not a decision node"));
            };
            let edge = &mut data.edges[step.edge_index];
            edge.b = beta;
            edge.completed = true;
            edge.r_count += 1;
            edge.cat_outcome = NO_OUTCOME;
            edge.cat_distance = NO_DISTANCE;
        }
        if let Some(start) = edge_write_start {
            profile.edge_write += start.elapsed();
        }
        recompute_node_profile(tree, step.node_id, config, profile)?;
        let propagate_start = profile.start();
        propagate_categorical(tree, step.node_id, config)?;
        if let Some(start) = propagate_start {
            profile.propagate += start.elapsed();
        }
        beta = tree.node(step.node_id)?.c_v.unwrap_or(beta);
    }
    Ok(())
}

fn backup_terminal(
    tree: &mut Tree,
    path: &[PathStep],
    terminal_alpha: [f32; 3],
    terminal_outcome: i8,
    config: &SearchConfig,
) -> PyResult<()> {
    let mut profile = SubmitProfile::default();
    backup_terminal_profile(tree, path, terminal_alpha, terminal_outcome, config, &mut profile)
}

fn backup_terminal_profile(
    tree: &mut Tree,
    path: &[PathStep],
    terminal_alpha: [f32; 3],
    terminal_outcome: i8,
    config: &SearchConfig,
    profile: &mut SubmitProfile,
) -> PyResult<()> {
    if path.is_empty() {
        return Ok(());
    }
    let final_step = path.last().expect("path is nonempty");
    let child_id = decision(tree, final_step.node_id)?.edges[final_step.edge_index]
        .child
        .ok_or_else(|| PyRuntimeError::new_err("terminal backup edge has no child"))?;
    let align_start = profile.start();
    let aligned_alpha = align_child_to_parent(tree, child_id, final_step.node_id, terminal_alpha)?;
    if let Some(start) = align_start {
        profile.align += start.elapsed();
    }
    let aligned_outcome = align_outcome(
        terminal_outcome,
        tree.node(child_id)?.current_player,
        tree.node(final_step.node_id)?.current_player,
    );
    let edge_write_start = profile.start();
    publish_categorical_edge(
        tree,
        final_step.node_id,
        final_step.edge_index,
        aligned_outcome,
        1,
        aligned_alpha,
        true,
    )?;
    if let Some(start) = edge_write_start {
        profile.edge_write += start.elapsed();
    }
    recompute_node_profile(tree, final_step.node_id, config, profile)?;
    let propagate_start = profile.start();
    propagate_categorical(tree, final_step.node_id, config)?;
    if let Some(start) = propagate_start {
        profile.propagate += start.elapsed();
    }
    let mut beta = tree.node(final_step.node_id)?.c_v.unwrap_or(aligned_alpha);
    for step in path[..path.len() - 1].iter().rev() {
        let child_id = decision(tree, step.node_id)?.edges[step.edge_index]
            .child
            .ok_or_else(|| PyRuntimeError::new_err("backup edge has no child"))?;
        let publish_start = profile.start();
        let published =
            publish_categorical_edge_from_child(tree, step.node_id, step.edge_index, child_id, false)?;
        if let Some(start) = publish_start {
            profile.publish_child += start.elapsed();
        }
        if published {
            recompute_node_profile(tree, step.node_id, config, profile)?;
            let propagate_start = profile.start();
            propagate_categorical(tree, step.node_id, config)?;
            if let Some(start) = propagate_start {
                profile.propagate += start.elapsed();
            }
            beta = tree.node(step.node_id)?.c_v.unwrap_or(categorical_proxy(
                tree.node(step.node_id)?.cat_outcome,
            ));
            continue;
        }
        let align_start = profile.start();
        beta = align_child_to_parent(tree, child_id, step.node_id, beta)?;
        if let Some(start) = align_start {
            profile.align += start.elapsed();
        }
        let edge_write_start = profile.start();
        {
            let parent = tree.node_mut(step.node_id)?;
            let NodeKind::Decision(data) = &mut parent.kind else {
                return Err(PyRuntimeError::new_err("backup parent is not a decision node"));
            };
            let edge = &mut data.edges[step.edge_index];
            edge.b = beta;
            edge.completed = true;
            edge.r_count += 1;
            edge.cat_outcome = NO_OUTCOME;
            edge.cat_distance = NO_DISTANCE;
        }
        if let Some(start) = edge_write_start {
            profile.edge_write += start.elapsed();
        }
        recompute_node_profile(tree, step.node_id, config, profile)?;
        let propagate_start = profile.start();
        propagate_categorical(tree, step.node_id, config)?;
        if let Some(start) = propagate_start {
            profile.propagate += start.elapsed();
        }
        beta = tree.node(step.node_id)?.c_v.unwrap_or(beta);
    }
    Ok(())
}

fn recompute_node(tree: &mut Tree, node_id: NodeId, config: &SearchConfig) -> PyResult<()> {
    let mut profile = SubmitProfile::default();
    recompute_node_profile(tree, node_id, config, &mut profile)
}

fn recompute_node_profile(
    tree: &mut Tree,
    node_id: NodeId,
    config: &SearchConfig,
    profile: &mut SubmitProfile,
) -> PyResult<()> {
    profile.recompute_calls += 1;
    let recompute_start = profile.start();
    let categorize_start = profile.start();
    try_categorize_node(tree, node_id)?;
    if let Some(start) = categorize_start {
        profile.recompute_categorize += start.elapsed();
    }

    let (cache, cached_pi) = if let NodeKind::Terminal(data) = &tree.node(node_id)?.kind {
        let alpha = data.alpha;
        let node = tree.node_mut(node_id)?;
        node.n_down = 1;
        (Some(alpha), None)
    } else if tree.node(node_id)?.cat_outcome != NO_OUTCOME {
        let n_down = decision(tree, node_id)?
            .edges
            .iter()
            .map(|edge| edge.r_count)
            .sum::<u32>();
        let alpha = categorical_proxy(tree.node(node_id)?.cat_outcome);
        let node = tree.node_mut(node_id)?;
        node.n_down = n_down;
        (Some(alpha), None)
    } else {
        let (n_down, value_alpha, action_count) = {
            let data = decision(tree, node_id)?;
            (
                data.edges.iter().map(|edge| edge.r_count).sum::<u32>(),
                data.value_alpha,
                data.edges.len(),
            )
        };
        if n_down == 0 {
            let node = tree.node_mut(node_id)?;
            node.n_down = 0;
            (Some(value_alpha), None)
        } else {
            profile.posterior_policy_calls += 1;
            profile.posterior_policy_action_visits +=
                config.posterior_best_samples as usize * action_count;
            let policy_start = profile.start();
            let pi = posterior_best_policy_target_ref(tree, node_id, config)?;
            if let Some(start) = policy_start {
                profile.posterior_policy += start.elapsed();
            }
            let mut evidence = [0.0f32; 3];
            for (edge_index, weight) in pi.iter().enumerate() {
                let alpha = decision_edge_posterior_ref(tree, node_id, edge_index, config)?;
                for k in 0..3 {
                    evidence[k] += *weight * alpha[k];
                }
            }
            let gamma = n_down as f32 / (config.kappa_n as f32 + n_down as f32);
            let mut c_v = [0.0f32; 3];
            for k in 0..3 {
                c_v[k] = (1.0 - gamma) * value_alpha[k] + gamma * evidence[k];
            }
            let node = tree.node_mut(node_id)?;
            node.n_down = n_down;
            (Some(c_v), Some(pi))
        }
    };
    let node = tree.node_mut(node_id)?;
    node.c_v = cache;
    node.cached_pi = cached_pi;
    node.cache_version = node.cache_version.wrapping_add(1);
    if let Some(start) = recompute_start {
        profile.recompute += start.elapsed();
    }
    Ok(())
}

fn publish_categorical_edge_from_child(
    tree: &mut Tree,
    parent_id: NodeId,
    edge_index: usize,
    child_id: NodeId,
    increment_count: bool,
) -> PyResult<bool> {
    try_categorize_node(tree, child_id)?;
    let child = tree.node(child_id)?;
    if child.cat_outcome == NO_OUTCOME {
        return Ok(false);
    }
    let outcome = align_outcome(
        child.cat_outcome,
        child.current_player,
        tree.node(parent_id)?.current_player,
    );
    let distance = child.cat_distance + 1;
    let alpha = align_child_to_parent(
        tree,
        child_id,
        parent_id,
        child.c_v.unwrap_or(categorical_proxy(child.cat_outcome)),
    )?;
    publish_categorical_edge(
        tree,
        parent_id,
        edge_index,
        outcome,
        distance,
        alpha,
        increment_count,
    )
}

fn publish_categorical_edge_from_child_metadata(
    tree: &mut Tree,
    parent_id: NodeId,
    edge_index: usize,
    child_id: NodeId,
) -> PyResult<bool> {
    try_categorize_node_from_known_edges(tree, child_id)?;
    let child = tree.node(child_id)?;
    if child.cat_outcome == NO_OUTCOME {
        return Ok(false);
    }
    let outcome = align_outcome(
        child.cat_outcome,
        child.current_player,
        tree.node(parent_id)?.current_player,
    );
    let distance = child.cat_distance + 1;
    let alpha = align_child_to_parent(
        tree,
        child_id,
        parent_id,
        child.c_v.unwrap_or(categorical_proxy(child.cat_outcome)),
    )?;
    publish_categorical_edge_metadata(tree, parent_id, edge_index, outcome, distance, alpha)
}

fn refresh_categorical_edges(
    tree: &mut Tree,
    node_id: NodeId,
    config: &SearchConfig,
) -> PyResult<()> {
    let child_edges = match &tree.node(node_id)?.kind {
        NodeKind::Decision(data) => data
            .edges
            .iter()
            .enumerate()
            .filter_map(|(edge_index, edge)| edge.child.map(|child| (edge_index, child)))
            .collect::<Vec<_>>(),
        NodeKind::Terminal(_) => Vec::new(),
    };
    for (edge_index, child_id) in child_edges {
        publish_categorical_edge_from_child(tree, node_id, edge_index, child_id, false)?;
    }
    recompute_node(tree, node_id, config)?;
    Ok(())
}

fn publish_categorical_edge(
    tree: &mut Tree,
    parent_id: NodeId,
    edge_index: usize,
    outcome: i8,
    distance: i32,
    alpha: [f32; 3],
    increment_count: bool,
) -> PyResult<bool> {
    let parent = tree.node_mut(parent_id)?;
    parent.cached_pi = None;
    let NodeKind::Decision(data) = &mut parent.kind else {
        return Err(PyRuntimeError::new_err(
            "categorical edge parent is not a decision node",
        ));
    };
    let edge = data
        .edges
        .get_mut(edge_index)
        .ok_or_else(|| PyRuntimeError::new_err("edge index out of range"))?;
    let was_same = edge.cat_outcome == outcome && edge.cat_distance == distance;
    edge.b = alpha;
    edge.completed = true;
    edge.cat_outcome = outcome;
    edge.cat_distance = distance;
    if increment_count || !was_same {
        edge.r_count = edge.r_count.saturating_add(1);
    }
    Ok(!was_same)
}

fn publish_categorical_edge_metadata(
    tree: &mut Tree,
    parent_id: NodeId,
    edge_index: usize,
    outcome: i8,
    distance: i32,
    alpha: [f32; 3],
) -> PyResult<bool> {
    let parent = tree.node_mut(parent_id)?;
    parent.cached_pi = None;
    let NodeKind::Decision(data) = &mut parent.kind else {
        return Err(PyRuntimeError::new_err(
            "categorical edge parent is not a decision node",
        ));
    };
    let edge = data
        .edges
        .get_mut(edge_index)
        .ok_or_else(|| PyRuntimeError::new_err("edge index out of range"))?;
    let was_same = edge.cat_outcome == outcome && edge.cat_distance == distance;
    edge.b = alpha;
    edge.completed = true;
    edge.cat_outcome = outcome;
    edge.cat_distance = distance;
    Ok(!was_same)
}

fn try_categorize_node(tree: &mut Tree, node_id: NodeId) -> PyResult<bool> {
    if tree.node(node_id)?.cat_outcome != NO_OUTCOME {
        return Ok(true);
    }
    if let NodeKind::Terminal(data) = &tree.node(node_id)?.kind {
        publish_categorical_node(tree, node_id, data.outcome, 0, None)?;
        return Ok(true);
    }
    let action_count = decision(tree, node_id)?.legal_actions.len();
    for edge_index in 0..action_count {
        let child_id = decision(tree, node_id)?.edges[edge_index].child;
        if let Some(child_id) = child_id {
            publish_categorical_edge_from_child(tree, node_id, edge_index, child_id, false)?;
        }
    }

    let data = decision(tree, node_id)?;
    if data.edges.is_empty() {
        return Ok(false);
    }
    let mut known = true;
    let mut wins: Vec<(usize, i32)> = Vec::new();
    let mut draws: Vec<(usize, i32)> = Vec::new();
    let mut losses: Vec<(usize, i32)> = Vec::new();
    for (edge_index, edge) in data.edges.iter().enumerate() {
        match edge.cat_outcome {
            OUTCOME_WIN => wins.push((edge_index, edge.cat_distance)),
            OUTCOME_DRAW => draws.push((edge_index, edge.cat_distance)),
            OUTCOME_LOSS => losses.push((edge_index, edge.cat_distance)),
            _ => known = false,
        }
    }
    if let Some((edge_index, distance)) = choose_distance_edge(&wins, true) {
        let action = data.legal_actions[edge_index];
        publish_categorical_node(tree, node_id, OUTCOME_WIN, distance, Some(action))?;
        return Ok(true);
    }
    if !known {
        return Ok(false);
    }
    if let Some((edge_index, distance)) = choose_distance_edge(&draws, true) {
        let action = data.legal_actions[edge_index];
        publish_categorical_node(tree, node_id, OUTCOME_DRAW, distance, Some(action))?;
        return Ok(true);
    }
    if let Some((edge_index, distance)) = choose_distance_edge(&losses, false) {
        let action = data.legal_actions[edge_index];
        publish_categorical_node(tree, node_id, OUTCOME_LOSS, distance, Some(action))?;
        return Ok(true);
    }
    Ok(false)
}

fn try_categorize_node_from_known_edges(tree: &mut Tree, node_id: NodeId) -> PyResult<bool> {
    if tree.node(node_id)?.cat_outcome != NO_OUTCOME {
        return Ok(true);
    }
    if let NodeKind::Terminal(data) = &tree.node(node_id)?.kind {
        publish_categorical_node(tree, node_id, data.outcome, 0, None)?;
        return Ok(true);
    }

    let data = decision(tree, node_id)?;
    if data.edges.is_empty() {
        return Ok(false);
    }
    let mut known = true;
    let mut wins: Vec<(usize, i32)> = Vec::new();
    let mut draws: Vec<(usize, i32)> = Vec::new();
    let mut losses: Vec<(usize, i32)> = Vec::new();
    for (edge_index, edge) in data.edges.iter().enumerate() {
        match edge.cat_outcome {
            OUTCOME_WIN => wins.push((edge_index, edge.cat_distance)),
            OUTCOME_DRAW => draws.push((edge_index, edge.cat_distance)),
            OUTCOME_LOSS => losses.push((edge_index, edge.cat_distance)),
            _ => known = false,
        }
    }
    if let Some((edge_index, distance)) = choose_distance_edge(&wins, true) {
        let action = data.legal_actions[edge_index];
        publish_categorical_node(tree, node_id, OUTCOME_WIN, distance, Some(action))?;
        return Ok(true);
    }
    if !known {
        return Ok(false);
    }
    if let Some((edge_index, distance)) = choose_distance_edge(&draws, true) {
        let action = data.legal_actions[edge_index];
        publish_categorical_node(tree, node_id, OUTCOME_DRAW, distance, Some(action))?;
        return Ok(true);
    }
    if let Some((edge_index, distance)) = choose_distance_edge(&losses, false) {
        let action = data.legal_actions[edge_index];
        publish_categorical_node(tree, node_id, OUTCOME_LOSS, distance, Some(action))?;
        return Ok(true);
    }
    Ok(false)
}

fn publish_categorical_node(
    tree: &mut Tree,
    node_id: NodeId,
    outcome: i8,
    distance: i32,
    action: Option<Action>,
) -> PyResult<()> {
    let n_down = match &tree.node(node_id)?.kind {
        NodeKind::Decision(data) => data.edges.iter().map(|edge| edge.r_count).sum::<u32>(),
        NodeKind::Terminal(_) => 1,
    };
    let node = tree.node_mut(node_id)?;
    node.cat_outcome = outcome;
    node.cat_distance = distance;
    node.cat_action = action;
    node.c_v = Some(categorical_proxy(outcome));
    node.cached_pi = None;
    node.n_down = n_down;
    node.cache_version = node.cache_version.wrapping_add(1);
    Ok(())
}

fn propagate_categorical(tree: &mut Tree, start_node_id: NodeId, config: &SearchConfig) -> PyResult<()> {
    let mut node_id = start_node_id;
    let mut seen = HashSet::new();
    while seen.insert(node_id) {
        try_categorize_node(tree, node_id)?;
        if tree.node(node_id)?.cat_outcome == NO_OUTCOME {
            return Ok(());
        }
        let Some(parent_id) = tree.node(node_id)?.parent else {
            return Ok(());
        };
        let edge_index = match tree.node(node_id)?.parent_link {
            Some(ParentLink::DecisionAction { action }) => decision(tree, parent_id)?
                .edge_index_for_action(action)
                .ok_or_else(|| PyRuntimeError::new_err("parent edge missing for child"))?,
            None => return Ok(()),
        };
        publish_categorical_edge_from_child(tree, parent_id, edge_index, node_id, false)?;
        recompute_node(tree, parent_id, config)?;
        node_id = parent_id;
    }
    Ok(())
}

fn choose_distance_edge(edges: &[(usize, i32)], prefer_short: bool) -> Option<(usize, i32)> {
    edges.iter().copied().min_by_key(|(edge_index, distance)| {
        let distance_key = if prefer_short { *distance } else { -*distance };
        (distance_key, *edge_index as i32)
    })
}

fn solved_edge_index(tree: &Tree, node_id: NodeId) -> PyResult<usize> {
    let node = tree.node(node_id)?;
    let action = node.cat_action;
    let data = decision(tree, node_id)?;
    if let Some(action) = action {
        if let Some(edge_index) = data.edge_index_for_action(action) {
            return Ok(edge_index);
        }
    }
    Ok(0)
}

fn solved_policy_target(tree: &Tree, node_id: NodeId) -> PyResult<Vec<f32>> {
    let action_count = decision(tree, node_id)?.legal_actions.len();
    let mut pi = vec![0.0f32; action_count];
    let edge_index = solved_edge_index(tree, node_id)?;
    if edge_index < pi.len() {
        pi[edge_index] = 1.0;
    }
    Ok(pi)
}

fn q_native_field(
    tree: &Tree,
    node_id: NodeId,
    edge_index: usize,
) -> PyResult<(i8, f32, i8, i32)> {
    let edge = decision(tree, node_id)?
        .edges
        .get(edge_index)
        .ok_or_else(|| PyRuntimeError::new_err("edge index out of range"))?;
    if edge.cat_outcome != NO_OUTCOME {
        Ok((TARGET_CATEGORICAL, 1.0, edge.cat_outcome, edge.cat_distance))
    } else {
        Ok((TARGET_DIRICHLET, 1.0, NO_OUTCOME, NO_DISTANCE))
    }
}

fn v_native_field(tree: &Tree, node_id: NodeId) -> PyResult<(i8, f32, i8, i32)> {
    let node = tree.node(node_id)?;
    if node.cat_outcome != NO_OUTCOME {
        Ok((TARGET_CATEGORICAL, 1.0, node.cat_outcome, node.cat_distance))
    } else {
        Ok((TARGET_DIRICHLET, 1.0, NO_OUTCOME, NO_DISTANCE))
    }
}

fn thompson_select(
    tree: &mut Tree,
    node_id: NodeId,
    config: &SearchConfig,
) -> PyResult<usize> {
    let action_count = decision(tree, node_id)?.legal_actions.len();
    if config.solve_categorical {
        for edge_index in 0..action_count {
            let edge = &decision(tree, node_id)?.edges[edge_index];
            if edge.cat_outcome == NO_OUTCOME && !edge.pending {
                return Ok(edge_index);
            }
        }
        return Err(PyRuntimeError::new_err("cannot select from solved node"));
    }
    let mut best_index = None;
    let mut best_utility = f32::NEG_INFINITY;
    for edge_index in 0..action_count {
        if decision(tree, node_id)?.edges[edge_index].cat_outcome != NO_OUTCOME {
            continue;
        }
        if decision(tree, node_id)?.edges[edge_index].pending {
            continue;
        }
        let alpha = decision_edge_posterior(tree, node_id, edge_index, config)?;
        let sample = sample_dirichlet(alpha, &mut tree.rng)?;
        let utility = sample[2] - sample[0];
        if utility > best_utility {
            best_utility = utility;
            best_index = Some(edge_index);
        }
    }
    best_index.ok_or_else(|| PyRuntimeError::new_err("cannot select from solved node"))
}

fn posterior_best_policy_target(
    tree: &mut Tree,
    node_id: NodeId,
    config: &SearchConfig,
) -> PyResult<Vec<f32>> {
    let action_count = decision(tree, node_id)?.legal_actions.len();
    let mut counts = vec![0u32; action_count];
    for _ in 0..config.posterior_best_samples {
        let mut best_index = 0usize;
        let mut best_utility = f32::NEG_INFINITY;
        for edge_index in 0..action_count {
            let utility = if decision(tree, node_id)?.edges[edge_index].cat_outcome != NO_OUTCOME {
                outcome_utility(decision(tree, node_id)?.edges[edge_index].cat_outcome)
            } else {
                let alpha = decision_edge_posterior(tree, node_id, edge_index, config)?;
                let sample = sample_dirichlet(alpha, &mut tree.rng)?;
                sample[2] - sample[0]
            };
            if utility > best_utility {
                best_utility = utility;
                best_index = edge_index;
            }
        }
        counts[best_index] += 1;
    }
    let denom = config.posterior_best_samples as f32;
    Ok(counts.into_iter().map(|count| count as f32 / denom).collect())
}

fn posterior_best_policy_target_ref(
    tree: &Tree,
    node_id: NodeId,
    config: &SearchConfig,
) -> PyResult<Vec<f32>> {
    let action_count = decision(tree, node_id)?.legal_actions.len();
    let node = tree.node(node_id)?;
    let mut rng = ChaCha20Rng::seed_from_u64(
        config.seed
            ^ tree.id.rotate_left(11)
            ^ (node_id as u64).rotate_left(29)
            ^ (node.generation as u64).rotate_left(43),
    );
    let mut counts = vec![0u32; action_count];
    for _ in 0..config.posterior_best_samples {
        let mut best_index = 0usize;
        let mut best_utility = f32::NEG_INFINITY;
        for edge_index in 0..action_count {
            let utility = if decision(tree, node_id)?.edges[edge_index].cat_outcome != NO_OUTCOME {
                outcome_utility(decision(tree, node_id)?.edges[edge_index].cat_outcome)
            } else {
                let alpha = decision_edge_posterior_ref(tree, node_id, edge_index, config)?;
                let sample = sample_dirichlet(alpha, &mut rng)?;
                sample[2] - sample[0]
            };
            if utility > best_utility {
                best_utility = utility;
                best_index = edge_index;
            }
        }
        counts[best_index] += 1;
    }
    let denom = config.posterior_best_samples as f32;
    Ok(counts.into_iter().map(|count| count as f32 / denom).collect())
}

fn decision_edge_posterior(
    tree: &mut Tree,
    node_id: NodeId,
    edge_index: usize,
    config: &SearchConfig,
) -> PyResult<[f32; 3]> {
    decision_edge_posterior_ref(tree, node_id, edge_index, config)
}

fn decision_edge_posterior_ref(
    tree: &Tree,
    node_id: NodeId,
    edge_index: usize,
    _config: &SearchConfig,
) -> PyResult<[f32; 3]> {
    let data = decision(tree, node_id)?;
    let edge = data
        .edges
        .get(edge_index)
        .ok_or_else(|| PyRuntimeError::new_err("edge index out of range"))?;
    if edge.completed {
        return Ok(edge.b);
    }
    if let Some(child_id) = edge.child {
        let child = tree.node(child_id)?;
        if is_summarizable(child) {
            return align_child_to_parent(tree, child_id, node_id, child.c_v.unwrap());
        }
        if let NodeKind::Decision(child_data) = &child.kind {
            return align_child_to_parent(tree, child_id, node_id, child_data.value_alpha);
        }
    }
    Ok(data.q_alpha[edge_index])
}

fn is_summarizable(node: &Node) -> bool {
    match node.kind {
        NodeKind::Terminal(_) => true,
        NodeKind::Decision(_) => node.n_down > 0 && node.c_v.is_some(),
    }
}

fn align_child_to_parent(
    tree: &Tree,
    child_id: NodeId,
    parent_id: NodeId,
    alpha: [f32; 3],
) -> PyResult<[f32; 3]> {
    let child = tree.node(child_id)?;
    let parent = tree.node(parent_id)?;
    if child.current_player == parent.current_player {
        Ok(alpha)
    } else {
        Ok([alpha[2], alpha[1], alpha[0]])
    }
}

fn align_outcome(outcome: i8, source_player: i32, target_player: i32) -> i8 {
    if outcome < 0 || source_player == target_player {
        return outcome;
    }
    match outcome {
        OUTCOME_LOSS => OUTCOME_WIN,
        OUTCOME_WIN => OUTCOME_LOSS,
        _ => OUTCOME_DRAW,
    }
}

fn outcome_utility(outcome: i8) -> f32 {
    match outcome {
        OUTCOME_WIN => 1.0,
        OUTCOME_LOSS => -1.0,
        _ => 0.0,
    }
}

fn outcome_from_alpha(alpha: [f32; 3]) -> i8 {
    let mut best = 0usize;
    let mut best_value = f32::NEG_INFINITY;
    for (index, value) in alpha.iter().enumerate() {
        if *value > best_value {
            best_value = *value;
            best = index;
        }
    }
    best as i8
}

fn categorical_proxy(outcome: i8) -> [f32; 3] {
    let mut alpha = [CATEGORICAL_EPSILON; 3];
    if (0..3).contains(&(outcome as i32)) {
        alpha[outcome as usize] = 1.0 - 2.0 * CATEGORICAL_EPSILON;
    }
    alpha
}

fn decision_key(current_player: i32, observation: &[f32]) -> DecisionKey {
    DecisionKey {
        current_player,
        observation_bits: observation.iter().map(|value| value.to_bits()).collect(),
    }
}

fn decision(tree: &Tree, node_id: NodeId) -> PyResult<&DecisionData> {
    match &tree.node(node_id)?.kind {
        NodeKind::Decision(data) => Ok(data),
        NodeKind::Terminal(_) => Err(PyRuntimeError::new_err("node is not a decision node")),
    }
}

fn sample_dirichlet(alpha: [f32; 3], rng: &mut ChaCha20Rng) -> PyResult<[f32; 3]> {
    let mut sample = [0.0f32; 3];
    let mut sum = 0.0f32;
    for k in 0..3 {
        let gamma = Gamma::new(alpha[k] as f64, 1.0).map_err(|_| {
            PyRuntimeError::new_err("failed to create gamma distribution for Dirichlet sample")
        })?;
        sample[k] = gamma.sample(rng) as f32;
        sum += sample[k];
    }
    if sum <= 0.0 || !sum.is_finite() {
        return Err(PyRuntimeError::new_err("invalid Dirichlet sample"));
    }
    for value in &mut sample {
        *value /= sum;
    }
    Ok(sample)
}

fn sample_index(weights: &[f32], rng: &mut ChaCha20Rng) -> PyResult<usize> {
    let dist = WeightedIndex::new(weights.iter().map(|w| w.max(0.0) as f64))
        .map_err(|_| PyRuntimeError::new_err("invalid posterior_sample weights"))?;
    Ok(dist.sample(rng))
}

fn argmax(values: &[f32]) -> usize {
    let mut best_index = 0usize;
    let mut best_value = f32::NEG_INFINITY;
    for (index, value) in values.iter().enumerate() {
        if *value > best_value {
            best_value = *value;
            best_index = index;
        }
    }
    best_index
}

fn selected_tree_ids(forest: &Forest, tree_ids: Option<&PyAny>) -> PyResult<Vec<TreeId>> {
    match tree_ids {
        Some(ids) => array_flat_u64(ids.py(), ids),
        None => Ok(forest.active_tree_ids()),
    }
}

fn validate_commit(commit: &str) -> PyResult<()> {
    match commit {
        "posterior_sample" | "posterior_argmax" | "mean_utility_argmax" => Ok(()),
        other => Err(PyValueError::new_err(format!(
            "unsupported commit mode {other:?}"
        ))),
    }
}

fn batch_item(py: Python<'_>, batch: &PyAny, index: usize) -> PyResult<PyObject> {
    Ok(batch.get_item(index)?.into_py(py))
}

fn array_flat_f32(py: Python<'_>, obj: &PyAny) -> PyResult<Vec<f32>> {
    let np = PyModule::import(py, "numpy")?;
    let arr = np.call_method1("ascontiguousarray", (obj,))?;
    if let Ok(array) = arr.downcast::<PyArrayDyn<f32>>() {
        return Ok(array.readonly().as_slice()?.to_vec());
    }
    let arr = np.call_method1("ascontiguousarray", (obj, "float32"))?;
    Ok(arr
        .downcast::<PyArrayDyn<f32>>()?
        .readonly()
        .as_slice()?
        .to_vec())
}

fn array_flat_i64(py: Python<'_>, obj: &PyAny) -> PyResult<Vec<i64>> {
    let np = PyModule::import(py, "numpy")?;
    let arr = np.call_method1("ascontiguousarray", (obj,))?;
    if let Ok(array) = arr.downcast::<PyArrayDyn<i64>>() {
        return Ok(array.readonly().as_slice()?.to_vec());
    }
    if let Ok(array) = arr.downcast::<PyArrayDyn<i32>>() {
        return Ok(array
            .readonly()
            .as_slice()?
            .iter()
            .map(|value| i64::from(*value))
            .collect());
    }
    if let Ok(array) = arr.downcast::<PyArrayDyn<u32>>() {
        return Ok(array
            .readonly()
            .as_slice()?
            .iter()
            .map(|value| i64::from(*value))
            .collect());
    }
    if let Ok(array) = arr.downcast::<PyArrayDyn<u64>>() {
        return array
            .readonly()
            .as_slice()?
            .iter()
            .map(|value| {
                i64::try_from(*value)
                    .map_err(|_| PyValueError::new_err("integer value does not fit in i64"))
            })
            .collect();
    }
    let arr = np.call_method1("ascontiguousarray", (obj, "int64"))?;
    Ok(arr
        .downcast::<PyArrayDyn<i64>>()?
        .readonly()
        .as_slice()?
        .to_vec())
}

fn array_flat_u64(py: Python<'_>, obj: &PyAny) -> PyResult<Vec<u64>> {
    let values = array_flat_i64(py, obj)?;
    values
        .into_iter()
        .map(|value| {
            if value < 0 {
                Err(PyValueError::new_err("tree id must be nonnegative"))
            } else {
                Ok(value as u64)
            }
        })
        .collect()
}

fn array_flat_bool(py: Python<'_>, obj: &PyAny) -> PyResult<Vec<bool>> {
    let np = PyModule::import(py, "numpy")?;
    let arr = np.call_method1("ascontiguousarray", (obj, "bool_"))?;
    Ok(arr
        .downcast::<PyArrayDyn<bool>>()?
        .readonly()
        .as_slice()?
        .to_vec())
}

fn np_array<T: ToPyObject>(py: Python<'_>, data: T, dtype: &str) -> PyResult<PyObject> {
    let np = PyModule::import(py, "numpy")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", dtype)?;
    Ok(np
        .call_method("array", (data.to_object(py),), Some(kwargs))?
        .into_py(py))
}

fn np_array_reshape<T: ToPyObject>(
    py: Python<'_>,
    data: T,
    shape: Vec<usize>,
    dtype: &str,
) -> PyResult<PyObject> {
    let array = np_array(py, data, dtype)?;
    let reshaped = array.as_ref(py).call_method1("reshape", (shape,))?;
    Ok(reshaped.into_py(py))
}

fn alpha_rows_to_flat(rows: &[[f32; 3]]) -> Vec<f32> {
    rows.iter().flat_map(|row| row.iter().copied()).collect()
}

#[pymodule]
fn _dqaz(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<SearchConfig>()?;
    m.add_class::<SearchEngine>()?;
    m.add_class::<TransitionBatch>()?;
    m.add_class::<SearchResults>()?;
    m.add_class::<SearchTargets>()?;
    m.add_class::<JaxBackupBatch>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decision_data(legal_actions: Vec<Action>, q_alpha: Vec<[f32; 3]>) -> PyResult<DecisionData> {
        pyo3::prepare_freethreaded_python();
        let policy_logits = vec![0.0; legal_actions.len()];
        DecisionData::new(
            128,
            vec![0.0, 1.0, 2.0],
            legal_actions,
            policy_logits,
            [1.0, 1.0, 1.0],
            q_alpha,
        )
    }

    #[test]
    fn decision_data_stores_q_and_edges_for_valid_actions_only() {
        let data = decision_data(vec![3, 42], vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            .expect("valid sparse decision data");

        assert_eq!(data.legal_actions, vec![3, 42]);
        assert_eq!(data.q_alpha.len(), data.legal_actions.len());
        assert_eq!(data.edges.len(), data.legal_actions.len());
        assert_eq!(data.edge_index_for_action(42), Some(1));
        assert_eq!(data.edge_index_for_action(7), None);
    }

    #[test]
    fn decision_data_rejects_dense_q_length() {
        let err = decision_data(
            vec![3, 42],
            vec![
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
        )
        .expect_err("q_alpha must be compact");

        assert!(err.to_string().contains("q_alpha length"));
    }

    #[test]
    fn decision_data_rejects_duplicate_legal_actions() {
        let err = decision_data(vec![3, 3], vec![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
            .expect_err("legal actions must be unique");

        assert!(err.to_string().contains("duplicate legal action"));
    }
}
