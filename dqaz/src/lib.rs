use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use rand::distributions::WeightedIndex;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, Gamma};
use std::collections::{HashMap, VecDeque};
use std::sync::{Mutex, MutexGuard};

type TreeId = u64;
type NodeId = u32;
type RequestId = u64;
type BatchToken = u64;
type OutcomeId = u32;

const DUMMY_ALPHA: [f32; 3] = [1.0, 1.0, 1.0];

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
    posterior_best_samples: usize,
    #[pyo3(get)]
    kappa_n: f32,
    #[pyo3(get)]
    seed: u64,
    #[pyo3(get)]
    debug: bool,
    #[pyo3(get)]
    game: String,
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
        game = None
    ))]
    fn new(
        action_size: usize,
        observation_shape: Vec<usize>,
        simulations_per_root: u32,
        posterior_best_samples: usize,
        kappa_n: f32,
        seed: u64,
        debug: bool,
        game: Option<String>,
    ) -> PyResult<Self> {
        let config = Self {
            action_size,
            observation_shape,
            simulations_per_root,
            posterior_best_samples,
            kappa_n,
            seed,
            debug,
            game: game.unwrap_or_else(|| "toy_deterministic".to_string()),
        };
        config.validate()?;
        Ok(config)
    }

    fn __repr__(&self) -> String {
        format!(
            "SearchConfig(action_size={}, observation_shape={:?}, simulations_per_root={}, \
             posterior_best_samples={}, kappa_n={}, seed={}, debug={}, game={:?})",
            self.action_size,
            self.observation_shape,
            self.simulations_per_root,
            self.posterior_best_samples,
            self.kappa_n,
            self.seed,
            self.debug,
            self.game
        )
    }
}

impl SearchConfig {
    fn validate(&self) -> PyResult<()> {
        if self.action_size == 0 {
            return Err(PyValueError::new_err("action_size must be positive"));
        }
        if self.action_size > u16::MAX as usize {
            return Err(PyValueError::new_err("action_size must fit in u16"));
        }
        if self.observation_shape.iter().any(|d| *d == 0) {
            return Err(PyValueError::new_err(
                "observation_shape dimensions must be positive",
            ));
        }
        if self.simulations_per_root == 0 {
            return Err(PyValueError::new_err(
                "simulations_per_root must be positive",
            ));
        }
        if self.posterior_best_samples == 0 {
            return Err(PyValueError::new_err(
                "posterior_best_samples must be positive",
            ));
        }
        if !self.kappa_n.is_finite() || self.kappa_n < 0.0 {
            return Err(PyValueError::new_err("kappa_n must be finite and >= 0"));
        }
        GameKind::from_name(&self.game)?;
        Ok(())
    }

    fn observation_len(&self) -> usize {
        self.observation_shape.iter().product()
    }
}

#[pyclass(module = "dqaz")]
struct EvalBatch {
    #[pyo3(get)]
    token: BatchToken,
    #[pyo3(get)]
    size: usize,
    observations: PyObject,
    legal_masks: PyObject,
    tree_ids: PyObject,
    node_ids: PyObject,
    request_ids: PyObject,
    tree_generations: PyObject,
}

#[pymethods]
impl EvalBatch {
    #[getter]
    fn observations(&self, py: Python<'_>) -> PyObject {
        self.observations.clone_ref(py)
    }

    #[getter]
    fn legal_masks(&self, py: Python<'_>) -> PyObject {
        self.legal_masks.clone_ref(py)
    }

    #[getter]
    fn tree_ids(&self, py: Python<'_>) -> PyObject {
        self.tree_ids.clone_ref(py)
    }

    #[getter]
    fn node_ids(&self, py: Python<'_>) -> PyObject {
        self.node_ids.clone_ref(py)
    }

    #[getter]
    fn request_ids(&self, py: Python<'_>) -> PyObject {
        self.request_ids.clone_ref(py)
    }

    #[getter]
    fn tree_generations(&self, py: Python<'_>) -> PyObject {
        self.tree_generations.clone_ref(py)
    }
}

#[pyclass(module = "dqaz")]
struct SearchResults {
    tree_ids: PyObject,
    actions: PyObject,
    pi_search: PyObject,
    root_alpha: PyObject,
    root_q_mean: PyObject,
    legal_masks: PyObject,
}

#[pymethods]
impl SearchResults {
    #[getter]
    fn tree_ids(&self, py: Python<'_>) -> PyObject {
        self.tree_ids.clone_ref(py)
    }

    #[getter]
    fn actions(&self, py: Python<'_>) -> PyObject {
        self.actions.clone_ref(py)
    }

    #[getter]
    fn pi_search(&self, py: Python<'_>) -> PyObject {
        self.pi_search.clone_ref(py)
    }

    #[getter]
    fn root_alpha(&self, py: Python<'_>) -> PyObject {
        self.root_alpha.clone_ref(py)
    }

    #[getter]
    fn root_q_mean(&self, py: Python<'_>) -> PyObject {
        self.root_q_mean.clone_ref(py)
    }

    #[getter]
    fn legal_masks(&self, py: Python<'_>) -> PyObject {
        self.legal_masks.clone_ref(py)
    }
}

#[pyclass(module = "dqaz")]
struct SearchTargets {
    observations: PyObject,
    legal_masks: PyObject,
    policy_target: PyObject,
    q_target_alpha: PyObject,
    q_loss_weight: PyObject,
    v_target_alpha: PyObject,
    row_mask: PyObject,
    tree_ids: PyObject,
    node_ids: PyObject,
    depths: PyObject,
}

#[pymethods]
impl SearchTargets {
    #[getter]
    fn observations(&self, py: Python<'_>) -> PyObject {
        self.observations.clone_ref(py)
    }

    #[getter]
    fn legal_masks(&self, py: Python<'_>) -> PyObject {
        self.legal_masks.clone_ref(py)
    }

    #[getter]
    fn policy_target(&self, py: Python<'_>) -> PyObject {
        self.policy_target.clone_ref(py)
    }

    #[getter]
    fn q_target_alpha(&self, py: Python<'_>) -> PyObject {
        self.q_target_alpha.clone_ref(py)
    }

    #[getter]
    fn q_loss_weight(&self, py: Python<'_>) -> PyObject {
        self.q_loss_weight.clone_ref(py)
    }

    #[getter]
    fn v_target_alpha(&self, py: Python<'_>) -> PyObject {
        self.v_target_alpha.clone_ref(py)
    }

    #[getter]
    fn row_mask(&self, py: Python<'_>) -> PyObject {
        self.row_mask.clone_ref(py)
    }

    #[getter]
    fn tree_ids(&self, py: Python<'_>) -> PyObject {
        self.tree_ids.clone_ref(py)
    }

    #[getter]
    fn node_ids(&self, py: Python<'_>) -> PyObject {
        self.node_ids.clone_ref(py)
    }

    #[getter]
    fn depths(&self, py: Python<'_>) -> PyObject {
        self.depths.clone_ref(py)
    }
}

#[pyclass(module = "dqaz")]
struct SearchEngine {
    inner: Mutex<EngineInner>,
}

#[pymethods]
impl SearchEngine {
    #[new]
    fn new(config: SearchConfig) -> PyResult<Self> {
        config.validate()?;
        let game = GameKind::from_name(&config.game)?;
        Ok(Self {
            inner: Mutex::new(EngineInner::new(config, game)),
        })
    }

    fn add_roots(&self, py: Python<'_>, root_states: &PyAny) -> PyResult<PyObject> {
        let (shape, bytes) = numpy_flat_u8(py, root_states)?;
        if shape.len() != 2 {
            return Err(PyValueError::new_err(
                "root_states must have shape [batch, state_bytes]",
            ));
        }
        let mut inner = self.lock()?;
        let state_bytes = inner.game.state_bytes();
        if shape[1] != state_bytes {
            return Err(PyValueError::new_err(format!(
                "root_states second dimension must be {}, got {}",
                state_bytes, shape[1]
            )));
        }
        let mut ids = Vec::with_capacity(shape[0]);
        for row in bytes.chunks_exact(state_bytes) {
            ids.push(inner.add_root(row)?);
        }
        numpy_array(py, ids, "uint64", &[shape[0]])
    }

    fn request_evaluations(
        &self,
        py: Python<'_>,
        max_batch_size: usize,
    ) -> PyResult<Py<EvalBatch>> {
        let mut inner = self.lock()?;
        let batch = inner.request_evaluations(py, max_batch_size)?;
        Py::new(py, batch)
    }

    fn submit_evaluations(
        &self,
        py: Python<'_>,
        token: BatchToken,
        policy_logits: &PyAny,
        value_alpha: &PyAny,
        q_alpha: &PyAny,
    ) -> PyResult<()> {
        let (policy_shape, policy) = numpy_flat_f32(py, policy_logits)?;
        let (value_shape, value) = numpy_flat_f32(py, value_alpha)?;
        let (q_shape, q) = numpy_flat_f32(py, q_alpha)?;
        self.lock()?
            .submit_evaluations(token, policy_shape, policy, value_shape, value, q_shape, q)
    }

    #[pyo3(signature = (tree_ids = None))]
    fn is_done(&self, py: Python<'_>, tree_ids: Option<&PyAny>) -> PyResult<bool> {
        let ids = optional_tree_ids(py, tree_ids)?;
        self.lock()?.is_done(ids.as_deref())
    }

    fn stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        self.lock()?.stats(py)
    }

    #[pyo3(signature = (tree_ids = None, commit = "posterior_sample"))]
    fn finish(
        &self,
        py: Python<'_>,
        tree_ids: Option<&PyAny>,
        commit: &str,
    ) -> PyResult<Py<SearchResults>> {
        let ids = optional_tree_ids(py, tree_ids)?;
        let results = self.lock()?.finish(py, ids.as_deref(), commit)?;
        Py::new(py, results)
    }

    #[pyo3(signature = (tree_ids = None))]
    fn export_targets(
        &self,
        py: Python<'_>,
        tree_ids: Option<&PyAny>,
    ) -> PyResult<Py<SearchTargets>> {
        let ids = optional_tree_ids(py, tree_ids)?;
        let targets = self.lock()?.export_targets(py, ids.as_deref())?;
        Py::new(py, targets)
    }

    #[pyo3(signature = (tree_ids = None))]
    fn clear(&self, py: Python<'_>, tree_ids: Option<&PyAny>) -> PyResult<()> {
        let ids = optional_tree_ids(py, tree_ids)?;
        self.lock()?.clear(ids.as_deref())
    }

    fn clear_all(&self) -> PyResult<()> {
        self.lock()?.clear(None)
    }

    fn advance_roots(&self, py: Python<'_>, tree_ids: &PyAny, actions: &PyAny) -> PyResult<()> {
        let ids = numpy_flat_u64(py, tree_ids)?.1;
        let action_values = numpy_flat_i32(py, actions)?.1;
        self.lock()?.advance_roots(&ids, &action_values)
    }

    fn advance_categorical_roots(
        &self,
        py: Python<'_>,
        tree_ids: &PyAny,
        outcome_ids: &PyAny,
    ) -> PyResult<()> {
        let ids = numpy_flat_u64(py, tree_ids)?.1;
        let outcomes = numpy_flat_u32(py, outcome_ids)?.1;
        self.lock()?.advance_categorical_roots(&ids, &outcomes)
    }
}

impl SearchEngine {
    fn lock(&self) -> PyResult<MutexGuard<'_, EngineInner>> {
        self.inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("SearchEngine mutex is poisoned"))
    }
}

#[derive(Clone)]
enum GameKind {
    ToyDeterministic,
    ToyCategorical,
}

impl GameKind {
    fn from_name(name: &str) -> PyResult<Self> {
        match name {
            "toy_deterministic" => Ok(Self::ToyDeterministic),
            "toy_categorical" => Ok(Self::ToyCategorical),
            _ => Err(PyValueError::new_err(format!(
                "unknown dqaz game {name:?}; expected 'toy_deterministic' or 'toy_categorical'"
            ))),
        }
    }

    fn state_bytes(&self) -> usize {
        3
    }

    fn decode_state(&self, bytes: &[u8]) -> PyResult<State> {
        if bytes.len() != self.state_bytes() {
            return Err(PyValueError::new_err("wrong serialized state length"));
        }
        match self {
            Self::ToyDeterministic => Ok(State::ToyDeterministic {
                depth: bytes[0],
                player: bytes[1] & 1,
                last_action: bytes[2],
            }),
            Self::ToyCategorical => Ok(State::ToyCategorical {
                phase: bytes[0],
                player: bytes[1] & 1,
                outcome: bytes[2],
            }),
        }
    }

    fn node_kind(&self, state: &State) -> PyResult<GameNodeKind> {
        match (self, state) {
            (
                Self::ToyDeterministic,
                State::ToyDeterministic {
                    depth, last_action, ..
                },
            ) => {
                if *depth >= 3 {
                    Ok(GameNodeKind::Terminal {
                        alpha: deterministic_terminal_alpha(*last_action),
                    })
                } else {
                    Ok(GameNodeKind::Decision)
                }
            }
            (Self::ToyCategorical, State::ToyCategorical { phase, player, .. }) => match *phase {
                0 => Ok(GameNodeKind::Decision),
                1 => Ok(GameNodeKind::Categorical(vec![
                    CategoricalOutcome {
                        outcome_id: 0,
                        probability: 0.25,
                        state: State::ToyCategorical {
                            phase: 2,
                            player: *player,
                            outcome: 0,
                        },
                    },
                    CategoricalOutcome {
                        outcome_id: 1,
                        probability: 0.75,
                        state: State::ToyCategorical {
                            phase: 2,
                            player: *player,
                            outcome: 1,
                        },
                    },
                ])),
                _ => {
                    let alpha = match state {
                        State::ToyCategorical { outcome: 0, .. } => [1.0, 1.0, 5.0],
                        State::ToyCategorical { outcome: 1, .. } => [5.0, 1.0, 1.0],
                        State::ToyCategorical { outcome: 2, .. } => [1.0, 2.0, 2.0],
                        _ => [2.0, 1.0, 1.0],
                    };
                    Ok(GameNodeKind::Terminal { alpha })
                }
            },
            _ => Err(PyValueError::new_err(
                "serialized state does not match configured toy game",
            )),
        }
    }

    fn legal_mask(&self, state: &State, action_size: usize, out: &mut [bool]) -> PyResult<()> {
        out.fill(false);
        match self.node_kind(state)? {
            GameNodeKind::Decision => out.fill(true),
            _ => {}
        }
        if out.len() != action_size {
            return Err(PyRuntimeError::new_err("internal legal mask length mismatch"));
        }
        Ok(())
    }

    fn step_action(&self, state: &State, action: usize, action_size: usize) -> PyResult<State> {
        if action >= action_size {
            return Err(PyValueError::new_err("action out of range"));
        }
        match (self, state) {
            (
                Self::ToyDeterministic,
                State::ToyDeterministic { depth, player, .. },
            ) => Ok(State::ToyDeterministic {
                depth: depth.saturating_add(1),
                player: player ^ 1,
                last_action: action as u8,
            }),
            (Self::ToyCategorical, State::ToyCategorical { phase: 0, player, .. }) => {
                if action == 0 {
                    Ok(State::ToyCategorical {
                        phase: 1,
                        player: player ^ 1,
                        outcome: 0,
                    })
                } else {
                    Ok(State::ToyCategorical {
                        phase: 2,
                        player: player ^ 1,
                        outcome: (action as u8).saturating_add(1),
                    })
                }
            }
            _ => Err(PyValueError::new_err(
                "cannot step action from a non-decision state",
            )),
        }
    }

    fn encode_observation(&self, state: &State, out: &mut [f32], action_size: usize) {
        out.fill(0.0);
        let denom = action_size.max(1) as f32;
        match (self, state) {
            (
                Self::ToyDeterministic,
                State::ToyDeterministic {
                    depth,
                    player,
                    last_action,
                },
            ) => {
                if !out.is_empty() {
                    out[0] = *depth as f32 / 3.0;
                }
                if out.len() > 1 {
                    out[1] = *player as f32;
                }
                if out.len() > 2 {
                    out[2] = *last_action as f32 / denom;
                }
                if out.len() > 3 {
                    out[3] = 1.0;
                }
            }
            (
                Self::ToyCategorical,
                State::ToyCategorical {
                    phase,
                    player,
                    outcome,
                },
            ) => {
                if !out.is_empty() {
                    out[0] = *phase as f32 / 2.0;
                }
                if out.len() > 1 {
                    out[1] = *player as f32;
                }
                if out.len() > 2 {
                    out[2] = *outcome as f32 / denom;
                }
                if out.len() > 3 {
                    out[3] = 1.0;
                }
            }
            _ => {}
        }
    }

    fn align_wdl(&self, from_state: &State, to_state: &State, alpha: [f32; 3]) -> [f32; 3] {
        if from_state.player() == to_state.player() {
            alpha
        } else {
            [alpha[2], alpha[1], alpha[0]]
        }
    }
}

fn deterministic_terminal_alpha(last_action: u8) -> [f32; 3] {
    match last_action % 3 {
        0 => [1.0, 1.0, 4.0],
        1 => [1.0, 4.0, 1.0],
        _ => [4.0, 1.0, 1.0],
    }
}

#[derive(Clone)]
enum State {
    ToyDeterministic {
        depth: u8,
        player: u8,
        last_action: u8,
    },
    ToyCategorical {
        phase: u8,
        player: u8,
        outcome: u8,
    },
}

impl State {
    fn player(&self) -> u8 {
        match self {
            Self::ToyDeterministic { player, .. } => *player,
            Self::ToyCategorical { player, .. } => *player,
        }
    }
}

enum GameNodeKind {
    Decision,
    Categorical(Vec<CategoricalOutcome>),
    Terminal { alpha: [f32; 3] },
}

struct CategoricalOutcome {
    outcome_id: OutcomeId,
    probability: f32,
    state: State,
}

struct EngineInner {
    config: SearchConfig,
    game: GameKind,
    trees: Vec<Option<Tree>>,
    free_tree_slots: Vec<usize>,
    next_tree_id: TreeId,
    next_request_id: RequestId,
    next_batch_token: BatchToken,
    request_table: HashMap<BatchToken, Vec<RequestRecord>>,
    round_robin_cursor: usize,
}

impl EngineInner {
    fn new(config: SearchConfig, game: GameKind) -> Self {
        Self {
            config,
            game,
            trees: Vec::new(),
            free_tree_slots: Vec::new(),
            next_tree_id: 1,
            next_request_id: 1,
            next_batch_token: 1,
            request_table: HashMap::new(),
            round_robin_cursor: 0,
        }
    }

    fn add_root(&mut self, row: &[u8]) -> PyResult<TreeId> {
        let state = self.game.decode_state(row)?;
        let tree_id = self.next_tree_id;
        self.next_tree_id += 1;
        let seed = mix_seed(self.config.seed, tree_id, 0);
        let tree = Tree::new(tree_id, seed, state, &self.game, &self.config)?;
        if let Some(slot) = self.free_tree_slots.pop() {
            self.trees[slot] = Some(tree);
        } else {
            self.trees.push(Some(tree));
        }
        Ok(tree_id)
    }

    fn request_evaluations(
        &mut self,
        py: Python<'_>,
        max_batch_size: usize,
    ) -> PyResult<EvalBatch> {
        if max_batch_size == 0 {
            return self.empty_eval_batch(py);
        }

        let live = self.live_indices();
        if live.is_empty() {
            return self.empty_eval_batch(py);
        }

        let mut records = Vec::new();
        while records.len() < max_batch_size {
            let mut made_progress = false;
            for offset in 0..live.len() {
                if records.len() >= max_batch_size {
                    break;
                }
                let live_pos = (self.round_robin_cursor + offset) % live.len();
                let idx = live[live_pos];
                let Some(tree) = self.trees[idx].as_mut() else {
                    continue;
                };
                let result = next_request(
                    tree,
                    &self.game,
                    &self.config,
                    &mut self.next_request_id,
                )?;
                match result {
                    NextRequestResult::NeuralRequest(record) => {
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
            self.round_robin_cursor = (self.round_robin_cursor + 1) % live.len();
            if !made_progress {
                break;
            }
        }

        if records.is_empty() {
            return self.empty_eval_batch(py);
        }

        let token = self.next_batch_token;
        self.next_batch_token += 1;

        let size = records.len();
        let obs_len = self.config.observation_len();
        let mut observations = Vec::with_capacity(size * obs_len);
        let mut legal_masks = Vec::with_capacity(size * self.config.action_size);
        let mut tree_ids = Vec::with_capacity(size);
        let mut node_ids = Vec::with_capacity(size);
        let mut request_ids = Vec::with_capacity(size);
        let mut tree_generations = Vec::with_capacity(size);

        for record in &records {
            let tree = self.tree_by_id(record.tree_id)?;
            let node = tree.node(record.node_id)?;
            let mut obs = vec![0.0; obs_len];
            self.game
                .encode_observation(&node.state, &mut obs, self.config.action_size);
            observations.extend(obs);

            let decision = node
                .kind
                .decision()
                .ok_or_else(|| PyRuntimeError::new_err("request node is not a decision node"))?;
            legal_masks.extend(decision.legal_mask.iter().copied());

            tree_ids.push(record.tree_id);
            node_ids.push(record.node_id);
            request_ids.push(record.request_id);
            tree_generations.push(record.tree_generation);
        }

        self.request_table.insert(token, records);

        let mut obs_shape = Vec::with_capacity(1 + self.config.observation_shape.len());
        obs_shape.push(size);
        obs_shape.extend(self.config.observation_shape.iter().copied());
        Ok(EvalBatch {
            token,
            size,
            observations: numpy_array(py, observations, "float32", &obs_shape)?,
            legal_masks: numpy_array(
                py,
                legal_masks,
                "bool_",
                &[size, self.config.action_size],
            )?,
            tree_ids: numpy_array(py, tree_ids, "uint64", &[size])?,
            node_ids: numpy_array(py, node_ids, "uint32", &[size])?,
            request_ids: numpy_array(py, request_ids, "uint64", &[size])?,
            tree_generations: numpy_array(py, tree_generations, "uint32", &[size])?,
        })
    }

    fn submit_evaluations(
        &mut self,
        token: BatchToken,
        policy_shape: Vec<usize>,
        policy: Vec<f32>,
        value_shape: Vec<usize>,
        value: Vec<f32>,
        q_shape: Vec<usize>,
        q: Vec<f32>,
    ) -> PyResult<()> {
        let records = self
            .request_table
            .remove(&token)
            .ok_or_else(|| PyKeyError::new_err(format!("unknown batch token {token}")))?;
        let b = records.len();
        let a = self.config.action_size;

        if policy_shape != [b, a] {
            return Err(PyValueError::new_err(format!(
                "policy_logits must have shape [{b}, {a}], got {policy_shape:?}"
            )));
        }
        if value_shape != [b, 3] {
            return Err(PyValueError::new_err(format!(
                "value_alpha must have shape [{b}, 3], got {value_shape:?}"
            )));
        }
        if q_shape != [b, a, 3] {
            return Err(PyValueError::new_err(format!(
                "q_alpha must have shape [{b}, {a}, 3], got {q_shape:?}"
            )));
        }

        if self.config.debug {
            if value.iter().any(|x| !x.is_finite() || *x <= 0.0)
                || q.iter().any(|x| !x.is_finite() || *x <= 0.0)
            {
                return Err(PyValueError::new_err(
                    "value_alpha and q_alpha must be finite and strictly positive",
                ));
            }
        }

        for (row, record) in records.into_iter().enumerate() {
            let Some(tree_idx) = self.tree_index_by_id(record.tree_id) else {
                continue;
            };
            let tree = self.trees[tree_idx].as_mut().expect("tree index is live");
            if tree.generation != record.tree_generation {
                continue;
            }
            if tree.pending_request != Some(record.request_id) {
                continue;
            }
            if !tree.has_node(record.node_id)
                || tree.node(record.node_id)?.generation != record.node_generation
            {
                tree.pending_request = None;
                continue;
            }

            let mut row_q = Vec::with_capacity(a);
            for action in 0..a {
                let offset = row * a * 3 + action * 3;
                row_q.push([q[offset], q[offset + 1], q[offset + 2]]);
            }
            let value_alpha = [value[row * 3], value[row * 3 + 1], value[row * 3 + 2]];
            let row_policy = policy[row * a..(row + 1) * a].to_vec();

            {
                let node = tree.node_mut(record.node_id)?;
                let decision = node.kind.decision_mut().ok_or_else(|| {
                    PyRuntimeError::new_err("submitted request node is not decision")
                })?;
                match decision.eval_status {
                    DecisionEvalStatus::PendingEval { request_id }
                        if request_id == record.request_id => {}
                    _ => {
                        tree.pending_request = None;
                        continue;
                    }
                }
                decision.policy_logits = row_policy;
                decision.value_alpha = value_alpha;
                decision.q_alpha = row_q;
                decision.eval_status = DecisionEvalStatus::Expanded;
                node.c_v = Some(value_alpha);
                node.n_down = 0;
                node.cache_version = node.cache_version.wrapping_add(1);
            }

            tree.pending_request = None;
            if !record.is_root_request {
                backup_path(tree, &self.game, &self.config, &record.path, value_alpha)?;
            }
        }

        Ok(())
    }

    fn is_done(&self, selected: Option<&[TreeId]>) -> PyResult<bool> {
        let ids = self.selected_tree_ids(selected)?;
        for id in ids {
            let tree = self.tree_by_id(id)?;
            if !tree.is_done(&self.config) {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        let d = PyDict::new(py);
        d.set_item("live_trees", self.live_indices().len())?;
        d.set_item("pending_batches", self.request_table.len())?;
        d.set_item(
            "pending_requests",
            self.request_table.values().map(Vec::len).sum::<usize>(),
        )?;
        Ok(d.into_py(py))
    }

    fn finish(
        &mut self,
        py: Python<'_>,
        selected: Option<&[TreeId]>,
        commit: &str,
    ) -> PyResult<SearchResults> {
        let ids = self.selected_tree_ids(selected)?;
        let g = ids.len();
        let a = self.config.action_size;
        let mut actions = Vec::with_capacity(g);
        let mut pi_rows = Vec::with_capacity(g * a);
        let mut root_alpha = Vec::with_capacity(g * a * 3);
        let mut root_q_mean = Vec::with_capacity(g * a);
        let mut legal_masks = Vec::with_capacity(g * a);

        for id in &ids {
            let tree_idx = self
                .tree_index_by_id(*id)
                .ok_or_else(|| PyKeyError::new_err(format!("unknown tree id {id}")))?;
            let tree = self.trees[tree_idx].as_mut().expect("tree index is live");
            if tree.pending_request.is_some() {
                return Err(PyValueError::new_err("finish called with pending request"));
            }
            let root = tree.root;
            let legal_mask = {
                let decision = tree.node(root)?.kind.decision().ok_or_else(|| {
                    PyValueError::new_err("finish called on non-decision root")
                })?;
                if !matches!(decision.eval_status, DecisionEvalStatus::Expanded) {
                    return Err(PyValueError::new_err("finish called before root expanded"));
                }
                if tree.node(root)?.n_down < self.config.simulations_per_root {
                    return Err(PyValueError::new_err(
                        "finish called before simulations_per_root reached",
                    ));
                }
                decision.legal_mask.clone()
            };

            let pi = posterior_best_for_node(tree, root, &self.game, &self.config)?;
            let mut means = vec![0.0; a];
            for action in 0..a {
                let legal = legal_mask[action];
                legal_masks.push(legal);
                let alpha = if legal {
                    decision_edge_posterior(tree, root, action, &self.game, &self.config)?
                } else {
                    DUMMY_ALPHA
                };
                root_alpha.extend(alpha);
                let sum = alpha[0] + alpha[1] + alpha[2];
                means[action] = if legal {
                    (alpha[2] - alpha[0]) / sum
                } else {
                    f32::NEG_INFINITY
                };
                root_q_mean.push(means[action]);
            }

            let committed = match commit {
                "posterior_sample" => sample_weighted_index(&pi, &mut tree.rng)?,
                "posterior_argmax" => argmax_masked(&pi, &legal_mask)?,
                "mean_utility_argmax" => argmax_masked(&means, &legal_mask)?,
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "unsupported commit mode {commit:?}"
                    )))
                }
            };
            actions.push(committed as i32);
            pi_rows.extend(pi);
        }

        Ok(SearchResults {
            tree_ids: numpy_array(py, ids, "uint64", &[g])?,
            actions: numpy_array(py, actions, "int32", &[g])?,
            pi_search: numpy_array(py, pi_rows, "float32", &[g, a])?,
            root_alpha: numpy_array(py, root_alpha, "float32", &[g, a, 3])?,
            root_q_mean: numpy_array(py, root_q_mean, "float32", &[g, a])?,
            legal_masks: numpy_array(py, legal_masks, "bool_", &[g, a])?,
        })
    }

    fn export_targets(
        &mut self,
        py: Python<'_>,
        selected: Option<&[TreeId]>,
    ) -> PyResult<SearchTargets> {
        let ids = self.selected_tree_ids(selected)?;
        let a = self.config.action_size;
        let obs_len = self.config.observation_len();
        let mut observations = Vec::new();
        let mut legal_masks = Vec::new();
        let mut policy_target = Vec::new();
        let mut q_target_alpha = Vec::new();
        let mut q_loss_weight = Vec::new();
        let mut v_target_alpha = Vec::new();
        let mut row_mask = Vec::new();
        let mut out_tree_ids = Vec::new();
        let mut node_ids = Vec::new();
        let mut depths = Vec::new();

        for tree_id in ids {
            let tree_idx = self
                .tree_index_by_id(tree_id)
                .ok_or_else(|| PyKeyError::new_err(format!("unknown tree id {tree_id}")))?;
            let tree = self.trees[tree_idx].as_mut().expect("tree index is live");
            if tree.pending_request.is_some() {
                return Err(PyValueError::new_err("export called with pending request"));
            }

            let retained = retained_nodes(tree);
            for node_id in retained {
                if !is_exportable_decision(tree, node_id)? {
                    continue;
                }
                let pi = posterior_best_for_node(tree, node_id, &self.game, &self.config)?;
                let node = tree.node(node_id)?;
                let decision = node.kind.decision().expect("exportable decision");

                let mut obs = vec![0.0; obs_len];
                self.game
                    .encode_observation(&node.state, &mut obs, self.config.action_size);
                observations.extend(obs);
                legal_masks.extend(decision.legal_mask.iter().copied());
                policy_target.extend(pi.iter().copied());
                q_loss_weight.extend(pi.iter().copied());
                for action in 0..a {
                    let alpha = if decision.legal_mask[action] {
                        decision_edge_posterior(tree, node_id, action, &self.game, &self.config)?
                    } else {
                        DUMMY_ALPHA
                    };
                    q_target_alpha.extend(alpha);
                }
                v_target_alpha.extend(node.c_v.expect("exportable c_v"));
                row_mask.push(true);
                out_tree_ids.push(tree_id);
                node_ids.push(node_id);
                depths.push(node.depth);
            }
        }

        let rows = row_mask.len();
        let mut obs_shape = Vec::with_capacity(1 + self.config.observation_shape.len());
        obs_shape.push(rows);
        obs_shape.extend(self.config.observation_shape.iter().copied());

        Ok(SearchTargets {
            observations: numpy_array(py, observations, "float32", &obs_shape)?,
            legal_masks: numpy_array(py, legal_masks, "bool_", &[rows, a])?,
            policy_target: numpy_array(py, policy_target, "float32", &[rows, a])?,
            q_target_alpha: numpy_array(py, q_target_alpha, "float32", &[rows, a, 3])?,
            q_loss_weight: numpy_array(py, q_loss_weight, "float32", &[rows, a])?,
            v_target_alpha: numpy_array(py, v_target_alpha, "float32", &[rows, 3])?,
            row_mask: numpy_array(py, row_mask, "bool_", &[rows])?,
            tree_ids: numpy_array(py, out_tree_ids, "uint64", &[rows])?,
            node_ids: numpy_array(py, node_ids, "uint32", &[rows])?,
            depths: numpy_array(py, depths, "uint32", &[rows])?,
        })
    }

    fn clear(&mut self, selected: Option<&[TreeId]>) -> PyResult<()> {
        let ids = self.selected_tree_ids(selected)?;
        for id in ids {
            if let Some(idx) = self.tree_index_by_id(id) {
                self.trees[idx] = None;
                self.free_tree_slots.push(idx);
            }
        }
        Ok(())
    }

    fn advance_roots(&mut self, ids: &[TreeId], actions: &[i32]) -> PyResult<()> {
        if ids.len() != actions.len() {
            return Err(PyValueError::new_err(
                "tree_ids and actions must have the same length",
            ));
        }
        for (&tree_id, &action_i32) in ids.iter().zip(actions) {
            if action_i32 < 0 {
                return Err(PyValueError::new_err("actions must be nonnegative"));
            }
            let action = action_i32 as usize;
            let tree_idx = self
                .tree_index_by_id(tree_id)
                .ok_or_else(|| PyKeyError::new_err(format!("unknown tree id {tree_id}")))?;
            let tree = self.trees[tree_idx].as_mut().expect("tree index is live");
            let root = tree.root;
            let legal = {
                let decision = tree.node(root)?.kind.decision().ok_or_else(|| {
                    PyValueError::new_err("advance_roots requires a decision root")
                })?;
                action < decision.legal_mask.len() && decision.legal_mask[action]
            };
            if !legal {
                return Err(PyValueError::new_err("invalid legal action in advance_roots"));
            }
            let child = get_or_create_decision_child(
                tree,
                root,
                action,
                &self.game,
                &self.config,
            )?;
            tree.root = child;
            tree.pending_request = None;
            tree.generation = tree.generation.wrapping_add(1);
            tree.rng = ChaCha20Rng::seed_from_u64(mix_seed(
                self.config.seed,
                tree.id,
                tree.generation,
            ));
            reset_subtree_parent_and_depth(tree, child, None, None, 0)?;
        }
        Ok(())
    }

    fn advance_categorical_roots(
        &mut self,
        ids: &[TreeId],
        outcome_ids: &[OutcomeId],
    ) -> PyResult<()> {
        if ids.len() != outcome_ids.len() {
            return Err(PyValueError::new_err(
                "tree_ids and outcome_ids must have the same length",
            ));
        }
        for (&tree_id, &outcome_id) in ids.iter().zip(outcome_ids) {
            let tree_idx = self
                .tree_index_by_id(tree_id)
                .ok_or_else(|| PyKeyError::new_err(format!("unknown tree id {tree_id}")))?;
            let tree = self.trees[tree_idx].as_mut().expect("tree index is live");
            let root = tree.root;
            let child = {
                let data = tree.node(root)?.kind.categorical().ok_or_else(|| {
                    PyValueError::new_err(
                        "advance_categorical_roots requires a categorical root",
                    )
                })?;
                data.outcomes
                    .iter()
                    .find(|edge| edge.outcome_id == outcome_id)
                    .map(|edge| edge.child)
                    .ok_or_else(|| {
                        PyValueError::new_err(
                            "invalid outcome id in advance_categorical_roots",
                        )
                    })?
            };
            tree.root = child;
            tree.pending_request = None;
            tree.generation = tree.generation.wrapping_add(1);
            tree.rng = ChaCha20Rng::seed_from_u64(mix_seed(
                self.config.seed,
                tree.id,
                tree.generation,
            ));
            reset_subtree_parent_and_depth(tree, child, None, None, 0)?;
        }
        Ok(())
    }

    fn empty_eval_batch(&self, py: Python<'_>) -> PyResult<EvalBatch> {
        let mut obs_shape = Vec::with_capacity(1 + self.config.observation_shape.len());
        obs_shape.push(0);
        obs_shape.extend(self.config.observation_shape.iter().copied());
        Ok(EvalBatch {
            token: 0,
            size: 0,
            observations: numpy_array(py, Vec::<f32>::new(), "float32", &obs_shape)?,
            legal_masks: numpy_array(
                py,
                Vec::<bool>::new(),
                "bool_",
                &[0, self.config.action_size],
            )?,
            tree_ids: numpy_array(py, Vec::<u64>::new(), "uint64", &[0])?,
            node_ids: numpy_array(py, Vec::<u32>::new(), "uint32", &[0])?,
            request_ids: numpy_array(py, Vec::<u64>::new(), "uint64", &[0])?,
            tree_generations: numpy_array(py, Vec::<u32>::new(), "uint32", &[0])?,
        })
    }

    fn live_indices(&self) -> Vec<usize> {
        self.trees
            .iter()
            .enumerate()
            .filter_map(|(idx, tree)| tree.as_ref().map(|_| idx))
            .collect()
    }

    fn selected_tree_ids(&self, selected: Option<&[TreeId]>) -> PyResult<Vec<TreeId>> {
        if let Some(ids) = selected {
            return Ok(ids.to_vec());
        }
        Ok(self
            .trees
            .iter()
            .filter_map(|tree| tree.as_ref().map(|tree| tree.id))
            .collect())
    }

    fn tree_index_by_id(&self, id: TreeId) -> Option<usize> {
        self.trees
            .iter()
            .position(|tree| tree.as_ref().is_some_and(|tree| tree.id == id))
    }

    fn tree_by_id(&self, id: TreeId) -> PyResult<&Tree> {
        self.trees
            .iter()
            .flatten()
            .find(|tree| tree.id == id)
            .ok_or_else(|| PyKeyError::new_err(format!("unknown tree id {id}")))
    }
}

struct Tree {
    id: TreeId,
    generation: u32,
    nodes: Vec<Node>,
    root: NodeId,
    pending_request: Option<RequestId>,
    rng: ChaCha20Rng,
}

impl Tree {
    fn new(
        id: TreeId,
        seed: u64,
        root_state: State,
        game: &GameKind,
        config: &SearchConfig,
    ) -> PyResult<Self> {
        let mut tree = Self {
            id,
            generation: 0,
            nodes: Vec::new(),
            root: 0,
            pending_request: None,
            rng: ChaCha20Rng::seed_from_u64(seed),
        };
        let root = tree.create_node_from_state(game, config, root_state, None, None, 0)?;
        tree.root = root;
        Ok(tree)
    }

    fn create_node_from_state(
        &mut self,
        game: &GameKind,
        config: &SearchConfig,
        state: State,
        parent: Option<NodeId>,
        parent_link: Option<ParentLink>,
        depth: u32,
    ) -> PyResult<NodeId> {
        let id = self.nodes.len() as NodeId;
        let node_kind = match game.node_kind(&state)? {
            GameNodeKind::Decision => {
                let mut legal_mask = vec![false; config.action_size];
                game.legal_mask(&state, config.action_size, &mut legal_mask)?;
                NodeKind::Decision(DecisionData {
                    eval_status: DecisionEvalStatus::Unexpanded,
                    policy_logits: vec![0.0; config.action_size],
                    value_alpha: DUMMY_ALPHA,
                    q_alpha: vec![DUMMY_ALPHA; config.action_size],
                    legal_mask,
                    edges: vec![DecisionEdge::default(); config.action_size],
                })
            }
            GameNodeKind::Terminal { alpha } => {
                validate_positive_alpha(alpha, config.debug)?;
                NodeKind::Terminal(TerminalData { alpha })
            }
            GameNodeKind::Categorical(_) => NodeKind::Categorical(CategoricalData {
                outcomes: Vec::new(),
                complete: false,
            }),
        };

        self.nodes.push(Node {
            generation: 0,
            parent,
            parent_link,
            depth,
            state: state.clone(),
            kind: node_kind,
            c_v: None,
            n_down: 0,
            cache_version: 0,
        });

        if matches!(self.nodes[id as usize].kind, NodeKind::Terminal(_)) {
            recompute_node_cache(self, id, game, config)?;
        }

        if let GameNodeKind::Categorical(outcomes) = game.node_kind(&state)? {
            validate_categorical_outcomes(&outcomes, config.debug)?;
            let mut edges = Vec::with_capacity(outcomes.len());
            for outcome in outcomes {
                let child = self.create_node_from_state(
                    game,
                    config,
                    outcome.state,
                    Some(id),
                    Some(ParentLink::CategoricalOutcome {
                        outcome_id: outcome.outcome_id,
                    }),
                    depth + 1,
                )?;
                edges.push(CategoricalEdge {
                    outcome_id: outcome.outcome_id,
                    probability: outcome.probability,
                    child,
                    completed: false,
                    b: DUMMY_ALPHA,
                    r_count: 0,
                    child_cache_version: None,
                });
            }
            let node = self.node_mut(id)?;
            node.kind = NodeKind::Categorical(CategoricalData {
                outcomes: edges,
                complete: false,
            });
        }

        Ok(id)
    }

    fn is_done(&self, config: &SearchConfig) -> bool {
        if self.pending_request.is_some() {
            return false;
        }
        let Ok(root) = self.node(self.root) else {
            return false;
        };
        match &root.kind {
            NodeKind::Decision(decision) => {
                matches!(decision.eval_status, DecisionEvalStatus::Expanded)
                    && root.n_down >= config.simulations_per_root
            }
            NodeKind::Terminal(_) => true,
            NodeKind::Categorical(_) => false,
        }
    }

    fn node(&self, id: NodeId) -> PyResult<&Node> {
        self.nodes
            .get(id as usize)
            .ok_or_else(|| PyRuntimeError::new_err(format!("missing node id {id}")))
    }

    fn node_mut(&mut self, id: NodeId) -> PyResult<&mut Node> {
        self.nodes
            .get_mut(id as usize)
            .ok_or_else(|| PyRuntimeError::new_err(format!("missing node id {id}")))
    }

    fn has_node(&self, id: NodeId) -> bool {
        (id as usize) < self.nodes.len()
    }
}

struct Node {
    generation: u32,
    parent: Option<NodeId>,
    parent_link: Option<ParentLink>,
    depth: u32,
    state: State,
    kind: NodeKind,
    c_v: Option<[f32; 3]>,
    n_down: u32,
    cache_version: u32,
}

enum NodeKind {
    Decision(DecisionData),
    Categorical(CategoricalData),
    Terminal(TerminalData),
}

impl NodeKind {
    fn decision(&self) -> Option<&DecisionData> {
        match self {
            Self::Decision(data) => Some(data),
            _ => None,
        }
    }

    fn decision_mut(&mut self) -> Option<&mut DecisionData> {
        match self {
            Self::Decision(data) => Some(data),
            _ => None,
        }
    }

    fn categorical(&self) -> Option<&CategoricalData> {
        match self {
            Self::Categorical(data) => Some(data),
            _ => None,
        }
    }

    fn categorical_mut(&mut self) -> Option<&mut CategoricalData> {
        match self {
            Self::Categorical(data) => Some(data),
            _ => None,
        }
    }
}

#[allow(dead_code)]
#[derive(Clone, Copy)]
enum ParentLink {
    DecisionAction { action: usize },
    CategoricalOutcome { outcome_id: OutcomeId },
}

struct DecisionData {
    eval_status: DecisionEvalStatus,
    policy_logits: Vec<f32>,
    value_alpha: [f32; 3],
    q_alpha: Vec<[f32; 3]>,
    legal_mask: Vec<bool>,
    edges: Vec<DecisionEdge>,
}

#[derive(Clone, Copy)]
enum DecisionEvalStatus {
    Unexpanded,
    PendingEval { request_id: RequestId },
    Expanded,
}

#[derive(Clone)]
struct DecisionEdge {
    child: Option<NodeId>,
    completed: bool,
    b: [f32; 3],
    r_count: u32,
    child_cache_version: Option<u32>,
}

impl Default for DecisionEdge {
    fn default() -> Self {
        Self {
            child: None,
            completed: false,
            b: DUMMY_ALPHA,
            r_count: 0,
            child_cache_version: None,
        }
    }
}

struct CategoricalData {
    outcomes: Vec<CategoricalEdge>,
    complete: bool,
}

struct CategoricalEdge {
    outcome_id: OutcomeId,
    probability: f32,
    child: NodeId,
    completed: bool,
    b: [f32; 3],
    r_count: u32,
    child_cache_version: Option<u32>,
}

struct TerminalData {
    alpha: [f32; 3],
}

#[derive(Clone)]
struct RequestRecord {
    request_id: RequestId,
    tree_id: TreeId,
    tree_generation: u32,
    node_id: NodeId,
    node_generation: u32,
    path: Vec<PathStep>,
    is_root_request: bool,
}

#[derive(Clone)]
enum PathStep {
    DecisionAction {
        node_id: NodeId,
        action: usize,
        child_id: NodeId,
    },
    CategoricalOutcome {
        node_id: NodeId,
        edge_index: usize,
        outcome_id: OutcomeId,
        child_id: NodeId,
    },
}

enum NextRequestResult {
    NeuralRequest(RequestRecord),
    CompletedOneSimulation,
    BlockedByPendingRequest,
    TreeDone,
    NoProgress,
}

fn next_request(
    tree: &mut Tree,
    game: &GameKind,
    config: &SearchConfig,
    next_request_id: &mut RequestId,
) -> PyResult<NextRequestResult> {
    if tree.pending_request.is_some() {
        return Ok(NextRequestResult::BlockedByPendingRequest);
    }
    if tree.is_done(config) {
        return Ok(NextRequestResult::TreeDone);
    }

    let mut node_id = tree.root;
    let mut path = Vec::new();
    loop {
        let kind_tag = node_kind_tag(tree.node(node_id)?);
        match kind_tag {
            NodeKindTag::Terminal => {
                if path.is_empty() {
                    return Ok(NextRequestResult::TreeDone);
                }
                let alpha = match &tree.node(node_id)?.kind {
                    NodeKind::Terminal(data) => data.alpha,
                    _ => unreachable!(),
                };
                backup_path(tree, game, config, &path, alpha)?;
                return Ok(NextRequestResult::CompletedOneSimulation);
            }
            NodeKindTag::Categorical => {
                let Some((edge_index, outcome_id, child_id)) =
                    categorical_select(tree, node_id)?
                else {
                    return Ok(NextRequestResult::NoProgress);
                };
                path.push(PathStep::CategoricalOutcome {
                    node_id,
                    edge_index,
                    outcome_id,
                    child_id,
                });
                node_id = child_id;
            }
            NodeKindTag::Decision => {
                let status = tree
                    .node(node_id)?
                    .kind
                    .decision()
                    .expect("decision tag")
                    .eval_status;
                match status {
                    DecisionEvalStatus::Unexpanded => {
                        let request_id = *next_request_id;
                        *next_request_id += 1;
                        {
                            let decision = tree
                                .node_mut(node_id)?
                                .kind
                                .decision_mut()
                                .expect("decision node");
                            decision.eval_status =
                                DecisionEvalStatus::PendingEval { request_id };
                        }
                        tree.pending_request = Some(request_id);
                        return Ok(NextRequestResult::NeuralRequest(RequestRecord {
                            request_id,
                            tree_id: tree.id,
                            tree_generation: tree.generation,
                            node_id,
                            node_generation: tree.node(node_id)?.generation,
                            path: path.clone(),
                            is_root_request: path.is_empty(),
                        }));
                    }
                    DecisionEvalStatus::PendingEval { .. } => {
                        return Ok(NextRequestResult::BlockedByPendingRequest);
                    }
                    DecisionEvalStatus::Expanded => {
                        let action = thompson_select(tree, node_id, game, config)?;
                        let child_id =
                            get_or_create_decision_child(tree, node_id, action, game, config)?;
                        path.push(PathStep::DecisionAction {
                            node_id,
                            action,
                            child_id,
                        });
                        node_id = child_id;
                    }
                }
            }
        }
    }
}

#[derive(Clone, Copy)]
enum NodeKindTag {
    Decision,
    Categorical,
    Terminal,
}

fn node_kind_tag(node: &Node) -> NodeKindTag {
    match node.kind {
        NodeKind::Decision(_) => NodeKindTag::Decision,
        NodeKind::Categorical(_) => NodeKindTag::Categorical,
        NodeKind::Terminal(_) => NodeKindTag::Terminal,
    }
}

fn get_or_create_decision_child(
    tree: &mut Tree,
    node_id: NodeId,
    action: usize,
    game: &GameKind,
    config: &SearchConfig,
) -> PyResult<NodeId> {
    if let Some(child) = tree
        .node(node_id)?
        .kind
        .decision()
        .and_then(|data| data.edges.get(action))
        .and_then(|edge| edge.child)
    {
        return Ok(child);
    }

    let parent_state = tree.node(node_id)?.state.clone();
    let child_state = game.step_action(&parent_state, action, config.action_size)?;
    let child = tree.create_node_from_state(
        game,
        config,
        child_state,
        Some(node_id),
        Some(ParentLink::DecisionAction { action }),
        tree.node(node_id)?.depth + 1,
    )?;
    let decision = tree
        .node_mut(node_id)?
        .kind
        .decision_mut()
        .ok_or_else(|| PyRuntimeError::new_err("parent is not decision"))?;
    decision.edges[action].child = Some(child);
    Ok(child)
}

fn categorical_select(tree: &mut Tree, node_id: NodeId) -> PyResult<Option<(usize, OutcomeId, NodeId)>> {
    let mut missing = Vec::new();
    let mut weights = Vec::new();
    let mut complete = true;
    {
        let data = tree
            .node(node_id)?
            .kind
            .categorical()
            .ok_or_else(|| PyRuntimeError::new_err("node is not categorical"))?;
        for (idx, edge) in data.outcomes.iter().enumerate() {
            if edge.probability <= 0.0 {
                continue;
            }
            let available = edge.completed || is_summarizable_node(tree, edge.child)?;
            if !available {
                missing.push((idx, edge.probability, edge.outcome_id, edge.child));
                complete = false;
            }
            weights.push((idx, edge.probability, edge.outcome_id, edge.child));
        }
    }

    if !missing.is_empty() {
        missing.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.2.cmp(&b.2))
        });
        let (idx, _, outcome_id, child) = missing[0];
        return Ok(Some((idx, outcome_id, child)));
    }

    if complete {
        let probs: Vec<f32> = weights.iter().map(|(_, p, _, _)| *p).collect();
        let selected = sample_weighted_index(&probs, &mut tree.rng)?;
        let (idx, _, outcome_id, child) = weights[selected];
        Ok(Some((idx, outcome_id, child)))
    } else {
        Ok(None)
    }
}

fn thompson_select(
    tree: &mut Tree,
    node_id: NodeId,
    game: &GameKind,
    config: &SearchConfig,
) -> PyResult<usize> {
    let decision = tree
        .node(node_id)?
        .kind
        .decision()
        .ok_or_else(|| PyRuntimeError::new_err("node is not decision"))?;
    let mut candidates = Vec::new();
    for action in 0..config.action_size {
        if decision.legal_mask[action] {
            let alpha = decision_edge_posterior(tree, node_id, action, game, config)?;
            candidates.push((action, alpha));
        }
    }
    if candidates.is_empty() {
        return Err(PyValueError::new_err("decision node has no legal actions"));
    }

    let mut best_action = candidates[0].0;
    let mut best_utility = f32::NEG_INFINITY;
    for (action, alpha) in candidates {
        let phi = sample_dirichlet3(alpha, &mut tree.rng)?;
        let utility = phi[2] - phi[0];
        if utility > best_utility {
            best_utility = utility;
            best_action = action;
        }
    }
    Ok(best_action)
}

fn backup_path(
    tree: &mut Tree,
    game: &GameKind,
    config: &SearchConfig,
    path: &[PathStep],
    leaf_alpha: [f32; 3],
) -> PyResult<()> {
    let mut beta = leaf_alpha;
    for step in path.iter().rev() {
        let (parent_id, child_id) = match step {
            PathStep::DecisionAction {
                node_id,
                action,
                child_id,
            } => {
                let child_state = tree.node(*child_id)?.state.clone();
                let parent_state = tree.node(*node_id)?.state.clone();
                let child_cache_version = tree.node(*child_id)?.cache_version;
                let aligned = game.align_wdl(&child_state, &parent_state, beta);
                let decision = tree
                    .node_mut(*node_id)?
                    .kind
                    .decision_mut()
                    .ok_or_else(|| PyRuntimeError::new_err("backup parent is not decision"))?;
                let edge = &mut decision.edges[*action];
                edge.completed = true;
                edge.b = aligned;
                edge.r_count = edge.r_count.saturating_add(1);
                edge.child_cache_version = Some(child_cache_version);
                (*node_id, *child_id)
            }
            PathStep::CategoricalOutcome {
                node_id,
                edge_index,
                outcome_id,
                child_id,
            } => {
                let child_state = tree.node(*child_id)?.state.clone();
                let parent_state = tree.node(*node_id)?.state.clone();
                let child_cache_version = tree.node(*child_id)?.cache_version;
                let aligned = game.align_wdl(&child_state, &parent_state, beta);
                let data = tree
                    .node_mut(*node_id)?
                    .kind
                    .categorical_mut()
                    .ok_or_else(|| PyRuntimeError::new_err("backup parent is not categorical"))?;
                let edge = data
                    .outcomes
                    .get_mut(*edge_index)
                    .ok_or_else(|| PyRuntimeError::new_err("categorical edge index missing"))?;
                if edge.outcome_id != *outcome_id {
                    return Err(PyRuntimeError::new_err(
                        "categorical outcome changed during backup",
                    ));
                }
                edge.completed = true;
                edge.b = aligned;
                edge.r_count = edge.r_count.saturating_add(1);
                edge.child_cache_version = Some(child_cache_version);
                (*node_id, *child_id)
            }
        };
        let _ = child_id;
        recompute_node_cache(tree, parent_id, game, config)?;
        if is_summarizable_node(tree, parent_id)? {
            beta = tree
                .node(parent_id)?
                .c_v
                .ok_or_else(|| PyRuntimeError::new_err("summarizable node lacks C^V"))?;
        } else {
            break;
        }
    }
    Ok(())
}

fn recompute_node_cache(
    tree: &mut Tree,
    node_id: NodeId,
    game: &GameKind,
    config: &SearchConfig,
) -> PyResult<()> {
    let tag = node_kind_tag(tree.node(node_id)?);
    match tag {
        NodeKindTag::Terminal => {
            let alpha = match tree.node(node_id)?.kind {
                NodeKind::Terminal(ref data) => data.alpha,
                _ => unreachable!(),
            };
            let node = tree.node_mut(node_id)?;
            node.c_v = Some(alpha);
            node.n_down = 1;
            node.cache_version = node.cache_version.wrapping_add(1);
        }
        NodeKindTag::Categorical => {
            let (available, mix, n_down) = {
                let data = tree.node(node_id)?.kind.categorical().expect("categorical");
                let mut mix = [0.0, 0.0, 0.0];
                let mut available = true;
                let mut n_down = 0u32;
                for edge in &data.outcomes {
                    n_down = n_down.saturating_add(edge.r_count);
                    if edge.probability <= 0.0 {
                        continue;
                    }
                    if let Some(alpha) =
                        categorical_edge_posterior(tree, node_id, edge, game)?
                    {
                        for i in 0..3 {
                            mix[i] += edge.probability * alpha[i];
                        }
                    } else {
                        available = false;
                    }
                }
                (available, mix, n_down)
            };
            let node = tree.node_mut(node_id)?;
            let data = node.kind.categorical_mut().expect("categorical");
            data.complete = available;
            node.n_down = n_down;
            if available {
                node.c_v = Some(mix);
                node.cache_version = node.cache_version.wrapping_add(1);
            } else {
                node.c_v = None;
            }
        }
        NodeKindTag::Decision => {
            let (expanded, value_alpha, n_down) = {
                let decision = tree.node(node_id)?.kind.decision().expect("decision");
                let expanded = matches!(decision.eval_status, DecisionEvalStatus::Expanded);
                let n_down = decision
                    .edges
                    .iter()
                    .zip(decision.legal_mask.iter())
                    .filter(|(_, legal)| **legal)
                    .map(|(edge, _)| edge.r_count)
                    .sum::<u32>();
                (expanded, decision.value_alpha, n_down)
            };
            if !expanded {
                let node = tree.node_mut(node_id)?;
                node.c_v = None;
                node.n_down = 0;
                return Ok(());
            }

            let c_v = if n_down == 0 {
                value_alpha
            } else {
                let pi = posterior_best_for_node(tree, node_id, game, config)?;
                let mut expected = [0.0, 0.0, 0.0];
                for (action, weight) in pi.iter().enumerate() {
                    if *weight == 0.0 {
                        continue;
                    }
                    let alpha =
                        decision_edge_posterior(tree, node_id, action, game, config)?;
                    for i in 0..3 {
                        expected[i] += *weight * alpha[i];
                    }
                }
                let gamma = n_down as f32 / (config.kappa_n + n_down as f32);
                [
                    (1.0 - gamma) * value_alpha[0] + gamma * expected[0],
                    (1.0 - gamma) * value_alpha[1] + gamma * expected[1],
                    (1.0 - gamma) * value_alpha[2] + gamma * expected[2],
                ]
            };
            let node = tree.node_mut(node_id)?;
            node.c_v = Some(c_v);
            node.n_down = n_down;
            node.cache_version = node.cache_version.wrapping_add(1);
        }
    }
    Ok(())
}

fn is_summarizable_node(tree: &Tree, node_id: NodeId) -> PyResult<bool> {
    let node = tree.node(node_id)?;
    match &node.kind {
        NodeKind::Terminal(_) => Ok(node.c_v.is_some()),
        NodeKind::Categorical(data) => Ok(data.complete && node.c_v.is_some()),
        NodeKind::Decision(decision) => {
            let has_child_evidence = decision
                .edges
                .iter()
                .zip(decision.legal_mask.iter())
                .any(|(edge, legal)| *legal && edge.r_count > 0);
            Ok(matches!(decision.eval_status, DecisionEvalStatus::Expanded)
                && has_child_evidence
                && node.c_v.is_some())
        }
    }
}

fn decision_edge_posterior(
    tree: &Tree,
    node_id: NodeId,
    action: usize,
    game: &GameKind,
    config: &SearchConfig,
) -> PyResult<[f32; 3]> {
    let node = tree.node(node_id)?;
    let decision = node
        .kind
        .decision()
        .ok_or_else(|| PyRuntimeError::new_err("node is not decision"))?;
    if action >= config.action_size || !decision.legal_mask[action] {
        return Ok(DUMMY_ALPHA);
    }
    let edge = &decision.edges[action];
    if edge.completed {
        return Ok(edge.b);
    }
    decision_edge_base(tree, node_id, action, game)
}

fn decision_edge_base(
    tree: &Tree,
    node_id: NodeId,
    action: usize,
    game: &GameKind,
) -> PyResult<[f32; 3]> {
    let node = tree.node(node_id)?;
    let decision = node
        .kind
        .decision()
        .ok_or_else(|| PyRuntimeError::new_err("node is not decision"))?;
    let edge = &decision.edges[action];
    if let Some(child_id) = edge.child {
        let child = tree.node(child_id)?;
        if is_summarizable_node(tree, child_id)? {
            return Ok(game.align_wdl(
                &child.state,
                &node.state,
                child.c_v.expect("summarizable child cache"),
            ));
        }
        if let NodeKind::Decision(child_decision) = &child.kind {
            if matches!(child_decision.eval_status, DecisionEvalStatus::Expanded) {
                return Ok(game.align_wdl(
                    &child.state,
                    &node.state,
                    child_decision.value_alpha,
                ));
            }
        }
    }
    Ok(decision.q_alpha[action])
}

fn categorical_edge_posterior(
    tree: &Tree,
    node_id: NodeId,
    edge: &CategoricalEdge,
    game: &GameKind,
) -> PyResult<Option<[f32; 3]>> {
    if edge.completed {
        return Ok(Some(edge.b));
    }
    if is_summarizable_node(tree, edge.child)? {
        let node = tree.node(node_id)?;
        let child = tree.node(edge.child)?;
        return Ok(Some(game.align_wdl(
            &child.state,
            &node.state,
            child.c_v.expect("summarizable child cache"),
        )));
    }
    Ok(None)
}

fn posterior_best_for_node(
    tree: &mut Tree,
    node_id: NodeId,
    game: &GameKind,
    config: &SearchConfig,
) -> PyResult<Vec<f32>> {
    let decision = tree
        .node(node_id)?
        .kind
        .decision()
        .ok_or_else(|| PyRuntimeError::new_err("node is not decision"))?;
    let mut legal_alphas = Vec::new();
    for action in 0..config.action_size {
        if decision.legal_mask[action] {
            let alpha = decision_edge_posterior(tree, node_id, action, game, config)?;
            legal_alphas.push((action, alpha));
        }
    }
    if legal_alphas.is_empty() {
        return Err(PyValueError::new_err("decision node has no legal actions"));
    }

    let mut counts = vec![0usize; config.action_size];
    for _ in 0..config.posterior_best_samples {
        let mut best_action = legal_alphas[0].0;
        let mut best_utility = f32::NEG_INFINITY;
        for (action, alpha) in &legal_alphas {
            let phi = sample_dirichlet3(*alpha, &mut tree.rng)?;
            let utility = phi[2] - phi[0];
            if utility > best_utility {
                best_utility = utility;
                best_action = *action;
            }
        }
        counts[best_action] += 1;
    }

    let denom = config.posterior_best_samples as f32;
    Ok(counts.into_iter().map(|count| count as f32 / denom).collect())
}

fn retained_nodes(tree: &Tree) -> Vec<NodeId> {
    let mut out = Vec::new();
    let mut queue = VecDeque::new();
    queue.push_back(tree.root);
    while let Some(node_id) = queue.pop_front() {
        if !tree.has_node(node_id) {
            continue;
        }
        out.push(node_id);
        let node = tree.node(node_id).expect("node checked");
        match &node.kind {
            NodeKind::Decision(data) => {
                for edge in &data.edges {
                    if let Some(child) = edge.child {
                        queue.push_back(child);
                    }
                }
            }
            NodeKind::Categorical(data) => {
                for edge in &data.outcomes {
                    queue.push_back(edge.child);
                }
            }
            NodeKind::Terminal(_) => {}
        }
    }
    out
}

fn is_exportable_decision(tree: &Tree, node_id: NodeId) -> PyResult<bool> {
    let node = tree.node(node_id)?;
    let NodeKind::Decision(decision) = &node.kind else {
        return Ok(false);
    };
    if !matches!(decision.eval_status, DecisionEvalStatus::Expanded) {
        return Ok(false);
    }
    if node.c_v.is_none() {
        return Ok(false);
    }
    let has_child_evidence = decision
        .edges
        .iter()
        .zip(decision.legal_mask.iter())
        .any(|(edge, legal)| *legal && edge.r_count > 0);
    Ok(has_child_evidence)
}

fn reset_subtree_parent_and_depth(
    tree: &mut Tree,
    node_id: NodeId,
    parent: Option<NodeId>,
    parent_link: Option<ParentLink>,
    depth: u32,
) -> PyResult<()> {
    {
        let node = tree.node_mut(node_id)?;
        node.parent = parent;
        node.parent_link = parent_link;
        node.depth = depth;
    }
    let children = {
        let node = tree.node(node_id)?;
        match &node.kind {
            NodeKind::Decision(data) => data
                .edges
                .iter()
                .enumerate()
                .filter_map(|(action, edge)| {
                    edge.child.map(|child| {
                        (
                            child,
                            Some(ParentLink::DecisionAction { action }),
                        )
                    })
                })
                .collect::<Vec<_>>(),
            NodeKind::Categorical(data) => data
                .outcomes
                .iter()
                .map(|edge| {
                    (
                        edge.child,
                        Some(ParentLink::CategoricalOutcome {
                            outcome_id: edge.outcome_id,
                        }),
                    )
                })
                .collect::<Vec<_>>(),
            NodeKind::Terminal(_) => Vec::new(),
        }
    };
    for (child, link) in children {
        reset_subtree_parent_and_depth(tree, child, Some(node_id), link, depth + 1)?;
    }
    Ok(())
}

fn argmax_masked(values: &[f32], mask: &[bool]) -> PyResult<usize> {
    let mut best_idx = None;
    let mut best_value = f32::NEG_INFINITY;
    for (idx, (value, legal)) in values.iter().zip(mask).enumerate() {
        if *legal && *value > best_value {
            best_value = *value;
            best_idx = Some(idx);
        }
    }
    best_idx.ok_or_else(|| PyValueError::new_err("no legal action"))
}

fn sample_weighted_index(weights: &[f32], rng: &mut ChaCha20Rng) -> PyResult<usize> {
    let sanitized: Vec<f32> = weights
        .iter()
        .map(|w| if w.is_finite() && *w > 0.0 { *w } else { 0.0 })
        .collect();
    if sanitized.iter().all(|w| *w == 0.0) {
        return Err(PyValueError::new_err("cannot sample from all-zero weights"));
    }
    let dist = WeightedIndex::new(&sanitized)
        .map_err(|err| PyValueError::new_err(format!("invalid weights: {err}")))?;
    Ok(dist.sample(rng))
}

fn sample_dirichlet3(alpha: [f32; 3], rng: &mut ChaCha20Rng) -> PyResult<[f32; 3]> {
    let mut draws = [0.0; 3];
    let mut sum = 0.0;
    for i in 0..3 {
        if !alpha[i].is_finite() || alpha[i] <= 0.0 {
            return Err(PyValueError::new_err(
                "Dirichlet alpha must be finite and strictly positive",
            ));
        }
        let gamma = Gamma::new(alpha[i] as f64, 1.0)
            .map_err(|err| PyValueError::new_err(format!("invalid gamma alpha: {err}")))?;
        draws[i] = gamma.sample(rng) as f32;
        sum += draws[i];
    }
    if sum <= 0.0 || !sum.is_finite() {
        return Err(PyRuntimeError::new_err("Dirichlet sample failed"));
    }
    Ok([draws[0] / sum, draws[1] / sum, draws[2] / sum])
}

fn validate_positive_alpha(alpha: [f32; 3], debug: bool) -> PyResult<()> {
    if debug && alpha.iter().any(|x| !x.is_finite() || *x <= 0.0) {
        return Err(PyValueError::new_err(
            "terminal alpha must be finite and strictly positive",
        ));
    }
    Ok(())
}

fn validate_categorical_outcomes(
    outcomes: &[CategoricalOutcome],
    debug: bool,
) -> PyResult<()> {
    if outcomes.is_empty() {
        return Err(PyValueError::new_err(
            "categorical nodes must have at least one outcome",
        ));
    }
    if debug {
        let sum: f32 = outcomes.iter().map(|outcome| outcome.probability).sum();
        if outcomes
            .iter()
            .any(|outcome| !outcome.probability.is_finite() || outcome.probability < 0.0)
            || (sum - 1.0).abs() > 1.0e-4
        {
            return Err(PyValueError::new_err(
                "categorical probabilities must be nonnegative and sum to 1",
            ));
        }
    }
    Ok(())
}

fn mix_seed(seed: u64, tree_id: u64, generation: u32) -> u64 {
    let mut x = seed ^ tree_id.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= (generation as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn numpy_array<T: ToPyObject>(
    py: Python<'_>,
    data: Vec<T>,
    dtype_name: &str,
    shape: &[usize],
) -> PyResult<PyObject> {
    let np = PyModule::import(py, "numpy")?;
    let dtype = np.getattr(dtype_name)?;
    let py_data = data.to_object(py);
    let arr = np.call_method1("asarray", (py_data, dtype))?;
    let shape_tuple = PyTuple::new(py, shape.iter().copied());
    Ok(arr.call_method1("reshape", (shape_tuple,))?.into_py(py))
}

fn numpy_flat_u8(py: Python<'_>, obj: &PyAny) -> PyResult<(Vec<usize>, Vec<u8>)> {
    numpy_flat(py, obj, "uint8")
}

fn numpy_flat_u64(py: Python<'_>, obj: &PyAny) -> PyResult<(Vec<usize>, Vec<u64>)> {
    numpy_flat(py, obj, "uint64")
}

fn numpy_flat_u32(py: Python<'_>, obj: &PyAny) -> PyResult<(Vec<usize>, Vec<u32>)> {
    numpy_flat(py, obj, "uint32")
}

fn numpy_flat_i32(py: Python<'_>, obj: &PyAny) -> PyResult<(Vec<usize>, Vec<i32>)> {
    numpy_flat(py, obj, "int32")
}

fn numpy_flat_f32(py: Python<'_>, obj: &PyAny) -> PyResult<(Vec<usize>, Vec<f32>)> {
    numpy_flat(py, obj, "float32")
}

fn numpy_flat<T>(py: Python<'_>, obj: &PyAny, dtype_name: &str) -> PyResult<(Vec<usize>, Vec<T>)>
where
    for<'a> T: FromPyObject<'a>,
{
    let np = PyModule::import(py, "numpy")?;
    let dtype = np.getattr(dtype_name)?;
    let arr = np.call_method1("asarray", (obj, dtype))?;
    let shape = arr.getattr("shape")?.extract::<Vec<usize>>()?;
    let flat = arr.call_method0("ravel")?.call_method0("tolist")?;
    Ok((shape, flat.extract::<Vec<T>>()?))
}

fn optional_tree_ids(py: Python<'_>, tree_ids: Option<&PyAny>) -> PyResult<Option<Vec<TreeId>>> {
    match tree_ids {
        Some(obj) if !obj.is_none() => Ok(Some(numpy_flat_u64(py, obj)?.1)),
        _ => Ok(None),
    }
}

#[pymodule]
fn _dqaz(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<SearchConfig>()?;
    m.add_class::<SearchEngine>()?;
    m.add_class::<EvalBatch>()?;
    m.add_class::<SearchResults>()?;
    m.add_class::<SearchTargets>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn align_wdl_flips_between_players() {
        let game = GameKind::ToyDeterministic;
        let from = State::ToyDeterministic {
            depth: 1,
            player: 1,
            last_action: 0,
        };
        let to = State::ToyDeterministic {
            depth: 0,
            player: 0,
            last_action: 0,
        };
        assert_eq!(game.align_wdl(&from, &to, [1.0, 2.0, 3.0]), [3.0, 2.0, 1.0]);
    }

    #[test]
    fn toy_categorical_node_has_normalized_outcomes() {
        let game = GameKind::ToyCategorical;
        let state = State::ToyCategorical {
            phase: 1,
            player: 0,
            outcome: 0,
        };
        let GameNodeKind::Categorical(outcomes) = game.node_kind(&state).unwrap() else {
            panic!("phase 1 toy categorical state must be categorical");
        };
        assert_eq!(outcomes.len(), 2);
        let total = outcomes.iter().map(|outcome| outcome.probability).sum::<f32>();
        assert!((total - 1.0).abs() < 1.0e-6);
        assert_eq!(outcomes[0].outcome_id, 0);
        assert_eq!(outcomes[1].outcome_id, 1);
    }

    #[test]
    fn dirichlet_sample_is_simplex_valued() {
        let mut rng = ChaCha20Rng::seed_from_u64(123);
        let sample = sample_dirichlet3([1.0, 2.0, 3.0], &mut rng).unwrap();
        let total = sample.iter().sum::<f32>();
        assert!((total - 1.0).abs() < 1.0e-5);
        assert!(sample.iter().all(|value| *value > 0.0));
    }
}
