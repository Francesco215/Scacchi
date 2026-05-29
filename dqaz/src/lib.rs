use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

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
        debug = false
    ))]
    fn new(
        action_size: usize,
        observation_shape: Vec<usize>,
        simulations_per_root: u32,
        posterior_best_samples: u32,
        kappa_n: f64,
        seed: u64,
        debug: bool,
    ) -> PyResult<Self> {
        if action_size == 0 {
            return Err(PyValueError::new_err("action_size must be positive"));
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
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "SearchConfig(action_size={}, observation_shape={:?}, \
             simulations_per_root={}, posterior_best_samples={}, \
             kappa_n={}, seed={}, debug={})",
            self.action_size,
            self.observation_shape,
            self.simulations_per_root,
            self.posterior_best_samples,
            self.kappa_n,
            self.seed,
            self.debug,
        )
    }
}

#[pyclass(module = "dqaz")]
struct TransitionBatch {
    #[pyo3(get)]
    token: u64,
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
struct SearchResults {}

#[pyclass(module = "dqaz")]
struct SearchTargets {}

#[pyclass(module = "dqaz")]
struct SearchEngine {
    #[allow(dead_code)]
    config: SearchConfig,
}

#[pymethods]
impl SearchEngine {
    #[new]
    fn new(config: SearchConfig) -> Self {
        Self { config }
    }

    #[pyo3(signature = (*_args, **_kwargs))]
    fn add_roots(&self, _args: &PyTuple, _kwargs: Option<&PyDict>) -> PyResult<PyObject> {
        Err(not_implemented())
    }

    #[pyo3(signature = (max_batch_size, pad_to=None))]
    fn request_transitions(
        &self,
        max_batch_size: usize,
        pad_to: Option<usize>,
    ) -> PyResult<Py<TransitionBatch>> {
        if max_batch_size == 0 {
            return Err(PyValueError::new_err("max_batch_size must be positive"));
        }
        if let Some(pad_to) = pad_to {
            if pad_to == 0 {
                return Err(PyValueError::new_err("pad_to must be positive"));
            }
        }
        Err(not_implemented())
    }

    #[pyo3(signature = (*_args, **_kwargs))]
    fn submit_transitions(&self, _args: &PyTuple, _kwargs: Option<&PyDict>) -> PyResult<()> {
        Err(not_implemented())
    }

    #[pyo3(signature = (tree_ids=None))]
    fn is_done(&self, tree_ids: Option<PyObject>) -> PyResult<bool> {
        let _ = tree_ids;
        Err(not_implemented())
    }

    fn stats(&self) -> PyResult<PyObject> {
        Err(not_implemented())
    }

    #[pyo3(signature = (tree_ids=None, commit="posterior_sample"))]
    fn finish(&self, tree_ids: Option<PyObject>, commit: &str) -> PyResult<Py<SearchResults>> {
        let _ = tree_ids;
        validate_commit(commit)?;
        Err(not_implemented())
    }

    #[pyo3(signature = (tree_ids=None))]
    fn export_targets(&self, tree_ids: Option<PyObject>) -> PyResult<Py<SearchTargets>> {
        let _ = tree_ids;
        Err(not_implemented())
    }

    fn advance_roots(&self, tree_ids: PyObject, actions: PyObject) -> PyResult<()> {
        let _ = (tree_ids, actions);
        Err(not_implemented())
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

fn not_implemented() -> PyErr {
    PyNotImplementedError::new_err(
        "dqaz Rust search has not been reimplemented for the fused Python env.step + NN boundary",
    )
}

#[pymodule]
fn _dqaz(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<SearchConfig>()?;
    m.add_class::<SearchEngine>()?;
    m.add_class::<TransitionBatch>()?;
    m.add_class::<SearchResults>()?;
    m.add_class::<SearchTargets>()?;
    Ok(())
}
