# dqaz

Standalone Rust/Python search backend for the Dirichlet-Q AlphaZero posterior tree.

The old Rust-owned toy-game adapter implementation has been removed. The crate
now exposes only the fused-boundary API shell while the backend is rebuilt
around Python-owned PGX `env.step` and neural network evaluation.

The crate is intentionally separate from the existing Scacchi Python package. It is registered as an editable local dependency in the root `uv.lock`; build or refresh it locally with:

```bash
env -u CONDA_PREFIX uv pip install -e dqaz
```
