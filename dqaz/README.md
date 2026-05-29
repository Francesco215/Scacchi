# dqaz

Standalone Rust/Python MVP search backend for the Dirichlet-Q AlphaZero posterior tree.

The crate is intentionally separate from the existing Scacchi Python package. It is registered as an editable local dependency in the root `uv.lock`; build or refresh it locally with:

```bash
env -u CONDA_PREFIX uv pip install -e dqaz
```
