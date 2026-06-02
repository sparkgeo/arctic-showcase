# Prescient Arctic Showcase

## Description
This project is a showcase for using Prescient in the context of an arctic usecase.

## Development setup

This repo uses a [UV workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/) with a virtual root. Requires Python 3.12+.

### Install UV

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install dependencies

```bash
uv sync --all-packages --group dev
```

This installs all workspace members (under `src/`) plus the dev tools.

### Common commands

```bash
uv run pytest                # Run tests
uv run ruff check .          # Lint
uv run ruff format .         # Format
uv run mypy src/             # Type check
uv run jupyter lab           # Start Jupyter Lab
```

### Adding dependencies

```bash
# Add a runtime dependency (available to all modules)
uv add <package>

# Add a dev-only dependency
uv add --group dev <package>

# Add a dependency scoped to a specific module
uv add --package <module_name> <package>
```

> **Note:** Never use `pip install` — always use `uv add` or `uv sync` to keep the lockfile in sync.