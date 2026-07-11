#!/usr/bin/env bash
# One-shot environment setup: clones LIBERO, installs it into the project's
# uv-managed venv, and pre-seeds its config so first import never blocks on
# an interactive prompt (important for unattended vast.ai boxes).
#
# Usage:
#   scripts/setup_libero.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# A stray VIRTUAL_ENV from an unrelated project silently redirects `uv pip`
# installs into the wrong venv. Always start clean.
unset VIRTUAL_ENV || true

LIBERO_DIR="third_party/LIBERO"
if [ ! -d "$LIBERO_DIR" ]; then
    echo "Cloning LIBERO..."
    git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git "$LIBERO_DIR"
fi

echo "Installing LIBERO (editable, no deps -- we manage the stack via uv)..."
# LIBERO's setup.py has no __init__.py at the top-level `libero/` package dir,
# which makes the default PEP 660 editable install register an empty package
# map. `editable_mode=compat` falls back to a plain sys.path insertion, which
# works fine since `libero.libero` etc. resolve as implicit namespace packages.
uv pip install -e "$LIBERO_DIR" --no-deps --config-settings editable_mode=compat

echo "Installing LIBERO's two hard runtime deps at the API versions its code expects..."
# LIBERO's own requirements.txt pins numpy==1.22.4 / robosuite==1.4.0 / etc,
# which would clobber our modern stack if installed with full deps. bddl and
# robomimic are the two packages LIBERO's code actually imports at runtime
# that aren't already in our stack, so we pin just those, with --no-deps.
uv pip install "bddl==1.0.1" "robomimic==0.2.0" "gym" --no-deps

echo "Seeding non-interactive LIBERO config..."
export LIBERO_CONFIG_PATH="$REPO_ROOT/.libero_config"
mkdir -p "$LIBERO_CONFIG_PATH"
mkdir -p "$REPO_ROOT/data/libero_datasets"
cat > "$LIBERO_CONFIG_PATH/config.yaml" <<EOF
benchmark_root: $REPO_ROOT/$LIBERO_DIR/libero/libero
bddl_files: $REPO_ROOT/$LIBERO_DIR/libero/libero/bddl_files
init_states: $REPO_ROOT/$LIBERO_DIR/libero/libero/init_files
datasets: $REPO_ROOT/data/libero_datasets
assets: $REPO_ROOT/$LIBERO_DIR/libero/libero/assets
EOF

echo "Verifying install..."
LIBERO_CONFIG_PATH="$LIBERO_CONFIG_PATH" .venv/bin/python -c "
from libero.libero import benchmark
bm_dict = benchmark.get_benchmark_dict()
print('LIBERO OK. Available suites:', list(bm_dict.keys()))
"

echo ""
echo "Done. Add this to your shell profile (or source .env) so LIBERO_CONFIG_PATH"
echo "is set in every session, including on the vast.ai box:"
echo "  export LIBERO_CONFIG_PATH=\"$LIBERO_CONFIG_PATH\""
echo ""
echo "Next: download a task-suite subset with scripts/download_libero_data.py"
