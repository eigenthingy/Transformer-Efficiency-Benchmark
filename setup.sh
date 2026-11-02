#!/usr/bin/env bash
# Create the project virtual environment and install dependencies (macOS / Linux).
#
# By default the venv inherits the parent interpreter's site-packages so an
# existing torch build is reused instead of downloading a second copy.
#
#   ./setup.sh                          # reuse site-packages
#   ./setup.sh --isolated               # fully isolated venv (installs its own torch)
#   PYTHON=python3.12 ./setup.sh        # pick the interpreter
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
VENV="$ROOT/.venv"
ISOLATED=0
VERIFY=1

for arg in "$@"; do
  case "$arg" in
    --isolated) ISOLATED=1 ;;
    --no-verify) VERIFY=0 ;;
    -h|--help) sed -n '2,12p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

echo "Project : $ROOT"
command -v "$PYTHON" >/dev/null || { echo "interpreter '$PYTHON' not found" >&2; exit 1; }

if [ -x "$VENV/bin/python" ]; then
  echo "venv    : already present at .venv (reusing)"
else
  if [ "$ISOLATED" -eq 1 ]; then
    echo "venv    : creating .venv (isolated)"
    "$PYTHON" -m venv "$VENV"
  else
    echo "venv    : creating .venv (inheriting site-packages)"
    "$PYTHON" -m venv --system-site-packages "$VENV"
  fi
fi

VENV_PY="$VENV/bin/python"
echo
echo "Installing dependencies from requirements.txt ..."
"$VENV_PY" -m pip install --upgrade pip
"$VENV_PY" -m pip install -r "$ROOT/requirements.txt"

if [ "$VERIFY" -eq 1 ]; then
  echo
  echo "Verifying the environment ..."
  "$VENV_PY" "$ROOT/check_env.py"
fi

cat <<'EOF'

Done. Activate it with:

    source .venv/bin/activate

Then run a pipeline check (~5 min on CPU, writes to results/smoke/):

    python run_benchmark.py --smoke

EOF
