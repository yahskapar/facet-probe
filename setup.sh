#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="facet-probe"
ENV_PREFIX=""
PYTHON_VERSION="3.11"
EXTRAS="dev"
DRY_RUN=0
YES=0
ALLOW_ACTIVE_CONDA=0

usage() {
  cat <<'EOF'
Usage:
  bash setup.sh conda [options]
  bash setup.sh uv [options]

Options:
  --name NAME             Conda environment name (default: facet-probe)
  --prefix PATH           Conda environment prefix. Overrides --name for conda.
  --python VERSION        Python version for uv venv (default: 3.11)
  --extras EXTRAS         Python extras to install, comma-separated (default: dev)
                           Use "base" or "none" for no extras.
                           Common values: dev, dev,hf,analysis,
                           dev,hf,analysis,models,providers,
                           dev,hf,analysis,irt,
                           dev,hf,analysis,models,providers,irt
  --yes, -y               Pass non-interactive yes to conda env creation/update
  --dry-run               Print commands without creating or modifying environments
  --allow-active-conda    Allow uv setup while CONDA_PREFIX is set
  --help, -h              Show this help

Examples:
  bash setup.sh conda
  conda activate facet-probe

  bash setup.sh uv
  source .venv/bin/activate
EOF
}

die() {
  printf 'setup.sh: error: %s\n' "$*" >&2
  exit 2
}

print_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
}

run_cmd() {
  print_cmd "$@"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    "$@"
  fi
}

require_command() {
  local name="$1"
  if [[ "$DRY_RUN" -eq 0 ]] && ! command -v "$name" >/dev/null 2>&1; then
    die "$name is not installed or not on PATH"
  fi
}

editable_spec() {
  if [[ -z "$EXTRAS" || "$EXTRAS" == "base" || "$EXTRAS" == "none" ]]; then
    printf '.'
  else
    printf '.[%s]' "$EXTRAS"
  fi
}

conda_env_exists() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    return 1
  fi
  if [[ -n "$ENV_PREFIX" ]]; then
    [[ -d "$ENV_PREFIX/conda-meta" ]]
    return
  fi
  conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"
}

setup_conda() {
  require_command conda
  local spec
  local target_args
  local activation_target
  spec="$(editable_spec)"
  if [[ -n "$ENV_PREFIX" ]]; then
    target_args=(--prefix "$ENV_PREFIX")
    activation_target="$ENV_PREFIX"
  else
    target_args=(--name "$ENV_NAME")
    activation_target="$ENV_NAME"
  fi

  if conda_env_exists; then
    local update_args=(conda env update "${target_args[@]}" --file environment.yml --prune)
    if [[ "$YES" -eq 1 ]]; then
      update_args+=(--yes)
    fi
    run_cmd "${update_args[@]}"
  else
    local create_args=(conda env create "${target_args[@]}" --file environment.yml)
    if [[ "$YES" -eq 1 ]]; then
      create_args+=(--yes)
    fi
    run_cmd "${create_args[@]}"
  fi

  run_cmd conda run "${target_args[@]}" python -m pip install -e "$spec"
  printf '\nActivate with:\n  conda activate %s\n' "$activation_target"
}

setup_uv() {
  require_command uv
  local spec
  spec="$(editable_spec)"

  if [[ -n "${CONDA_PREFIX:-}" && "$ALLOW_ACTIVE_CONDA" -eq 0 ]]; then
    if [[ "$DRY_RUN" -eq 1 ]]; then
      printf 'setup.sh: note: uv setup should normally be run outside an active conda env.\n' >&2
    else
      die "deactivate conda before uv setup, or pass --allow-active-conda"
    fi
  fi

  run_cmd uv venv --python "$PYTHON_VERSION" .venv
  run_cmd uv pip install --python "$REPO_ROOT/.venv/bin/python" -e "$spec"
  printf '\nActivate with:\n  source .venv/bin/activate\n'
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" || $# -eq 0 ]]; then
  usage
  exit 0
fi

MODE="$1"
shift

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)
      [[ $# -ge 2 ]] || die "--name requires a value"
      ENV_NAME="$2"
      shift 2
      ;;
    --prefix)
      [[ $# -ge 2 ]] || die "--prefix requires a value"
      ENV_PREFIX="$2"
      shift 2
      ;;
    --python)
      [[ $# -ge 2 ]] || die "--python requires a value"
      PYTHON_VERSION="$2"
      shift 2
      ;;
    --extras)
      [[ $# -ge 2 ]] || die "--extras requires a value"
      EXTRAS="$2"
      shift 2
      ;;
    --yes|-y)
      YES=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --allow-active-conda)
      ALLOW_ACTIVE_CONDA=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

cd "$REPO_ROOT"

case "$MODE" in
  conda)
    setup_conda
    ;;
  uv)
    setup_uv
    ;;
  *)
    die "first argument must be 'conda' or 'uv'"
    ;;
esac
