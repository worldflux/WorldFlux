#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Fetch pinned upstream oracle artifacts for WorldFlux parity suites.

Usage:
  scripts/parity/fetch_oracles.sh \
    --oracle-root /root/oracles \
    --dreamer-commit <sha> \
    --tdmpc2-commit <sha> \
    [--copy-to /path/to/worldflux/artifacts/upstream]

Options:
  --oracle-root      Working directory for upstream clones and extracted artifacts.
  --dreamer-commit   Pinned commit for https://github.com/danijar/dreamerv3.
  --tdmpc2-commit    Pinned commit for https://github.com/nicklashansen/tdmpc2.
  --copy-to          Optional destination to mirror extracted artifacts.
  --dreamer-repo     Override DreamerV3 repo URL.
  --tdmpc2-repo      Override TD-MPC2 repo URL.
EOF
}

ORACLE_ROOT=""
DREAMER_COMMIT=""
TDMPC2_COMMIT=""
COPY_TO=""
DREAMER_REPO="https://github.com/danijar/dreamerv3.git"
TDMPC2_REPO="https://github.com/nicklashansen/tdmpc2.git"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --oracle-root)
      ORACLE_ROOT="${2:-}"
      shift 2
      ;;
    --dreamer-commit)
      DREAMER_COMMIT="${2:-}"
      shift 2
      ;;
    --tdmpc2-commit)
      TDMPC2_COMMIT="${2:-}"
      shift 2
      ;;
    --copy-to)
      COPY_TO="${2:-}"
      shift 2
      ;;
    --dreamer-repo)
      DREAMER_REPO="${2:-}"
      shift 2
      ;;
    --tdmpc2-repo)
      TDMPC2_REPO="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${ORACLE_ROOT}" || -z "${DREAMER_COMMIT}" || -z "${TDMPC2_COMMIT}" ]]; then
  echo "Missing required arguments." >&2
  usage >&2
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  echo "git is required but not found" >&2
  exit 1
fi

clone_or_update() {
  local repo_url="$1"
  local target_dir="$2"
  local commit="$3"

  if [[ ! -d "${target_dir}/.git" ]]; then
    git clone "${repo_url}" "${target_dir}"
  fi

  git -C "${target_dir}" fetch --tags --prune origin
  git -C "${target_dir}" checkout --detach "${commit}"
}

mkdir -p "${ORACLE_ROOT}"
ORACLE_ROOT="$(cd "${ORACLE_ROOT}" && pwd)"

DREAMER_REPO_DIR="${ORACLE_ROOT}/repos/dreamerv3"
TDMPC2_REPO_DIR="${ORACLE_ROOT}/repos/tdmpc2"
DREAMER_OUT_DIR="${ORACLE_ROOT}/artifacts/upstream/dreamerv3/scores"
TDMPC2_OUT_DIR="${ORACLE_ROOT}/artifacts/upstream/tdmpc2/results/tdmpc"

mkdir -p "${DREAMER_OUT_DIR}" "${TDMPC2_OUT_DIR}"

echo "[oracles] syncing DreamerV3 at ${DREAMER_COMMIT}"
clone_or_update "${DREAMER_REPO}" "${DREAMER_REPO_DIR}" "${DREAMER_COMMIT}"
cp -f \
  "${DREAMER_REPO_DIR}/scores/atari100k-dreamerv3.json.gz" \
  "${DREAMER_OUT_DIR}/atari100k-dreamerv3.json.gz"

echo "[oracles] syncing TD-MPC2 at ${TDMPC2_COMMIT}"
clone_or_update "${TDMPC2_REPO}" "${TDMPC2_REPO_DIR}" "${TDMPC2_COMMIT}"
rm -f "${TDMPC2_OUT_DIR}"/*.csv
cp -f "${TDMPC2_REPO_DIR}/results/tdmpc/"*.csv "${TDMPC2_OUT_DIR}/"

if [[ -n "${COPY_TO}" ]]; then
  COPY_TO="$(cd "${COPY_TO}" && pwd)"
  mkdir -p "${COPY_TO}/dreamerv3/scores" "${COPY_TO}/tdmpc2/results/tdmpc"
  cp -f \
    "${DREAMER_OUT_DIR}/atari100k-dreamerv3.json.gz" \
    "${COPY_TO}/dreamerv3/scores/atari100k-dreamerv3.json.gz"
  rm -f "${COPY_TO}/tdmpc2/results/tdmpc/"*.csv
  cp -f "${TDMPC2_OUT_DIR}/"*.csv "${COPY_TO}/tdmpc2/results/tdmpc/"
  echo "[oracles] mirrored extracted artifacts to ${COPY_TO}"
fi

echo "[oracles] Dreamer artifact: ${DREAMER_OUT_DIR}/atari100k-dreamerv3.json.gz"
echo "[oracles] TD-MPC2 artifact dir: ${TDMPC2_OUT_DIR}"
echo "[oracles] done"
