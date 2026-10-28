#!/usr/bin/env bash
# nvllm — one-click installer
#
# Installs the nvllm CLI + the vLLM engine (pinned to the validated baseline)
# into a self-contained root directory (default: ~/.nvllm), and links
# `nvllm` into ~/.local/bin. Uninstall cleanly with scripts/uninstall.sh.
#
# Usage:
#   scripts/install.sh [options]
#     --vllm-version V    pin vLLM to V (default: 0.31.0, validated on DGX Spark)
#     --port P            service port recorded for the systemd unit (default: 8004)
#     --with-service      install + enable a systemd *user* service (autostart, restart-on-crash)
#     --python PATH       interpreter to use (default: python3)
#     --dry-run           print the steps without executing
#     --yes               skip confirmation prompts
#     -h, --help          show this help
#
# Environment:
#   NVLLM_HOME            install root (default: $HOME/.nvllm)
#   NVLLM_PORT            service port (default: 8004)
#   NVLLM_VLLM_VERSION    vLLM version pin (default: 0.31.0)
#
# The script is idempotent: re-running it upgrades in place.

set -euo pipefail

# ---------------------------------------------------------------- defaults ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ROOT="${NVLLM_HOME:-$HOME/.nvllm}"
VLLM_VERSION="${NVLLM_VLLM_VERSION:-0.31.0}"
PORT="${NVLLM_PORT:-8004}"
PYTHON_OVERRIDE="${PYTHON:-}"
WITH_SERVICE=0
DRY_RUN=0
ASSUME_YES=0

# ------------------------------------------------------------------ usage ---
usage() { sed -n '2,23p' "$0" | sed 's/^# \{0,1\}//'; }

log()  { printf '\033[1;34m[nvllm]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[nvllm] warning:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[nvllm] error:\033[0m %s\n' "$*" >&2; exit 1; }

# run [CMD...] — prints the command; skips execution under --dry-run
run() {
  if [ "$DRY_RUN" -eq 1 ]; then
    printf '    (dry-run) %s\n' "$*"
  else
    "$@"
  fi
}

# ------------------------------------------------------------- parse args ---
while [ $# -gt 0 ]; do
  case "$1" in
    --vllm-version) VLLM_VERSION="${2:?--vllm-version needs a value}"; shift 2 ;;
    --vllm-version=*) VLLM_VERSION="${1#*=}"; shift ;;
    --port) PORT="${2:?--port needs a value}"; shift 2 ;;
    --port=*) PORT="${1#*=}"; shift ;;
    --python) PYTHON_OVERRIDE="${2:?--python needs a value}"; shift 2 ;;
    --python=*) PYTHON_OVERRIDE="${1#*=}"; shift ;;
    --with-service) WITH_SERVICE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --yes) ASSUME_YES=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1 (try --help)" ;;
  esac
done

# ------------------------------------------- prebuilt release detection ----
# Self-contained release (scripts/build-release.sh output): install.sh sits
# NEXT TO a prebuilt venv/ that already contains vLLM + nvllm. In that case the
# install is just "move it into place" — interpreter selection and pip are
# skipped entirely (that is the whole point of the tarball).
# (检测锚点是 install.sh 的同级目录，不是 REPO_ROOT：仓库布局里 venv/ 在
#  scripts/ 的同级上级，发行布局里 venv/ 与 install.sh 同级。)
PREBUILT=0
PREBUILT_SRC=""
if [ -x "$SCRIPT_DIR/venv/bin/python" ]; then
  PREBUILT=1
  PREBUILT_SRC="$SCRIPT_DIR"
  # 防呆：落位是 rm -rf + cp -a，安装根与发行目录互为包含时会自我吞噬
  case "$ROOT/" in
    "$PREBUILT_SRC"/*) die "install root ($ROOT) is inside the release dir ($PREBUILT_SRC) — set NVLLM_HOME elsewhere" ;;
  esac
  case "$PREBUILT_SRC/" in
    "$ROOT"/*) die "release dir ($PREBUILT_SRC) is inside the install root ($ROOT) — set NVLLM_HOME elsewhere" ;;
  esac
fi

# ----------------------------------------------------------------- precheck ---
[ "$(uname -s)" = "Linux" ] || die "this installer targets Linux (validated on DGX Spark GB10)"
if [ "$(uname -m)" != "aarch64" ]; then
  warn "not aarch64 — the validated baseline is DGX Spark (GB10); continuing anyway"
fi

if [ "$PREBUILT" -eq 1 ]; then
  PYTHON="$PREBUILT_SRC/venv/bin/python"
  PYVER="$("$PYTHON" -c 'import platform; print(platform.python_version())')"
  log "prebuilt release — bundled venv (python $PYVER); pip steps will be skipped"
else
  # Pick an interpreter. vLLM 0.31 ships wheels for Python 3.10-3.13; a system
  # `python3` on 3.14+ would fail to install the engine, so prefer 3.11-3.13.
  if [ -n "$PYTHON_OVERRIDE" ]; then
    PYTHON="$PYTHON_OVERRIDE"
  else
    PYTHON=""
    for cand in python3.13 python3.12 python3.11; do
      if command -v "$cand" >/dev/null 2>&1; then
        PYTHON="$cand"
        break
      fi
    done
    if [ -z "$PYTHON" ]; then
      PYTHON="python3"
    fi
  fi

  command -v "$PYTHON" >/dev/null 2>&1 || die "python interpreter not found: $PYTHON"
  "$PYTHON" -c 'import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] <= (3, 13) else 1)' \
    || warn "interpreter $PYTHON is not in the validated range 3.11-3.13; vLLM wheels may be unavailable"
  PYVER="$("$PYTHON" -c 'import platform; print(platform.python_version())')"
  log "interpreter: $PYTHON ($PYVER)"
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  warn "nvidia-smi not found — no CUDA GPU visible; vLLM will install but cannot serve"
fi

if ! curl -sI -m 8 https://pypi.org/simple/ >/dev/null 2>&1; then
  warn "pypi.org unreachable — if you use a mirror, set PIP_INDEX_URL and re-run"
fi

# -------------------------------------------------------------------- venv ---
log "install root: $ROOT"
if [ "$PREBUILT" -eq 1 ]; then
  # 预构建 venv 直接落位（同文件系统时这是一次纯拷贝；升级时先清旧的）
  run mkdir -p "$ROOT"
  if [ -e "$ROOT/venv" ]; then
    log "existing venv found — replacing with the bundled one (upgrade)"
    run rm -rf "$ROOT/venv"
  fi
  run cp -a "$PREBUILT_SRC/venv" "$ROOT/venv"
  [ -f "$PREBUILT_SRC/LICENSE" ]  && run cp -f "$PREBUILT_SRC/LICENSE"  "$ROOT/LICENSE"
  [ -f "$PREBUILT_SRC/README.md" ] && run cp -f "$PREBUILT_SRC/README.md" "$ROOT/README.md"
elif [ -x "$ROOT/venv/bin/python" ]; then
  log "existing venv found — upgrading in place"
else
  # Prefer stdlib venv (ships pip). Fall back to `uv venv` on distros that lack
  # python3-venv/ensurepip.
  if "$PYTHON" -m venv --help >/dev/null 2>&1; then
    run "$PYTHON" -m venv "$ROOT/venv"
  elif command -v uv >/dev/null 2>&1; then
    run uv venv "$ROOT/venv" --python "$PYTHON"
  else
    die "could not create a venv: neither '$PYTHON -m venv' nor 'uv' is available"
  fi
fi

# pip_install ARGS... — route through the venv's pip, or uv pip when absent
pip_install() {
  local args=("$@")
  if [ -x "$ROOT/venv/bin/pip" ]; then
    run "$ROOT/venv/bin/pip" install "${args[@]}"
  elif command -v uv >/dev/null 2>&1; then
    run uv pip install --python "$ROOT/venv/bin/python" "${args[@]}"
  else
    die "no pip available in $ROOT/venv (and no uv fallback)"
  fi
}

# ------------------------------------------------------------- install pkgs ---
if [ "$PREBUILT" -eq 0 ]; then
  # Locate the nvllm package source (pyproject.toml + nvllm_service/). The
  # release layout keeps install.sh at the top level next to the source; the repo
  # layout keeps it under scripts/. Accept whichever sibling holds the package.
  SRC_ROOT=""
  for cand in "$SCRIPT_DIR" "$REPO_ROOT"; do
    if [ -f "$cand/pyproject.toml" ] && [ -d "$cand/nvllm_service" ]; then
      SRC_ROOT="$(cd "$cand" && pwd)"; break
    fi
  done
  [ -n "$SRC_ROOT" ] || die "could not find the nvllm package source (pyproject.toml + nvllm_service/) near $SCRIPT_DIR"

  log "installing vLLM engine: vllm==$VLLM_VERSION (validated baseline; pulls matching torch+CUDA — needs network)"
  pip_install "vllm==$VLLM_VERSION"

  log "installing nvllm from: $SRC_ROOT"
  pip_install "$SRC_ROOT"
fi

# -------------------------------------------------------------------- smoke ---
run "$ROOT/venv/bin/nvllm" --version
if [ "$DRY_RUN" -ne 1 ]; then
  [ -e "$ROOT/venv/bin/vllm" ] || die "vllm binary missing after install (vllm install failed?)"
fi

# ------------------------------------------------------------------- symlink ---
LINK_DIR="$HOME/.local/bin"
run mkdir -p "$LINK_DIR"
run ln -sf "$ROOT/venv/bin/nvllm" "$LINK_DIR/nvllm"
# retire the legacy nvllm-serve symlink if it belongs to this install
LEGACY_LINK="$LINK_DIR/nvllm-serve"
if [ -L "$LEGACY_LINK" ]; then
  target="$(readlink "$LEGACY_LINK" 2>/dev/null || true)"
  case "$target" in
    "$ROOT"/*|"$ROOT") run rm -f "$LEGACY_LINK" ;;
  esac
fi
case ":$PATH:" in
  *":$LINK_DIR:"*) : ;;
  *) warn "$LINK_DIR is not on your PATH — add it: export PATH=\"$LINK_DIR:\$PATH\"" ;;
esac

# ------------------------------------------------------------------ systemd ---
# retire the legacy nvllm-serve unit if present (it would double-start)
LEGACY_UNIT="$HOME/.config/systemd/user/nvllm-serve.service"
if [ -f "$LEGACY_UNIT" ]; then
  systemctl --user disable --now nvllm-serve 2>/dev/null || true
  rm -f "$LEGACY_UNIT"
fi
UNIT_FILE="$HOME/.config/systemd/user/nvllm.service"
if [ "$WITH_SERVICE" -eq 1 ]; then
  log "writing systemd user unit: $UNIT_FILE"
  if [ "$DRY_RUN" -ne 1 ]; then
    mkdir -p "$HOME/.config/systemd/user"
    cat > "$UNIT_FILE" <<EOF
[Unit]
Description=nvllm — resident LLM inference service (vLLM backend)
After=network-online.target
Wants=network-online.target

[Service]
ExecStart=$ROOT/venv/bin/nvllm serve --port $PORT
Environment=NVLLM_LOG_DIR=$ROOT/logs
Environment=NVLLM_STATE_DIR=$ROOT/state
Environment=NVLLM_WORKER_MAX_JOBS=2
WorkingDirectory=$ROOT
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF
    systemctl --user daemon-reload
    systemctl --user enable --now nvllm
    log "systemd user service enabled and started"
    echo "    Tip: run 'loginctl enable-linger $USER' so it starts at boot without a login session."
  fi
fi

# -------------------------------------------------------------- install.json ---
GIT_URL=""
if [ -d "$REPO_ROOT/.git" ] && command -v git >/dev/null 2>&1; then
  GIT_URL="$(git -C "$REPO_ROOT" remote get-url origin 2>/dev/null || true)"
fi
if [ "$DRY_RUN" -ne 1 ]; then
  PYTHON="$PYTHON" ROOT="$ROOT" REPO_ROOT="$REPO_ROOT" VLLM_VERSION="$VLLM_VERSION" \
  PORT="$PORT" WITH_SERVICE="$WITH_SERVICE" GIT_URL="$GIT_URL" PYVER="$PYVER" \
  PREBUILT="$PREBUILT" \
  "$PYTHON" - <<'PY'
import json, os, datetime
meta = {
    "root": os.environ["ROOT"],
    "source_dir": os.environ["REPO_ROOT"],
    "git_url": os.environ.get("GIT_URL") or None,
    "vllm": os.environ["VLLM_VERSION"],
    "python": os.environ["PYVER"],
    "port": int(os.environ["PORT"]),
    "service": os.environ["WITH_SERVICE"] == "1",
    "prebuilt": os.environ.get("PREBUILT") == "1",
    "installed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
}
with open(os.path.join(os.environ["ROOT"], "install.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)
    f.write("\n")
print(f"    install.json written (vllm={meta['vllm']}, service={meta['service']})")
PY
fi

# -------------------------------------------------------------------- summary ---
cat <<EOF

============================================================
  nvllm installed  (root: $ROOT)
  vLLM engine: $VLLM_VERSION   |   service port: $PORT
============================================================

Next steps
  1. Get an NVFP4 model into your HF cache, e.g.:
       pip install -U "huggingface_hub[cli]"   (or: pip install huggingface_hub)
       hf download nvidia/Llama-3.1-8B-Instruct-NVFP4
  2. Start the service (or: systemctl --user status nvllm):
       nvllm serve
  3. Verify:
       curl -s localhost:$PORT/api/tags | jq
       curl -s localhost:$PORT/v1/chat/completions -d '{
         "model": "nvidia/Llama-3.1-8B-Instruct-NVFP4",
         "messages": [{"role": "user", "content": "hello"}],
         "max_tokens": 32
       }'

DGX Spark (GB10) hints
  - First NVFP4 load JIT-compiles FlashInfer kernels; set NVLLM_WORKER_MAX_JOBS=2
    (the systemd unit already sets it) so nvcc is not OOM-killed.
  - Default gpu-memory-utilization is 0.5 — right for the shared 128G.

Uninstall:  scripts/uninstall.sh   (restores the environment; models are kept)
EOF
