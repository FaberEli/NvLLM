#!/usr/bin/env bash
# nvllm-serve — one-click updater
#
# Reinstalls nvllm-serve from the recorded source (local dir, falling back to
# the recorded git URL if the dir is gone), refreshes the ~/.local/bin symlink,
# and restarts the systemd user service if it is active.
#
# vLLM stays on the pinned version by design (the pin is what was validated on
# DGX Spark); pass --upgrade-vllm to follow upstream, or --vllm-version V to
# pin a specific one.
#
# Usage:
#   scripts/update.sh [options]
#     --upgrade-vllm      upgrade vLLM to the latest release (otherwise keep the pin)
#     --vllm-version V    install/pin vLLM to exactly V
#     --dry-run           print the steps without executing
#     -h, --help          show this help
#
# Environment:
#   NVLLM_HOME            install root (default: $HOME/.nvllm)

set -euo pipefail

ROOT="${NVLLM_HOME:-$HOME/.nvllm}"
UPGRADE_VLLM=0
VLLM_VERSION=""
DRY_RUN=0

usage() { sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; }

log()  { printf '\033[1;34m[nvllm-serve]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[nvllm-serve] warning:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[nvllm-serve] error:\033[0m %s\n' "$*" >&2; exit 1; }

run() {
  if [ "$DRY_RUN" -eq 1 ]; then
    printf '    (dry-run) %s\n' "$*"
  else
    "$@"
  fi
}

while [ $# -gt 0 ]; do
  case "$1" in
    --upgrade-vllm) UPGRADE_VLLM=1; shift ;;
    --vllm-version) VLLM_VERSION="${2:?--vllm-version needs a value}"; shift 2 ;;
    --vllm-version=*) VLLM_VERSION="${1#*=}"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1 (try --help)" ;;
  esac
done

[ -f "$ROOT/install.json" ] || die "no install found at $ROOT — run scripts/install.sh first"
[ -x "$ROOT/venv/bin/python" ] || die "venv missing at $ROOT/venv — run scripts/install.sh first"

META="$("$ROOT/venv/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["vllm"])' "$ROOT/install.json")"
log "current install (from $ROOT/install.json): vllm==$META"

# ----------------------------------------------------------------- nvllm-serve ---
read -r SOURCE_DIR GIT_URL < <("$ROOT/venv/bin/python" -c '
import json, sys
m = json.load(open(sys.argv[1], encoding="utf-8"))
print(m.get("source_dir") or "", m.get("git_url") or "")
' "$ROOT/install.json")

if [ -n "$SOURCE_DIR" ] && [ -f "$SOURCE_DIR/pyproject.toml" ]; then
  log "updating nvllm-serve from local source: $SOURCE_DIR"
  pip_target="$SOURCE_DIR"
elif [ -n "$GIT_URL" ]; then
  warn "source dir $SOURCE_DIR is gone — installing from git: $GIT_URL"
  pip_target="$GIT_URL"
else
  die "no usable install source recorded (no source_dir, no git_url); run scripts/install.sh again"
fi

pip_install() {
  if [ -x "$ROOT/venv/bin/pip" ]; then
    run "$ROOT/venv/bin/pip" install "$@"
  elif command -v uv >/dev/null 2>&1; then
    run uv pip install --python "$ROOT/venv/bin/python" "$@"
  else
    die "no pip available in $ROOT/venv (and no uv fallback)"
  fi
}
pip_install "$pip_target"

# ----------------------------------------------------------------------- vllm ---
if [ -n "$VLLM_VERSION" ]; then
  log "pinning vLLM to vllm==$VLLM_VERSION"
  pip_install "vllm==$VLLM_VERSION"
elif [ "$UPGRADE_VLLM" -eq 1 ]; then
  log "upgrading vLLM to the latest release (unvalidated — check vLLM changelog for breaking changes)"
  pip_install -U vllm
else
  log "keeping vLLM pinned: vllm==$META (use --upgrade-vllm to follow upstream)"
  pip_install "vllm==$META"
fi

# --------------------------------------------------------------------- symlink ---
LINK_DIR="$HOME/.local/bin"
run mkdir -p "$LINK_DIR"
run ln -sf "$ROOT/venv/bin/nvllm-serve" "$LINK_DIR/nvllm-serve"

# ---------------------------------------------------------------------- service ---
if [ "$DRY_RUN" -ne 1 ] && systemctl --user is-active nvllm-serve >/dev/null 2>&1; then
  log "restarting systemd user service nvllm-serve"
  systemctl --user restart nvllm-serve
fi

# ------------------------------------------------------------- record new vllm ---
if [ "$DRY_RUN" -ne 1 ]; then
  NEW_VLLM="$("$ROOT/venv/bin/pip" show vllm 2>/dev/null | awk '/^Version/{print $2}' || true)"
  [ -z "$NEW_VLLM" ] && NEW_VLLM="$("$ROOT/venv/bin/python" -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo "$META")"
  "$ROOT/venv/bin/python" - "$ROOT/install.json" "$NEW_VLLM" <<'PY'
import json, sys
path, ver = sys.argv[1], sys.argv[2]
m = json.load(open(path, encoding="utf-8"))
m["vllm"] = ver
m["updated_at"] = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(timespec="seconds")
json.dump(m, open(path, "w", encoding="utf-8"), indent=2)
open(path, "a", encoding="utf-8").write("\n")
PY
  log "updated install.json (vllm==$NEW_VLLM)"
fi

log "done. Verify with:  nvllm-serve --version"
