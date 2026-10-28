#!/usr/bin/env bash
# nvllm — one-click updater
#
# Updates nvllm in place. Update sources, tried in order:
#   1. the recorded local source (repo checkout / release staging) if it still
#      has a pyproject.toml, else the recorded git URL — the dev/repo path;
#   2. otherwise a **release tarball** (the Ollama-style small source bundle):
#      download nvllm-<ver> → extract → run its bundled install.sh (idempotent
#      upgrade: venv + pip vllm==pin + nvllm). This is the path that makes
#      `update` work for `curl|bash` installs, whose staging dir is not durable.
# Force a release with --version V (default: 0.6.0). Then refreshes the
# ~/.local/bin symlink and restarts the systemd user service if it is active.
#
# vLLM stays on the pinned version by design (the pin is what was validated on
# DGX Spark); pass --upgrade-vllm to follow upstream, or --vllm-version V to
# pin a specific one.
#
# Usage:
#   scripts/update.sh [options]
#     --version V         update from release V (default: 0.6.0) — forces the release path
#     --upgrade-vllm      upgrade vLLM to the latest release (otherwise keep the pin)
#     --vllm-version V    install/pin vLLM to exactly V
#     --dry-run           print the steps without executing
#     -h, --help          show this help
#
# Environment:
#   NVLLM_HOME            install root (default: $HOME/.nvllm)
#   NVLLM_VERSION         default release version for the release path (default: 0.6.0)
#   NVLLM_RELEASE_BASE    release download base (default: GitHub releases of FaberEli/NvLLM)

set -euo pipefail

ROOT="${NVLLM_HOME:-$HOME/.nvllm}"
UPGRADE_VLLM=0
VLLM_VERSION=""
DRY_RUN=0
RELEASE_VERSION="${NVLLM_VERSION:-0.6.0}"
WANT_RELEASE=0

usage() { sed -n '2,29p' "$0" | sed 's/^# \{0,1\}//'; }

log()  { printf '\033[1;34m[nvllm]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[nvllm] warning:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[nvllm] error:\033[0m %s\n' "$*" >&2; exit 1; }

run() {
  if [ "$DRY_RUN" -eq 1 ]; then
    printf '    (dry-run) %s\n' "$*"
  else
    "$@"
  fi
}

# pip_install ARGS... — route through the venv's pip, or uv pip when absent
pip_install() {
  if [ -x "$ROOT/venv/bin/pip" ]; then
    run "$ROOT/venv/bin/pip" install "$@"
  elif command -v uv >/dev/null 2>&1; then
    run uv pip install --python "$ROOT/venv/bin/python" "$@"
  else
    die "no pip available in $ROOT/venv (and no uv fallback)"
  fi
}

# release_update VER — the Ollama-style update path: download the small source
# tarball for VER, extract it, and run its bundled install.sh (an idempotent
# upgrade: venv + pip vllm==pin + nvllm + symlink + install.json). This is the
# path `curl|bash` installs use to update, since their staging dir is not
# durable; --version V forces it for any install.
release_update() {
  local ver="$1"
  local base="${NVLLM_RELEASE_BASE:-https://github.com/FaberEli/NvLLM/releases/download}"
  local url="$base/$ver/nvllm-$ver-linux-aarch64.tar.gz"
  local tmp; tmp="$(mktemp -d)"
  log "downloading release $ver: $url"
  if ! curl -fL --retry 2 -o "$tmp/nvllm.tar.gz" "$url" 2>/dev/null; then
    warn "single tarball not found — trying split parts (large releases)…"
    local n=0
    while :; do
      if curl -fL --retry 2 -o "$tmp/part-$n" "$(printf '%s.part-%02d' "$url" "$n")" 2>/dev/null; then
        n=$((n + 1))
      else
        break
      fi
    done
    [ "$n" -gt 0 ] || { rm -rf "$tmp"; die "release $ver not found (tried $url and its split parts)"; }
    : > "$tmp/nvllm.tar.gz"
    local i; for i in $(seq 0 $((n - 1))); do cat "$tmp/part-$i" >> "$tmp/nvllm.tar.gz"; done
  fi
  tar xzf "$tmp/nvllm.tar.gz" -C "$tmp"
  local stage="" d
  for d in "$tmp"/*/; do
    if [ -f "$d/install.sh" ] && [ -f "$d/pyproject.toml" ]; then stage="${d%/}"; break; fi
  done
  [ -n "$stage" ] || { rm -rf "$tmp"; die "release $ver layout unexpected (no install.sh + pyproject.toml)"; }
  # Preserve the existing vllm pin unless --vllm-version / --upgrade-vllm overrides it.
  local vargs=()
  if [ -n "$VLLM_VERSION" ]; then
    vargs+=(--vllm-version "$VLLM_VERSION")
  elif [ "$UPGRADE_VLLM" -ne 1 ] && [ -n "${META:-}" ]; then
    vargs+=(--vllm-version "$META")
  fi
  log "running the release's install.sh (idempotent in-place upgrade)…"
  bash "$stage/install.sh" --yes ${vargs[@]+"${vargs[@]}"}
  if [ "$UPGRADE_VLLM" -eq 1 ]; then
    log "upgrading vLLM to latest (unvalidated — check the changelog)…"
    "$ROOT/venv/bin/pip" install -U vllm
  fi
  rm -rf "$tmp"
}

while [ $# -gt 0 ]; do
  case "$1" in
    --version) RELEASE_VERSION="${2:?--version needs a value}"; WANT_RELEASE=1; shift 2 ;;
    --version=*) RELEASE_VERSION="${1#*=}"; WANT_RELEASE=1; shift ;;
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

# --------------------------------------------------------------------- nvllm ---
read -r SOURCE_DIR GIT_URL < <("$ROOT/venv/bin/python" -c '
import json, sys
m = json.load(open(sys.argv[1], encoding="utf-8"))
print(m.get("source_dir") or "", m.get("git_url") or "")
' "$ROOT/install.json")

# ---- choose update source: local/git for dev installs, release for tarball ----
RELEASE_PATH=0
if [ "$WANT_RELEASE" -eq 1 ]; then
  log "explicit --version → updating from release $RELEASE_VERSION"
  RELEASE_PATH=1
elif { [ -z "$SOURCE_DIR" ] || [ ! -f "$SOURCE_DIR/pyproject.toml" ]; } && [ -z "$GIT_URL" ]; then
  warn "no durable local source recorded (source_dir: '$SOURCE_DIR') — updating from release $RELEASE_VERSION"
  RELEASE_PATH=1
fi

if [ "$RELEASE_PATH" -eq 1 ]; then
  if [ "$DRY_RUN" -eq 1 ]; then
    log "(dry-run) would download release $RELEASE_VERSION, run its install.sh (venv + pip vllm + nvllm), refresh the symlink, and restart the service"
    exit 0
  fi
  release_update "$RELEASE_VERSION"
  # the release install.sh already refreshed the symlink + wrote install.json;
  # just (re)start the service if it is active.
  if systemctl --user is-active nvllm >/dev/null 2>&1; then
    log "restarting systemd user service nvllm"; systemctl --user restart nvllm
  fi
  log "done. Verify with:  nvllm --version"
  exit 0
fi

# ---- local / git source path (dev, repo installs) ----
if [ -n "$SOURCE_DIR" ] && [ -f "$SOURCE_DIR/pyproject.toml" ]; then
  log "updating nvllm from local source: $SOURCE_DIR"
  pip_target="$SOURCE_DIR"
elif [ -n "$GIT_URL" ]; then
  warn "source dir $SOURCE_DIR is gone — installing from git: $GIT_URL"
  pip_target="$GIT_URL"
else
  die "no usable install source recorded (no source_dir, no git_url); re-run scripts/install.sh or pass --version"
fi
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
run ln -sf "$ROOT/venv/bin/nvllm" "$LINK_DIR/nvllm"
# retire a legacy nvllm-serve symlink from pre-0.3.0 installs, if it points here
LEGACY_LINK="$LINK_DIR/nvllm-serve"
if [ -L "$LEGACY_LINK" ]; then
  target="$(readlink "$LEGACY_LINK" 2>/dev/null || true)"
  case "$target" in
    "$ROOT"/*|"$ROOT") run rm -f "$LEGACY_LINK" ;;
  esac
fi

# ---------------------------------------------------------------------- service ---
if [ "$DRY_RUN" -ne 1 ]; then
  if systemctl --user is-active nvllm >/dev/null 2>&1; then
    log "restarting systemd user service nvllm"
    systemctl --user restart nvllm
  elif systemctl --user is-active nvllm-serve >/dev/null 2>&1; then
    log "restarting legacy systemd user service nvllm-serve"
    systemctl --user restart nvllm-serve
  fi
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

log "done. Verify with:  nvllm --version"
