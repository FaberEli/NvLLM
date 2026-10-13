#!/usr/bin/env bash
# nvllm-serve — one-click uninstall + environment restore
#
# Stops the systemd user service (if installed), terminates the running
# nvllm-serve service and its `vllm serve` workers (matched strictly to THIS
# install root), removes the install root (venv + logs + metadata), and removes
# the ~/.local/bin/nvllm-serve symlink — restoring the environment to its
# pre-install state.
#
# User assets are KEPT by default: the HuggingFace model cache (your NVFP4
# weights), the FlashInfer JIT kernel cache, and torch caches. Pass --purge to
# delete those as well.
#
# Usage:
#   scripts/uninstall.sh [options]
#     --purge    also delete the HF model cache, FlashInfer JIT cache, torch caches
#     --yes      skip the confirmation prompt
#     -h, --help show this help
#
# Environment:
#   NVLLM_HOME            install root (default: $HOME/.nvllm)

set -euo pipefail

ROOT="${NVLLM_HOME:-$HOME/.nvllm}"
PURGE=0
ASSUME_YES=0

usage() { sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'; }

log()  { printf '\033[1;34m[nvllm-serve]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[nvllm-serve] warning:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[nvllm-serve] error:\033[0m %s\n' "$*" >&2; exit 1; }

while [ $# -gt 0 ]; do
  case "$1" in
    --purge) PURGE=1; shift ;;
    --yes) ASSUME_YES=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1 (try --help)" ;;
  esac
done

[ -e "$ROOT" ] || { warn "no install found at $ROOT — nothing to remove"; exit 0; }

# -------------------------------------------------------------------- service ---
if [ -f "$HOME/.config/systemd/user/nvllm-serve.service" ]; then
  log "disabling + stopping systemd user service"
  systemctl --user disable --now nvllm-serve 2>/dev/null || true
  rm -f "$HOME/.config/systemd/user/nvllm-serve.service"
  systemctl --user daemon-reload
fi

# -------------------------------------------------------------------- workers ---
log "stopping any running nvllm-serve / vllm workers under $ROOT"
# Match strictly on this install root's venv bin dir so a dev .venv or other
# Python installs are never touched. `pkill -f` sees the full command line.
PAT="$ROOT/venv/bin/"
pkill -f "$PAT" 2>/dev/null || true
sleep 1
if pkill -9 -f "$PAT" 2>/dev/null; then
  warn "some processes needed SIGKILL to stop"
fi

# ------------------------------------------------------------------ user assets ---
HF_CACHE="${HF_HUB_CACHE:-${HF_HOME:+$HF_HOME/hub}}"
[ -z "$HF_CACHE" ] && HF_CACHE="$HOME/.cache/huggingface"
declare -a KEPT=()
for d in "$HF_CACHE" "$HOME/.cache/flashinfer" "$HOME/.cache/torch" "$HOME/.cache/torch_extensions"; do
  [ -e "$d" ] && KEPT+=("$d")
done

if [ "$PURGE" -eq 1 ]; then
  log "purging user asset caches (model weights included!)"
  for d in "${KEPT[@]}"; do rm -rf "$d"; done
fi

# ------------------------------------------------------------------- confirm ---
if [ "$ASSUME_YES" -ne 1 ]; then
  echo "This will remove:"
  echo "  $ROOT   (venv: vLLM + torch + nvllm-serve, logs, install metadata)"
  echo "  $HOME/.local/bin/nvllm-serve   (symlink)"
  if [ "$PURGE" -eq 1 ]; then
    echo "  plus, with --purge:"
    for d in "${KEPT[@]}"; do echo "  $d"; done
  else
    echo
    echo "Kept (your assets):"
    for d in "${KEPT[@]}"; do
      sz="$(du -sh "$d" 2>/dev/null | cut -f1 || echo '?')"
      echo "  $d   ($sz)"
    done
    echo
    echo "Re-run with --purge to delete those too."
  fi
  printf 'Proceed? [y/N] '
  read -r ans
  case "$ans" in
    y|Y|yes|YES) : ;;
    *) die "aborted — nothing was removed" ;;
  esac
fi

# ----------------------------------------------------------------------- remove ---
# Remove the PATH symlink FIRST, while its target still exists, so we can tell
# it belongs to this install (a raw readlink works on the first hop even if the
# link later dangles).
LINK="$HOME/.local/bin/nvllm-serve"
if [ -L "$LINK" ]; then
  target="$(readlink "$LINK" 2>/dev/null || true)"
  case "$target" in
    "$ROOT"/*|"$ROOT")
      log "removing symlink: $LINK"
      rm -f "$LINK" ;;
    *)
      warn "symlink $LINK points elsewhere ($target) — leaving it alone" ;;
  esac
fi

log "removing install root: $ROOT"
rm -rf "$ROOT"

# -------------------------------------------------------------------- selfcheck ---
log "self-check"
if [ ! -e "$ROOT" ]; then
  log "OK: $ROOT is gone"
else
  die "removal failed: $ROOT still exists"
fi
if [ "$PURGE" -eq 0 ]; then
  for d in "${KEPT[@]}"; do
    [ -e "$d" ] && log "kept: $d"
  done
fi
log "uninstall complete — environment restored (models and caches kept)"
