#!/usr/bin/env bash
# nvllm — build the release artifact (small: nvllm source + installer; the vLLM
# engine is pulled at install time, not bundled).
#
# Produces dist/nvllm-<ver>-linux-aarch64.tar.gz (a few MB — NO bundled venv):
#
#   nvllm-<ver>-linux-aarch64/
#     install.sh        installer — builds a venv, then `pip install vllm==<ver>` + nvllm (needs network)
#     pyproject.toml    the nvllm package manifest
#     nvllm_service/    nvllm source
#     scripts/          update.sh / uninstall.sh / bootstrap.sh (post-install helpers)
#     LICENSE  README.md
#
# Unlike the 0.4.0 "self-contained" release (which bundled a 3-4 GB venv and had
# to be split across two GitHub assets to fit the 2 GB single-asset cap), this
# artifact is tiny and uploads as a single asset. The heavy part (vLLM + torch +
# CUDA, ~3-4 GB) is `pip install`-ed at install time — the same "download a small
# bootstrap, pull the big bits separately" shape Ollama uses (Ollama's binary is
# small because it links system CUDA and ships no PyTorch; NvLLM is a vLLM
# orchestrator, so it pulls the engine rather than reimplementing it).
#
# Build host: Linux (aarch64 ideal). No vLLM/torch are installed here — this only
# stages source, so it finishes in seconds and needs little disk.
#
# Usage:
#   scripts/build-release.sh                     # build into dist/
#   DIST_DIR=/path  scripts/build-release.sh     # different output dir
#
# Upload the (tiny) Release (this box usually has no gh/token — run one by hand):
#   gh release create <ver> dist/nvllm-<ver>-linux-aarch64.tar.gz --title "nvllm <ver>"
# Users then one-line install (needs network + Python 3.11-3.13):
#   curl -fsSL https://raw.githubusercontent.com/FaberEli/NvLLM/main/scripts/bootstrap.sh | bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DIST_DIR="${DIST_DIR:-$REPO_ROOT/dist}"
VERSION="$(sed -n 's/^version = "\([^"]*\)"/\1/p' "$REPO_ROOT/pyproject.toml" | head -1)"
[ -n "$VERSION" ] || { echo "nvllm: cannot read version from pyproject.toml" >&2; exit 1; }
ART="nvllm-${VERSION}-linux-aarch64"
STAGE="$DIST_DIR/$ART"
TARBALL="$DIST_DIR/$ART.tar.gz"

# ----------------------------------------------------------------- precheck ---
[ "$(uname -s)" = "Linux" ] || { echo "nvllm: build on Linux" >&2; exit 1; }
[ -f "$REPO_ROOT/pyproject.toml" ] || { echo "nvllm: pyproject.toml not found in $REPO_ROOT" >&2; exit 1; }
[ -d "$REPO_ROOT/nvllm_service" ] || { echo "nvllm: nvllm_service/ not found in $REPO_ROOT" >&2; exit 1; }

echo "[nvllm] staging $ART (nvllm source only — vLLM engine is pulled at install time)"

# -------------------------------------------------------------------- stage ---
rm -rf "$STAGE" "$TARBALL"
mkdir -p "$STAGE"

# nvllm package source (pip-installable: pyproject.toml + nvllm_service/)
cp -a "$REPO_ROOT/nvllm_service" "$STAGE/nvllm_service"
cp -a "$REPO_ROOT/pyproject.toml" "$STAGE/pyproject.toml"
[ -f "$REPO_ROOT/LICENSE" ]  && cp "$REPO_ROOT/LICENSE"  "$STAGE/LICENSE"
[ -f "$REPO_ROOT/README.md" ] && cp "$REPO_ROOT/README.md" "$STAGE/README.md"
[ -f "$REPO_ROOT/README.zh-CN.md" ] && cp "$REPO_ROOT/README.zh-CN.md" "$STAGE/README.zh-CN.md"

# post-install helpers + the installer (install.sh at the top level, as before)
mkdir -p "$STAGE/scripts"
for s in update.sh uninstall.sh bootstrap.sh; do
  [ -f "$SCRIPT_DIR/$s" ] && cp "$SCRIPT_DIR/$s" "$STAGE/scripts/$s"
done
cp "$SCRIPT_DIR/install.sh" "$STAGE/install.sh"
chmod +x "$STAGE/install.sh" 2>/dev/null || true
chmod +x "$STAGE/scripts/"*.sh 2>/dev/null || true

# slim: drop __pycache__ (regenerated on import)
find "$STAGE" -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true

# ------------------------------------------------------------------- smoke ---
echo "[nvllm] staged $(find "$STAGE" -type f | wc -l) files"
[ -f "$STAGE/pyproject.toml" ] && [ -d "$STAGE/nvllm_service" ] && [ -f "$STAGE/install.sh" ] \
  || { echo "nvllm: staged layout incomplete" >&2; exit 1; }

# --------------------------------------------------------------------- pack ---
tar czf "$TARBALL" -C "$DIST_DIR" "$ART"

SIZE="$(du -h "$TARBALL" | awk '{print $1}')"
echo
echo "============================================================"
echo "  built:  $TARBALL   ($SIZE)"
echo "  tiny artifact (nvllm source + installer; no bundled venv)."
echo "  at install time, install.sh pip-installs the vLLM engine (needs network)."
echo
echo "  上传 Release（手工执行，本机无 gh/token）:"
echo "    gh release create ${VERSION} ${TARBALL} --title 'nvllm ${VERSION}'"
echo
echo "  之后用户一行安装（需联网 + Python 3.11-3.13）:"
echo "    curl -fsSL https://raw.githubusercontent.com/FaberEli/NvLLM/main/scripts/bootstrap.sh | bash"
echo "============================================================"
