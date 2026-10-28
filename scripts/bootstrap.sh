#!/usr/bin/env bash
# nvllm — one-line installer
#
#   curl -fsSL https://raw.githubusercontent.com/FaberEli/NvLLM/main/scripts/bootstrap.sh | bash
#
# Downloads the small release tarball (nvllm source + installer, a few MB),
# extracts it, and runs the bundled installer. The installer builds a venv and
# `pip install`s the vLLM engine (vllm==pin → pulls torch+CUDA, ~3-4 GB, one-time)
# plus nvllm — so the target needs network + Python 3.11-3.13. (The 0.4.0 release
# bundled the venv and had to be split across two 2 GB GitHub assets; this one is
# a single small asset and pulls the engine at install time, Ollama-style.)
#
# Needs on the target: bash, curl, tar, aarch64 linux + CUDA GPU, network access
# to PyPI, and ~15 GB free disk (the vLLM engine + torch download).
#
# Environment:
#   NVLLM_VERSION        release version to install (default: 0.4.0)
#   NVLLM_RELEASE_URL    full tarball URL (overrides NVLLM_VERSION / BASE; must be a single file)
#   NVLLM_RELEASE_BASE   download base (default: GitHub releases of FaberEli/NvLLM)
#   NVLLM_HOME           install root (default: ~/.nvllm)
#
# Extra arguments are passed through to install.sh:
#   curl -fsSL … | bash --with-service     # also install the systemd user service

set -euo pipefail

VERSION="${NVLLM_VERSION:-0.6.0}"
BASE="${NVLLM_RELEASE_BASE:-https://github.com/FaberEli/NvLLM/releases/download}"
URL="${NVLLM_RELEASE_URL:-$BASE/$VERSION/nvllm-$VERSION-linux-aarch64.tar.gz}"

[ "$(uname -s)" = "Linux" ] || { echo "nvllm: this installer targets Linux (validated on DGX Spark GB10)" >&2; exit 1; }
for tool in curl tar; do
  command -v "$tool" >/dev/null 2>&1 || { echo "nvllm: missing required tool: $tool" >&2; exit 1; }
done

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

if [ -n "${NVLLM_RELEASE_URL:-}" ]; then
  echo "[nvllm] downloading $URL"
  curl -fL --retry 2 -o "$TMP/nvllm.tar.gz" "$URL"
else
  # 1) single-asset release (<= 2 GB)
  echo "[nvllm] trying single tarball: $URL"
  if curl -fL --retry 2 -o "$TMP/nvllm.tar.gz" "$URL" 2>/dev/null; then
    :
  else
    # 2) split release: GitHub caps each asset at 2 GB, so big releases ship as
    #    …tar.gz.part-00 / …part-01 / … — download them in order and reassemble.
    echo "[nvllm] single tarball not found — downloading split parts…"
    n=0
    while :; do
      part_url="$(printf '%s.part-%02d' "$URL" "$n")"
      if curl -fL --retry 2 -o "$TMP/part-$n" "$part_url" 2>/dev/null; then
        echo "[nvllm]   got part $(printf '%02d' "$n")"
        n=$((n + 1))
      else
        break
      fi
    done
    if [ "$n" -eq 0 ]; then
      echo "nvllm: release $VERSION not found — tried $URL and its split parts" >&2
      exit 1
    fi
    echo "[nvllm] reassembling $n part(s)…"
    : > "$TMP/nvllm.tar.gz"
    for i in $(seq 0 $((n - 1))); do
      cat "$TMP/part-$i" >> "$TMP/nvllm.tar.gz"
    done
  fi
fi

echo "[nvllm] extracting"
tar xzf "$TMP/nvllm.tar.gz" -C "$TMP"

# 找到解包出的 staging 目录（内含 install.sh + nvllm 源码；0.4.0 的旧布局还带 venv/）
STAGE=""
for d in "$TMP"/*/; do
  d="${d%/}"
  if [ -f "$d/install.sh" ] && [ -f "$d/pyproject.toml" ]; then
    STAGE="$d"
    break
  fi
done
[ -n "$STAGE" ] || { echo "nvllm: tarball layout unexpected (no install.sh + pyproject.toml found)" >&2; exit 1; }

echo "[nvllm] installing from $STAGE"
# Not `exec`: a plain call lets the EXIT trap above clean up the temp staging dir
# on both success and failure (exec would replace the process and skip the trap).
# set -e propagates install.sh's exit status.
bash "$STAGE/install.sh" --yes "$@"
