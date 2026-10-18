#!/usr/bin/env bash
# nvllm — one-line installer
#
#   curl -fsSL https://raw.githubusercontent.com/FaberEli/NvLLM/main/scripts/bootstrap.sh | bash
#
# Downloads the self-contained release tarball (python venv + vLLM + nvllm,
# ~2-4 GB), extracts it, and runs the bundled installer — no python or pip
# needed on the target machine.
#
# GitHub caps a single release asset at 2 GB, so a release larger than that is
# uploaded as split parts (…tar.gz.part-00, …part-01, …). This script tries
# the single-asset URL first (for releases ≤ 2 GB) and transparently falls
# back to downloading + reassembling the parts.
#
# Needs on the target: bash, curl, tar, ~10 GB free disk, aarch64 linux + CUDA GPU.
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

VERSION="${NVLLM_VERSION:-0.4.0}"
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

# 找到解包出的 staging 目录（内含 venv/ + install.sh）
STAGE=""
for d in "$TMP"/*/; do
  d="${d%/}"
  if [ -x "$d/venv/bin/python" ] && [ -f "$d/install.sh" ]; then
    STAGE="$d"
    break
  fi
done
[ -n "$STAGE" ] || { echo "nvllm: tarball layout unexpected (no bundled venv found)" >&2; exit 1; }

echo "[nvllm] installing from $STAGE"
exec bash "$STAGE/install.sh" --yes "$@"
