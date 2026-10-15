#!/usr/bin/env bash
# nvllm — build the self-contained release (self-contained: python venv + vLLM + nvllm)
#
# Produces dist/nvllm-<ver>-linux-aarch64.tar.gz (~2-4 GB):
#
#   nvllm-<ver>-linux-aarch64/
#     install.sh      落位安装器 — 检测到同级的 venv/ 时跳过 pip，直接落位
#     venv/           预构建环境（python + vllm + nvllm，已装好）
#     LICENSE  README.md
#
# 构建机要求：Linux aarch64（DGX Spark 或同架构）、python 3.11-3.13、
# 可访问 PyPI、空闲磁盘 ≥ 15GB。构建约 5-15 分钟（vLLM 依赖体积大）。
#
# 用法:
#   scripts/build-release.sh                     # 构建到 dist/
#   DIST_DIR=/path  scripts/build-release.sh     # 换输出目录
#   VLLM_VERSION=0.31.0 PYTHON=python3.12 scripts/build-release.sh
#
# 构建完成后上传 Release（本机通常无 gh/token，手工执行其一）:
#   gh release create <ver> dist/nvllm-<ver>-linux-aarch64.tar.gz \
#       --title "nvllm <ver>" --notes "self-contained linux-aarch64 release"
# 或在 github.com/FaberEli/NvLLM/releases 新建 release 并拖入 tarball。
# 之后用户即可一行安装:
#   curl -fsSL https://raw.githubusercontent.com/FaberEli/NvLLM/main/scripts/bootstrap.sh | bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

VLLM_VERSION="${VLLM_VERSION:-0.31.0}"
DIST_DIR="${DIST_DIR:-$REPO_ROOT/dist}"
VERSION="$(sed -n 's/^version = "\([^"]*\)"/\1/p' "$REPO_ROOT/pyproject.toml" | head -1)"
[ -n "$VERSION" ] || { echo "nvllm: cannot read version from pyproject.toml" >&2; exit 1; }
ART="nvllm-${VERSION}-linux-aarch64"
STAGE="$DIST_DIR/$ART"
TARBALL="$DIST_DIR/$ART.tar.gz"

# ----------------------------------------------------------------- precheck ---
[ "$(uname -s)" = "Linux" ] || { echo "nvllm: build on Linux" >&2; exit 1; }
if [ "$(uname -m)" != "aarch64" ]; then
  echo "nvllm: warning: building on $(uname -m) (not aarch64) — the artifact name still says aarch64" >&2
fi

# 解释器选择（与 install.sh 相同的偏好）
if [ -n "${PYTHON:-}" ]; then
  :
else
  PYTHON=""
  for cand in python3.13 python3.12 python3.11; do
    if command -v "$cand" >/dev/null 2>&1; then PYTHON="$cand"; break; fi
  done
  [ -n "$PYTHON" ] || PYTHON="python3"
fi
command -v "$PYTHON" >/dev/null 2>&1 || { echo "nvllm: python interpreter not found: $PYTHON" >&2; exit 1; }
"$PYTHON" -c 'import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] <= (3, 13) else 1)' \
  || { echo "nvllm: python 3.11-3.13 required (vLLM wheel availability)" >&2; exit 1; }
PYVER="$("$PYTHON" -c 'import platform; print(platform.python_version())')"
echo "[nvllm] building $ART  (python $PYVER, vllm==$VLLM_VERSION)"

# -------------------------------------------------------------------- build ---
rm -rf "$STAGE" "$TARBALL"
mkdir -p "$STAGE"

# 预构建 venv：装引擎 + nvllm（非 editable —— 发行版必须自包含，不依赖构建机上的仓库）
"$PYTHON" -m venv "$STAGE/venv"
"$STAGE/venv/bin/python" -m pip install --upgrade pip wheel
echo "[nvllm] installing vllm==$VLLM_VERSION (this pulls torch — slow, ~several GB)"
"$STAGE/venv/bin/python" -m pip install "vllm==$VLLM_VERSION"
echo "[nvllm] installing nvllm from $REPO_ROOT"
"$STAGE/venv/bin/python" -m pip install "$REPO_ROOT"

# 落位资产
cp "$SCRIPT_DIR/install.sh" "$STAGE/install.sh"
chmod +x "$STAGE/install.sh"
[ -f "$REPO_ROOT/LICENSE" ]  && cp "$REPO_ROOT/LICENSE"  "$STAGE/LICENSE"
[ -f "$REPO_ROOT/README.md" ] && cp "$REPO_ROOT/README.md" "$STAGE/README.md"

# 瘦身：清 __pycache__（导入时按需再生，省几十 MB）
find "$STAGE/venv" -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true

# ------------------------------------------------------------------- smoke ---
"$STAGE/venv/bin/nvllm" --version
[ -x "$STAGE/venv/bin/vllm" ] || { echo "nvllm: vllm binary missing after build" >&2; exit 1; }

# --------------------------------------------------------------------- pack ---
tar czf "$TARBALL" -C "$DIST_DIR" "$ART"

SIZE="$(du -h "$TARBALL" | awk '{print $1}')"
echo
echo "============================================================"
echo "  built:  $TARBALL   ($SIZE)"
echo
echo "  上传 Release（手工执行，本机无 gh/token）:"
echo "    gh release create ${VERSION} ${TARBALL} \\
        --title 'nvllm ${VERSION}' --notes 'self-contained linux-aarch64 release'"
echo
echo "  之后用户一行安装:"
echo "    curl -fsSL https://raw.githubusercontent.com/FaberEli/NvLLM/main/scripts/bootstrap.sh | bash"
echo "============================================================"
