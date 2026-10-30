# nvllm

基于 vLLM 的类 Ollama 推理套件——**NvLLM** 模型后端推理引擎：一个 `nvllm` CLI、一个常驻模型服务，以及接入你本地模型的本地 Agent（Claude Code、Codex 等）。

> **English documentation:** [README.md](README.md)

> **0.3.0 破坏性变更：** CLI 统一为单一 `nvllm <command>` 入口（Ollama 风格）。旧的 `nvllm-serve` 二进制已移除——由 `nvllm serve` 取代（安装器会自动清理旧符号链接 / 服务单元）。
>
> **0.4.0：** `nvllm create` 虚拟模型（基础模型 + system 提示 + 采样参数——纯元数据，不动权重）、`run`/REPL 的图像与工具调用体验，以及自包含的一行安装（tarball 内置 python + vLLM + nvllm）。
>
> **0.5.0：** Router 裁判升级为**决策模型**（System One）——一个小型常驻模型，通过*对候选选项打分*（而非自由文本）选出最合适的后端模型。两种裁判后端（默认 vLLM + logprob 打分；可选进程内 HuggingFace 单前向读出）、`nvllm route` 干跑（dry-run），许可证 **Apache-2.0 → MIT**。
>
> **0.6.0（本版本）：** 路由成为**可选、可激活**的能力。`nvllm router on` 扫描模型缓存中的 **System One 决策模型**（靠决策标记文件或名字中的 `decision` 识别——通用检测，绝不写死任何本地模型），让你挑选裁判——可选让一个已加载的本地模型给出**推荐**——并持久化选择；下次 `nvllm serve` 时裁判**常驻**运行，模型列表出现新的 **`AutoRouter`** 选项，选中它即把每个请求路由到最佳本地模型。`nvllm router off` / `nvllm router status` 用于停用与查看。发行改为 **Ollama 风格**：Release tarball 是小型源码包（几 MB、单 GitHub 资产），`install.sh` 在安装时通过 `pip` 拉取 vLLM 引擎——不再需要 2GB 分卷的捆绑 venv tarball。

为在 **DGX Spark (GB10)** 统一内存上以全速运行 **NVFP4 量化**模型而生。它从本地 HuggingFace hub 缓存中发现模型，在固定单一端口（默认 **8004**）上提供服务，并暴露 **Ollama**（`/api/*`）、**OpenAI**（`/v1/*`）和 **Anthropic**（`/v1/messages`）三套 API。每个模型各跑在自己的 `vllm serve` 子进程中，崩溃相互隔离，监督者可回收并重启。

> **交互式演示报告：** [`index.html`](index.html)——单文件、零依赖的功能与用法导览，带在线连接实测面板。线上版：<https://FaberEli.github.io/NvLLM/>（GitHub Pages）。在线面板需在 `nvllm serve` 运行中的机器上打开，面板即成为可用的客户端（服务对浏览器客户端发送宽松 CORS 头）。

## 类 Ollama 的 CLI（`nvllm`）

使用手感与 Ollama 一致——`nvllm <command>`——同时保留 NvLLM 自身特质：**模型来自本地 HuggingFace 缓存**（发现而非下载），模型名为 `org/name[:tag]`。

```bash
nvllm serve                          # 启动服务（前台）
nvllm list                           # 本地 HF 缓存中的模型（表格）
nvllm list nvidia/Qwen               # …按名称前缀过滤
nvllm show nvidia/Qwen3-8B-NVFP4     # 详情：家族、参数量、量化、上下文、能力
nvllm run nvidia/Qwen3-8B-NVFP4 "用一句话介绍你自己"   # 一次性提问
nvllm run nvidia/Qwen3-8B-NVFP4      # 交互式 REPL：/help /bye /clear /set /show
nvllm create assistant --from nvidia/Qwen3-8B-NVFP4 \
  --system "你是一个简洁的中文助手" --temperature 0.3    # 虚拟模型（仅元数据）
nvllm rm assistant                   # 删除虚拟模型
nvllm route "prove sqrt(2) is irrational"   # 预览 'auto' 会选哪个模型（干跑）
nvllm router on                            # 激活自动路由：挑选一个决策模型作裁判（交互式）
nvllm router on --judge acme/DecisionLM    # …或直接指定裁判（不交互）
nvllm router status                        # 查看激活状态 + 缓存中的决策模型
nvllm router off                           # 停用自动路由
nvllm launch claude -p "hello"       # 用本地模型驱动 Claude Code
nvllm launch codex exec "say hi"     # 用本地模型驱动 Codex CLI
nvllm launch --list                  # 支持的 Agent + 可用性
nvllm ps                             # 已加载（常驻）模型 + 保留期限
nvllm stop nvidia/Qwen3-8B-NVFP4     # 卸载一个模型
nvllm version                        # CLI + 服务版本
```

### 启动本地 Agent（`nvllm launch`）

`nvllm launch` 对标 Ollama 的 `ollama launch`：在前台启动一个 Agent CLI，其模型后端指向正在运行的 nvllm 服务——不经过云端、无需 API key。

- **Claude Code**（`nvllm launch claude …`）——服务在 `/v1/messages` 提供 Anthropic Messages API。原生 Claude Code 会在本地校验模型名（`ANTHROPIC_MODEL`/`--model` 只接受已知的 `claude-*` 名字），但 `settings.json` 的 `model` 键**接受任意**名字——因此 `launch` 会写一个小型配置目录（`<state-dir>/claude-launch/settings.json`，含 base URL、token 和你的真实模型），并以 `CLAUDE_CONFIG_DIR` 启动 `claude`。注意：Claude Code 总是发送 `tools` + `tool_choice:"auto"`，所以 worker 必须带工具支持——启动服务时加
  `NVLLM_WORKER_EXTRA_ARGS="--enable-auto-tool-choice --tool-call-parser hermes"`
  （解析器名视模型家族而定）。
- **Codex / Qwen Code / Gemini CLI / opencode / aider**——OpenAI 协议 Agent；`OPENAI_BASE_URL` 指向服务的 `/v1`。

```bash
nvllm launch claude                      # 交互式（默认模型：最大的
                                         # completion 能力模型，或 --model/-m，
                                         # 或 $NVLLM_LAUNCH_MODEL）
nvllm launch claude -p "review this diff"
nvllm launch --model nvidia/Qwen3-8B-NVFP4 codex exec "refactor main()"
```

Agent 名之后的参数原样透传给该 Agent（`nvllm` 自己的旗标放在 Agent 名之前）。

`nvllm run` REPL 保留多轮历史；`/set temperature 0.7`（或 `top_p` / `num_predict`）修改本会话的采样参数。行尾 `\` 续行。

### 虚拟模型（`nvllm create`）

对标 Ollama 的 `ollama create`——同时保留 NvLLM 特质：**从不复制或下载权重**。创建的模型是纯元数据——一个基础模型（来自你的 HF 缓存）+ system 提示 + 采样参数：

```bash
nvllm create assistant --from nvidia/Qwen3-8B-NVFP4 \
  --system "你是一个简洁的中文助手" --temperature 0.3 --top-p 0.9 --num-predict 512

nvllm show assistant     # 基础、system、参数（details.parent_model 标记它）
nvllm run assistant "用一句话介绍你自己"
nvllm rm assistant       # 删除虚拟模型；基础模型仍留在 HF 缓存
```

同一基础模型上的两个虚拟模型**共享同一个 worker 进程**——包装在每次请求时注入（system 消息 + 选项合并，请求选项优先），所以 `nvllm ps` 只显示基础模型。`nvllm rm <真实模型>` 仍会拒绝并给出指引（自行删除 `models--*` 缓存目录）；只有创建出来的模型可删。

### 图像与工具

代理已支持对 vLLM 的视觉与工具调用；CLI 两者都暴露：

```bash
nvllm run <VLM模型> --image cat.png "图里有什么"            # --image 可重复
nvllm run <模型> --tools tools.json "查一下北京的天气"      # OpenAI 工具数组，JSON 文件
```

在 `run` REPL 中，输入一个已存在的**图像路径**（`.png/.jpg/.jpeg/.webp/.gif/.bmp`）即发送图像轮（类似 Ollama 的粘贴）；`/tools tools.json` 为本会话加载工具定义（`/tools` 查看当前集合）。助手的工具调用打印为 `⚙ tool: name({"arg": …})` 并保留在历史中，后续轮次会包含它们。

**`nvllm` ↔ `ollama`**

| nvllm | ollama | 说明 |
|---|---|---|
| `nvllm serve` | `ollama serve` | 启动服务 |
| `nvllm list` | `ollama list` | NvLLM 列出**本地 HF 缓存**；Ollama 列出从注册表下载的模型 |
| `nvllm run` | `ollama run` | 一次性提问或交互式 REPL |
| `nvllm ps` | `ollama ps` | 已加载的模型 |
| `nvllm stop` | `ollama stop` | 卸载模型 |
| `nvllm show` | `ollama show` | 模型详情 |
| `nvllm launch` | `ollama launch` | 用本地模型启动本地 Agent（Claude Code / Codex / …） |
| `nvllm create NAME --from M` | `ollama create` | **仅元数据**：基础 + system + 参数（NvLLM 从不复制权重；Ollama 构建 GGUF） |
| `nvllm pull` | `ollama pull` | **不集成**——NvLLM 从不下载模型；打印指引（`hf download <org>/<name>`） |
| `nvllm rm NAME` | `ollama rm` | 对**创建的**模型有效；对 HF 缓存模型打印指引（删除 `models--*` 目录） |
| `nvllm router on\|off\|status` | — | NvLLM 独有：**激活 / 停用 / 查看**决策模型 Router（挑选裁判，可选推荐） |
| `model: "AutoRouter"` | — | NvLLM 独有：常驻决策模型裁判把该请求路由到最佳本地模型（路由激活后出现） |
| `model: "auto"` | — | NvLLM 独有：隐藏兼容别名，同样触发 Router |
| `nvllm route TEXT` | — | NvLLM 独有：预览路由决策 + 分布（只跑裁判） |

客户端命令通过 HTTP 与运行中的服务通信（默认 `http://127.0.0.1:8004`；用 `NVLLM_SERVER` 或 `--server` 覆盖）。

## 工作原理

- **发现** — 扫描 `HF_HUB_CACHE`（或 `~/.cache/huggingface/hub`），构建模型目录，检测能力（completion / embedding / tools），并标出 vLLM 无法服务的模型。
- **worker** — 每个模型一个 `vllm serve` 进程。管理器在空闲端口启动它，轮询 `/health` 直至就绪，并回收崩溃的 worker。同时最多 `--max-models` 个非裁判模型（默认 `1`）常驻；新增模型超限时逐出价值最低的常驻者（先过期者，再最近最少使用）。超出 `--keep-alive` 的空闲 worker 自动卸载。配置好的 Router 裁判从不被逐出。加载是**串行**的（一次只加载一个模型——共享/统一内存下安全），已加载模型**并发**服务。
- **代理** — 每个请求对照目录解析、确保 worker 就绪、然后转发。Ollama 请求体翻译成 OpenAI 再译回，包括流式（OpenAI SSE 进，Ollama NDJSON 出）。

## 路由（用决策模型自动选模）

把 `--router-judge` 设为一个**决策模型**（*System One* 结构化决策模型——如 `startlux-models/StartLux-Decision-v1.1-4B-Instant` 或 `internlm/Intern-Decision-4B`），而非聊天 LLM。它常驻且从不被逐出。此后任意推理请求可用 `model: "auto"`：裁判读取请求 + 候选模型列表，**把候选作为选项（A/B/C, …）打分并选出最优**，服务再转发给它。

路由是"N 选 1"决策——正是这类模型被训练的用途——所以 Router 不解析自由文本，而是读裁判**在候选选项上的概率分布**并取 argmax。该分布对外暴露（见下文 `/api/route`），你看到的是置信度，而不只是胜者。

## 激活 Router（0.6.0，可选）

Router **默认关闭**，是通过命令**激活**的能力——不是启动时要记住的旗标。激活流程有 Ollama 的味道：

```bash
nvllm router on        # 交互式：扫描缓存中的决策模型 → 挑选裁判
nvllm router status    # 是否激活？哪个裁判？缓存里有哪些决策模型？
nvllm router off       # 停用
```

`nvllm router on` 会做什么：

1. **扫描模型缓存中的 *System One* 决策模型。** 快照带决策标记文件（**`decision_config.json`** 或 **`model_router.yaml`**——厂商文件惯例）或名字带 `decision` 类别关键词（StartLux / Intern / SII 都发行 `*-Decision`）即识别为决策模型。检测是**通用**的——NvLLM **不**写死任何具体本地模型名。若**一个都没找到**，报告激活失败并打印 `hf download <org>/<decision-model>` 指引（或用 `--judge` 直接指定一个）。
2. **列出**你有的决策模型（编号），让你挑选裁判。也可以直接传裁判跳过交互：`nvllm router on --judge acme/DecisionLM`。
3. **可选：让本地模型推荐一个。** 一个额外选项（默认开）把决策模型的元数据交给一个**已加载**的本地模型（回退到最大可跑的聊天模型），它阅读后返回推荐；再展示一次列表，你可接受或改选。用 `--no-recommend` 关闭。
4. **持久化选择**（`state_dir()/router.json`）。在**下次 `nvllm serve`** 生效——那时裁判成为**常驻** worker（从不逐出），`nvllm list`、`/api/tags`、`/v1/models` 出现新的 **`AutoRouter`** 选项。选中它，每个请求都经常驻裁判路由到最佳本地模型：

   ```bash
   nvllm run AutoRouter "prove that sqrt(2) is irrational"   # 裁判挑选后端
   ```

隐藏的 `model: "auto"` 别名仍可用（0.5.0 兼容），`--router-judge` / `--router-candidates` / `--router-backend` / `--router-top-logprobs` 启动旗标也仍可用——**显式启动旗标优先**于持久化激活，激活优先于默认关闭。

### 裁判后端

| `--router-backend` | 裁判如何运行 | 备注 |
|---|---|---|
| `vllm`（默认） | 裁判是一个 `vllm serve` worker；从响应对选项字母的 `top_logprobs` 读选择 | 零新增依赖，复用全部 worker 生命周期；DGX Spark + vLLM 0.31 验证可用 |
| `transformers` | 裁判**进程内**加载（HuggingFace + torch），**一次前向**读出——原生"对允许选项符号做 softmax" | 更忠实；需要服务环境有 `transformers`+`torch`，且 transformers 构建认识该模型架构 |

两种后端共享同一打分核心，"分布 → 选择"的方式一致。若裁判不可咨询或无可用返回，Router 退化为**最大候选**，而非让请求失败。

### 干跑：`nvllm route`

预览决策**不消耗后端模型**（只跑裁判）——适合探索路由：

```bash
nvllm route "prove that sqrt(2) is irrational"
# route -> nvidia/Qwen3-14B-NVFP4:main    (backend: vllm-logprob, judge: …)
#   distribution:
#       0.3410  ##########  nvidia/Qwen3-14B-NVFP4:main
#       0.2827  ########  nvidia/Qwen3-32B-NVFP4:main
#       …
```

### 配置

**推荐（0.6.0，可选）：** 从 CLI 激活 Router，然后重启服务使裁判常驻：

```bash
nvllm serve                                # 启动服务
nvllm router on                            # 扫描缓存 → 挑选裁判（或：--judge …）
nvllm router status                        # 确认：裁判 + 候选 + 决策模型
# 重启使激活生效（裁判常驻，AutoRouter 出现）：
nvllm serve
nvllm list                                 # 注意新的 "AutoRouter" 条目
curl -s localhost:8004/api/router | jq     # enabled + judge + candidates
nvllm route "what is 2+2?"                 # 干跑预览（只跑裁判）
curl -s localhost:8004/api/chat -d '{      # 经 AutoRouter 的真实路由
  "model": "AutoRouter",
  "messages": [{"role": "user", "content": "hello"}]
}'
```

**备选（0.5.0，显式启动旗标）：** 在 `serve` 时传裁判——该路径优先于持久化激活：

```bash
nvllm serve \
  --router-judge startlux-models/StartLux-Decision-v1.1-4B-Instant \
  --router-candidates "nvidia/Qwen3-8B-NVFP4,nvidia/Qwen3-14B-NVFP4,nvidia/Qwen3-32B-NVFP4,nvidia/Phi-4-reasoning-plus-NVFP4" \
  --router-top-logprobs 10 \
  --gpu-memory-utilization 0.35     # 见下文 DGX Spark 说明
```

> **实战要点（DGX Spark GB10 验证）：**
> - **用 `--router-candidates` 约束候选。** 缓存里有几十个模型时选项列表很长；只保留你实际要在其中路由的几个。`--router-top-logprobs` 必须 ≥ 候选数，才能捕获每个选项字母。
> - **用决策模型，且优先 4B 档。** 同一批提示词实测：**4B** 决策模型对琐碎任务选轻量模型、对图像请求选视觉模型（信号清晰）；**0.8B** 接近抛硬币、偶尔方向错。候选都是相近的通用 LLM 时分布自然更平。
> - **裁判的许可证**：`StartLux-Decision` 为 CC-BY-NC-4.0（非商业），`Intern-Decision` / `SII-Decision` 为 Apache-2.0。按用途选择。
> - **`--gpu-memory-utilization`**：裁判也是 worker。Spark 上 OS/桌面占用 128G 的相当一部分，默认 `0.5` 可能启动失败（向 122G 设备申请 ~61G）。常驻裁判用更低值（如 `0.35`）；决策模型小，需求不大。

## 安装

### 一键（推荐）

前提：**Linux**（DGX Spark GB10 / aarch64 验证）+ CUDA GPU，**Python ≥ 3.11**。

```bash
git clone https://github.com/FaberEli/NvLLM.git && cd NvLLM
./scripts/install.sh
```

安装器会：

1. 在 `~/.nvllm/` 创建自包含安装根（venv + 日志 + 元数据）；
2. **拉取 vLLM 引擎**——锁定验证基线 **`vllm==0.31.0`**（含 CUDA 13.0 的匹配 torch 构建）；`--vllm-version V` 覆盖；
3. 安装 `nvllm` CLI + 其（少量）Python 依赖；
4. 把 `nvllm` 软链进 `~/.local/bin`（自动清理旧的 `nvllm-serve` 链接）；
5. 可选安装 systemd **用户**服务（自启、崩溃重启）：`./scripts/install.sh --with-service`。

幂等——重复执行即就地升级。

### 更新

```bash
./scripts/update.sh                 # 更新 nvllm；vLLM 保持锁定
./scripts/update.sh --upgrade-vllm  # 同时跟进 vLLM 最新 release
```

### 卸载（恢复环境）

```bash
./scripts/uninstall.sh              # 停止服务，删除 ~/.nvllm/ + PATH 软链
./scripts/uninstall.sh --purge      # …并删除模型/JIT 缓存（破坏性）
```

默认**保留你的模型权重与缓存**——脚本会先列出删什么、留什么，再询问。

### Release（一行）

Release tarball 现在是**小型**的（nvllm 源码 + 安装器，几 MB）。安装器创建 venv 并 `pip install` vLLM 引擎（`vllm==0.31.0` → 拉 torch+CUDA，一次性 ~3-4GB）加 nvllm，所以目标机需要**网络 + Python 3.11-3.13**。这与 Ollama 形态一致（小下载；重件安装时拉取），且不同于 0.4.0 的"捆绑 venv" release——以**单个 GitHub 资产**上传（无 2GB 分卷）。`install.sh` 仍会自动检测同级 `venv/`，若捆绑则跳过 pip（向后兼容）：

```bash
curl -fsSL https://raw.githubusercontent.com/FaberEli/NvLLM/main/scripts/bootstrap.sh | bash
curl -fsSL https://raw.githubusercontent.com/FaberEli/NvLLM/main/scripts/bootstrap.sh | bash --with-service
```

在任意 Linux 机器上构建 tarball（此处不需要 vLLM/torch——只暂存源码，几秒完成）：

```bash
scripts/build-release.sh        # → dist/nvllm-0.6.0-linux-aarch64.tar.gz（几 MB）
```

然后把这**单个**资产上传到 GitHub Release——体积小，无需分卷（永不触及 GitHub 单资产 2GB 上限）：

```bash
gh release create 0.6.0 dist/nvllm-0.6.0-linux-aarch64.tar.gz --title "nvllm 0.6.0"
```

可调项：`NVLLM_VERSION`、`NVLLM_RELEASE_URL`、`NVLLM_HOME`；额外参数（如 `--with-service`）透传给安装器。

### 开发安装

用可编辑 venv 直接在仓库里开发：

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pip install "vllm==0.31.0"          # 引擎（验证基线）
python -m pytest -q                 # 185 测试，无需 GPU
```

## 快速开始

```bash
nvllm serve          # 默认绑定 0.0.0.0:8004
```

然后用 CLI 或原生 HTTP：

```bash
nvllm list
nvllm run <org>/<Model> "hello"

curl -s localhost:8004/api/tags | jq
curl -s localhost:8004/v1/models | jq

# Ollama 聊天
curl -s localhost:8004/api/chat -d '{
  "model": "org/Model",
  "messages": [{"role": "user", "content": "hello"}],
  "stream": false
}' | jq

# OpenAI 聊天
curl -s localhost:8004/v1/chat/completions -d '{
  "model": "org/Model",
  "messages": [{"role": "user", "content": "hello"}]
}' | jq
```

## 配置

优先级：**CLI 旗标 > 环境变量 > 默认值**。

| 旗标 | 环境变量 | 默认 | 含义 |
|---|---|---|---|
| `--host` | `NVLLM_HOST` | `0.0.0.0` | 绑定主机 |
| `--port` | `NVLLM_PORT` | `8004` | 绑定端口 |
| `--hf-hub-root` | `HF_HUB_CACHE` / `HF_HOME` | `~/.cache/huggingface/hub` | 要扫描的 HF 缓存 |
| `--keep-alive` | `NVLLM_KEEP_ALIVE` | `300` | 空闲秒数后卸载（`-1` 永不卸载，`0` 立即卸载） |
| `--max-models` | `NVLLM_MAX_MODELS` | `1` | 同时最多常驻的非裁判模型数（裁判不计入） |
| `--max-model-len` | `NVLLM_MAX_MODEL_LEN` | （vLLM 默认） | 传给 worker |
| `--gpu-memory-utilization` | `NVLLM_GPU_MEMORY_UTILIZATION` | `0.5` | 传给 worker |
| `--router-judge` | `NVLLM_ROUTER_JUDGE` | — | 作 Router 裁判的**决策模型** |
| `--router-candidates` | `NVLLM_ROUTER_CANDIDATES` | all supported | Router 可路由到的逗号分隔仓库（只保留你要路由的模型） |
| `--router-backend` | `NVLLM_ROUTER_BACKEND` | `vllm` | 裁判后端：`vllm`（serve + logprob 打分）或 `transformers`（进程内单前向读出） |
| `--router-top-logprobs` | `NVLLM_ROUTER_TOP_LOGPROBS` | `32` | 向裁判请求的选项打分 top-k logprobs（≥ 候选数） |
| — | `NVLLM_STATE_DIR` | `~/.local/share/nvllm-serve` | 状态目录：创建模型库（`created.json`）、Router 激活（`router.json`）、`launch claude` settings.json |

> Router 激活命令 `nvllm router on` 对应这些旗标——`--judge`、`--candidates`、`--backend {vllm,transformers}`、`--top-logprobs`，外加 `--no-recommend`——并把选择存入 `state_dir()/router.json`（`serve` 时显式 `--router-judge` 仍优先）。

worker 还认：`NVLLM_MAX_NUM_SEQS`、`NVLLM_TRUST_REMOTE_CODE`、`NVLLM_WORKER_MAX_JOBS` 和 `NVLLM_WORKER_EXTRA_ARGS`（追加到每个 `vllm serve` 的原始参数，如 `"--enable-auto-tool-choice --tool-call-parser hermes"`）。

CLI 客户端命令（`nvllm` CLI）还认 `NVLLM_SERVER`——服务 URL（默认 `http://127.0.0.1:8004`）；每条命令也接受 `--server URL`。

> **DGX Spark (GB10)：** OS/桌面消耗 128G 统一内存的相当大比例，vLLM 默认 `gpu-memory-utilization` 0.92 启动即失败。`0.5` 是那里的安全默认；专用机器上可调高。

> **NVFP4 首次加载（DGX Spark）：** 首个请求会把 FlashInfer CUTLASS NVFP4 GEMM 内核 JIT 编译进 `~/.cache/flashinfer`。这是一次大型、吃内存的构建——默认并行度下会在 Spark 共享 128G 上 OOM 杀 `nvcc`（SIGKILL / exit 137）。设 `NVLLM_WORKER_MAX_JOBS=2`（或 3–4）。内核有缓存，后续加载快。

## 多模型并发

默认一次只常驻**一个**非裁判模型（`--max-models 1`）——Spark 共享 128G 下安全。调高可同时保温多个模型：

```bash
# 同时常驻两个 8B 档模型
NVLLM_MAX_MODELS=2 NVLLM_GPU_MEMORY_UTILIZATION=0.35 .venv/bin/python -m nvllm_service
```

- **常驻并发，加载串行。** 已加载模型并行处理请求，但一次只加载一个模型（两个 `vllm serve` 同时加载——各自 JIT 编译内核——可能 OOM 共享内存）。请求一个未加载模型时排队等待。
- **逐出 LRU，过期优先。** 新模型将超 `--max-models` 时，管理器先逐出过期常驻者，再逐出最近最少使用。裁判从不被逐出。
- **keep-alive 真正卸载。** 超 keep-alive 的空闲 worker 被监督者回收；处理中请求的 worker 从不被杀。
- **内存算账（DGX Spark，128G）：** `gpu-memory-utilization` 是每 worker 的、以*设备总内存*为分数。所以 `N` 个 worker × `U` 必须放得下：默认 `U=0.5` 时两个 worker 已占满 128G。`N` 个模型大约用 `U ≈ 余量 / N`（如 `N=2 → U≈0.35`）。

## 安全

服务通过 HTTP 暴露模型推理，且**无内置鉴权**。默认绑定 `0.0.0.0`（所有接口）的 `8004` 端口，多主机机器上网络上任何人都能用你的 GPU。

- 放在可信网络内，或绑定特定接口：`--host 127.0.0.1`（仅本机）或 `--host 192.168.1.10`。
- 必须对外暴露时，放在反向代理 / VPN / 防火墙后并自行加鉴权。本服务不校验调用方。
- 浏览器（跨源）访问按设计允许——每个响应都带 `Access-Control-Allow-Origin: *`，让 Web 客户端和 `index.html` 演示面板可从任意源工作。与开放绑定的信任姿态相同；不要无鉴权暴露到公网。

worker（`vllm serve` 子进程）无论 `--host` 如何都绑定 `127.0.0.1`，所以只有本服务可直接触达。

## 端点

| 方法 | 路径 | API |
|---|---|---|
| GET | `/healthz`, `/api/version` | 元信息 |
| GET | `/api/tags` | Ollama — 列出模型 |
| GET | `/api/ps` | Ollama — 已加载模型 |
| POST | `/api/show` | Ollama — 模型信息 |
| POST | `/api/create` | Ollama — 创建虚拟模型（基础 + system + 参数，仅元数据） |
| POST | `/api/rm` | Ollama — 删除虚拟模型（真实模型：404 + 指引） |
| POST | `/api/unload` | Ollama — 卸载已加载模型（`nvllm stop`） |
| POST | `/api/chat`, `/api/generate` | Ollama — 推理（流式 + 非流式） |
| POST | `/api/embed`, `/api/embeddings` | Ollama — 嵌入 |
| GET | `/api/router` | 路由状态：`enabled`（实时）+ `configured`（持久化）+ 裁判 + 候选 + 缓存中 `decision_models` |
| POST | `/api/router` | 激活（`on`）/ 停用（`off`）Router——持久化到 `state_dir()/router.json` |
| POST | `/api/router/recommend` | 让已加载模型读决策模型元数据并推荐裁判 |
| POST | `/api/route` | 干跑：只跑裁判，返回选择 + 分布（`nvllm route`） |
| GET | `/v1/models` | OpenAI — 列出模型 |
| POST | `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings` | OpenAI — 推理 |
| POST | `/v1/messages` | Anthropic Messages API — 供 `nvllm launch claude` 及其他 Anthropic 客户端（含 SSE 流式） |
| POST/GET | `/v1/aliases` | 模型名别名（客户端接受的名字 → 真实模型），`nvllm launch claude` 使用 |

## 测试

```bash
pip install -e ".[dev]"
python -m pytest -q
```

测试套件（185 测试）使用伪造 HF hub 缓存和一个 OpenAI 兼容 worker 桩（含为 Router 返回 logprobs 的桩），因此无需 GPU 或真实 vLLM 安装即可运行——也在 CI 中跑。

## NvLLM 是什么、借鉴什么、拉取什么

**身份。** NvLLM 是一个**类 Ollama 的常驻推理后端**，服务于**你本地 HuggingFace 缓存中的模型**，由 **vLLM** 驱动。它对标 Ollama 的 API 与手感（子命令 CLI、`/api/*`、`run`/REPL、`ps`/`stop`、`create`），并加上自身特质：模型**从本地 HF 缓存发现、从不下载**，且为在 **DGX Spark (GB10)** 统一内存上以全速运行 **NVFP4 量化**模型而调优（默认端口 **8004**）。

**本仓库只含 NvLLM 自身代码。** 它**不**包含上游 vLLM 引擎、参考用的 Ollama 或其他被引用项目的源码。（若存在同级 `ollama/`、`vllm/` 检出，那只是本地参考，不属于本仓库。）

**借鉴**
- **Ollama** —— CLI 形态、`/api/*` 端点与"常驻 runner"体验。NvLLM 说这些协议，Ollama 风格客户端可原样工作。
- **vLLM** —— 推理引擎。NvLLM **以子进程启动 `vllm serve`** 并经 HTTP 与之通信；**不**捆绑或 `import` vLLM 代码。它们是独立程序（类比 Ollama ↔ llama.cpp）。vLLM 是*运行时前提*，不是本仓库的代码依赖。

**安装 / 运行时拉取**（都不在本仓库）：
- **安装：** 锁定的 **`vllm==0.31.0`**（DGX Spark 验证基线）+ 匹配 torch/FlashInfer，装入 **`~/.nvllm/venv` 的隔离 venv**——因此**不**干扰你自己部署的任何 vLLM。`--vllm-version V` / `NVLLM_VLLM_VERSION` 覆盖。
- **运行时：** 你的 **HuggingFace 模型缓存**（你用 `hf download <org>/<name>` 等自备权重；NvLLM 从不下载）以及可选 `transformers` 裁判后端所需的 `transformers` + `torch`。

**原创构建**：HF 缓存模型发现与能力检测；常驻 worker 监督者（生成 / 健康 / LRU 逐出 / 在途保护）；Ollama + OpenAI + Anthropic 三协议代理与流式转译；**可选的决策模型 Router**（System One 裁判、logprob / 单前向读出、`nvllm router on` 激活 + `AutoRouter` 选项 + 本地模型推荐、`nvllm route` 干跑）；`nvllm launch` 本地 Agent；以及仅元数据的虚拟模型。

## 许可证

nvllm 以 **MIT License** 授权（见 [LICENSE](LICENSE)）。

用作 Router 裁判的决策模型是带各自许可证的第三方检查点——如 `Intern-Decision` / `SII-Decision`（Apache-2.0）、`StartLux-Decision`（CC-BY-NC-4.0，非商业）。它们**不**属于本仓库，受各自条款约束；请选许可证与用途相符的裁判。

## 里程碑

- **M1** — 目录发现 + 元数据 API + worker 进程管理。✅
- **M2** — 推理代理（Ollama + OpenAI，流式）。✅
- **M3** — Router / 自动路由（`--router-judge` + `model: "auto"`）。✅
- **M4** — DGX Spark (GB10) NVFP4 端到端冒烟：`Llama-3.1-8B-Instruct-NVFP4` 同时经 `/api/chat`（Ollama）与 `/v1/chat/completions`（OpenAI）服务。✅
- **M5** — 多模型并发：`--max-models`（非裁判模型共存、LRU/过期优先逐出）、keep-alive 到期回收、在途保护（请求永不被切断）。✅
- **M6** — Router 加固（裁判常驻修复、`--router-candidates` 白名单、短名解析）+ DGX Spark 上 NVFP4 吞吐/时延基准与真裁判 Router 冒烟（见 [`bench/REPORT.md`](bench/REPORT.md)）。✅
- **M7** — 深度代码评审（流式在途保护修复、死配置与版本清理）+ 开源打包（LICENSE、安全说明、文档）。✅
- **M8** — 一键打包（`scripts/install.sh` / `update.sh` / `uninstall.sh` 装入 `~/.nvllm/`，锁定 vLLM 基线，可选 systemd 用户服务，`--version` + vLLM 预检）+ 开源收尾（Apache-2.0、CI）。✅
- **M9** — 类 Ollama CLI（`nvllm` 子命令：serve / list / run / ps / stop / show / version，一次性 + 交互 REPL，`POST /api/unload`），同时保留 NvLLM 特质（HF 缓存模型发现、不下载模型——`pull`/`rm`/… 改为打印指引）。✅
- **M10** — 本地 Agent（`nvllm launch claude/codex/qwen/gemini/opencode/aider` 接入本地服务）+ `/v1/messages` 的 Anthropic Messages API（流式 SSE、工具、system）+ CLI 统一（单一 `nvllm` 入口；`nvllm-serve` 双入口在 0.3.0 退役——刻意的破坏性变更）。✅
- **M11 (0.4.0)** — 纯化：`nvllm create` 虚拟模型（基础 + system + 参数，worker 共享，按请求注入）+ 对它们的真 `nvllm rm`；CLI 视觉（`--image`）+ 工具调用（`--tools` / REPL `/tools`、`⚙ tool:` 展示）；`launch claude` 打通（Claude Code 客户端侧卡 `ANTHROPIC_MODEL`，但接受 `settings.json` 里任意名字 → `CLAUDE_CONFIG_DIR`）；自包含发行（`build-release.sh` tarball 捆绑 python+vLLM、`bootstrap.sh` 一行安装）。✅
- **M12 (0.4.0 收尾)** — 深度复盘与优化：6 处正确性修复 + 4 处结构统一（143 tests）。✅
- **M13 (0.5.0)** — Router 裁判成为**决策模型（System One）**：候选作为选项打分，选择从裁判在选项字母上的概率分布读出（默认 **vLLM + logprob** 后端，或进程内 **transformers** 单前向读出）；优雅回退链（logprob → 字母/文本 → 最大）；`nvllm route` 干跑 + `/api/route` 暴露分布；深度评审；**许可证 Apache-2.0 → MIT**；README 写清 NvLLM 是什么 / 借鉴 / 原创 / 拉取。DGX Spark 上对真实 0.8B 与 4B 决策模型端到端验证。✅
- **M14 (0.6.0)** — 路由成为**可选、可激活的能力**：`nvllm router on` 扫描模型缓存中的 **System One 决策模型**（`decision_config.json`/`model_router.yaml` 等决策标记文件，或 `decision` 名字——通用，绝不写死本地模型），让你挑选裁判，并可**让一个已加载的本地模型读它们的元数据后推荐一个**；选择**持久化**（`state_dir()/router.json`），下次 `nvllm serve` 时裁判**常驻**运行，新的 **`AutoRouter`** 模型选项出现（`nvllm list`、`/api/tags`、`/v1/models`），把每个请求经常驻裁判路由；`nvllm router off` / `nvllm router status` 停用与查看；`POST /api/router` + `POST /api/router/recommend` 端点。0.5.0 的显式旗标路径（`--router-judge`）与隐藏的 `model: "auto"` 别名保留（旗标优先于持久化激活）。打包切换为 **Ollama 风格**：Release tarball 是小型源码包（几 MB、单 GitHub 资产），`install.sh` 安装时 `pip` 拉取 vLLM 引擎——取代 0.4.0 必须拆 2GB 分卷的捆绑 venv tarball。✅
