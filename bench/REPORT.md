# NVFP4 on DGX Spark — 吞吐/延迟调优 + Router 冒烟报告

日期：2026-10-11。机器：DGX Spark (GB10)，121Gi 统一内存，GPU 与本会话后端
（`qwen3.8:latest` via ollama，占 ~37–46Gi）及桌面**共享**。

> 重要前提：GPU 并非独占。以下数字是"与一个常驻 27B 会话模型共享 GB10"时实测的，
> 属保守下限；独占时会更好。所有 worker 用 `NVLLM_WORKER_MAX_JOBS=2`（限 FlashInfer
> JIT 并行，避免 nvcc OOM）、保守的 `gpu-memory-utilization`。

## 1. NVFP4 吞吐/延迟（`Llama-3.1-8B-Instruct-NVFP4`，util 0.3）

harness：`bench/bench.py`（OpenAI 流式 API，测 TTFT/TPOT/单流/聚合吞吐）。
prompt=256 词，max_tokens=128，temperature=0。

### 并发扩展（请求数 = 2×并发）
| 并发 | TTFT ms | TTFT p95 ms | TPOT ms | 单流 tok/s | **聚合 tok/s** |
|---:|---:|---:|---:|---:|---:|
| 8  | 184.6 | 194.1 | 24.4 | 41.0 | 310.5 |
| 16 | 280.9 | 300.1 | 24.5 | 40.8 | **600.6** |
| 32 | 278.9 | 298.2 | 24.5 | 40.8 | 601.3 |

- **聚合吞吐在并发 ≈16 饱和于 ~600 tok/s**（32 不再提升）。
- **TPOT 恒定 ~24.5ms/token**（单流 ~41 tok/s），并发上升几乎不掉速——NVFP4 解码很稳。
- TTFT 随并发从 185ms→280ms（prefill 排队，正常）。

### TTFT 随上下文增长（并发 1，max_tokens 64）
| prompt | TTFT ms | TPOT ms |
|---|---:|---:|
| 256 词  | 52.4 | 25.8 |
| 1024 词 | 73.6 | 26.0 |
| 2048 词 | 115.1 | 26.3 |

- TTFT 随 prompt 长度**温和增长**（52→115ms）；TPOT 不随 prompt 长度变化（生成速度独立于上下文）。

### 首载（JIT + 权重）
- 首次加载 NVFP4 8B：**2m45s**（含 FlashInfer CUTLASS GEMM 内核 JIT，`MAX_JOBS=2`）。
- 内核缓存后（`~/.cache/flashinfer`）复载显著更快；9B judge 首载 ~2m45s–9m（视内核是否需新编）。

## 2. Router（`model:"auto"`）真实 judge 冒烟

### 结论：机制端到端验证通过
- judge 被咨询 → 解析选择 → 加载候选 → 响应完成；judge **常驻不被淘汰**。
- 干净路由示例（judge=`Llama-3.1-8B-Instruct`）：简单问题被正确路由到 **Qwen3-8B**（非 fallback）。

### 过程中发现并修复的问题
1. **judge 被当普通 worker 淘汰**（M3 遗留 bug）：`main.py` 用 `repo:tag` 的
   `info.name` 与不带 tag 的 `router_judge` 字符串比较 → judge 被误判为非 judge，
   加载候选时被顶掉。**已修复**（改按 `repo` 比较）+ 回归测试。
2. **候选含大模型会 OOM 整机**：Router 默认把目录里全部可服务模型当候选，judge 若选中
   128B 模型会直接 OOM。**新增 `--router-candidates` 白名单**限制候选 + 测试。
3. **推理型 judge（Nemotron-Nano-9B）不适合作 judge**：它输出 chain-of-thought 而非
   干净的模型名，导致 `parse_choice` 无法解析而 fallback。**改用非推理、指令跟随好的模型
   （如 `Llama-3.1-8B-Instruct`）作 judge 即干净路由**。
4. **judge 用短名**（如 "Llama-3.1-8B-Instruct"）而 `parse_choice` 只认全名：**已增强**
   支持 base name（去量化后缀），且仅当**唯一命中**才路由（judge 提到多个模型时安全 fallback）。
5. 路由调用 `max_tokens` 48→256，给推理型 judge 留足推理空间（直答 judge 提前 EOS，不受影响）。

### 使用建议
- **judge 选非推理模型**（`Llama-3.1-8B-Instruct` 表现好），避免 Nemotron/Qwen 的 reasoning 变体。
- 候选用 `--router-candidates` 限定为"装得下"的小/中模型，防止 judge 选中超大模型 OOM。
- 候选若是思考型（Qwen3-8B 等），调大 `num_predict` 避免 ` think` 占满预算导致答案被截断。

## 3. 推荐配置（DGX Spark，共享 GPU）

```bash
# 单模型（8B 级）高吞吐服务
NVLLM_WORKER_MAX_JOBS=2 \
NVLLM_GPU_MEMORY_UTILIZATION=0.3 \
.venv/bin/python -m nvllm_service --port 8004

# Router（judge + 小模型候选，安全）
NVLLM_WORKER_MAX_JOBS=2 NVLLM_GPU_MEMORY_UTILIZATION=0.2 NVLLM_MAX_MODELS=1 \
.venv/bin/python -m nvllm_service --port 8004 \
  --router-judge nvidia/Llama-3.1-8B-Instruct-NVFP4 \
  --router-candidates "nvidia/Qwen3-8B-NVFP4,nvidia/Qwen3-14B-NVFP4"
```

- **内存算术**：`gpu-memory-utilization` 按 121Gi 总量预留。N 个 worker × U 需 ≤ 可用余量。
  共享 GPU 时可用余量 ~75Gi；单模型 U≤0.3 稳，Router（judge+1 候选）各 U≤0.2 稳。
- **并发**：8B 级模型在并发 ~16 达到 ~600 tok/s 聚合吞吐，TPOT 恒定 ~24.5ms。
- **JIT 安全**：务必 `NVLLM_WORKER_MAX_JOBS=2`（Spark 共享内存上默认并行会 OOM 杀 nvcc）。

## 4. 复现
```bash
cd service
.venv/bin/python -m pytest -q                      # 49 passed（含 Router/并发回归）
# 吞吐（模型需先由服务加载）
.venv/bin/python bench/bench.py --model nvidia/Llama-3.1-8B-Instruct-NVFP4 \
    --concurrency 8 16 32 --requests 16 --prompt-words 256 --max-tokens 128
```
