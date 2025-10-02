# SGLang DeepSeek-V3.1 GPU 显存分配与`--mem-fraction-static`解析

> **命令**: `python3 -m sglang.launch_server --tp 8 --model deepseek-ai/DeepSeek-V3.1-Terminus --attention-backend trtllm_mla --trust-remote-code --enable-torch-compile --torch-compile-max-bs 16 [--mem-fraction-static X]`

---

## 目录

1. [参数定义与命令入口](#阶段1-参数定义与命令入口)
2. [显存分区启发式计算](#阶段2-显存分区启发式计算-server_argshandle_gpu_memory_settings)
3. [KV Cache 容量推导](#阶段3-kv-cache-容量推导-model_runnerprofile_max_num_token)
4. [KV Cache 实际分配流程](#阶段4-kv-cache-实际分配流程-model_runnerinit_memory_pool)
5. [CUDA Graph 与 torch.compile 内存开销](#阶段5-cuda-graph-与-torchcompile-内存开销)
6. [运行期剩余显存与调优策略](#阶段6-运行期剩余显存与调优策略)
7. [常见问题排查要点](#常见问题排查要点)

---

## 阶段1: 参数定义与命令入口

### CLI 入口
**文件**: `python/sglang/launch_server.py`
```python
# launch_server.py:20-28
if __name__ == "__main__":
    server_args = prepare_server_args(sys.argv[1:])
    ...
    launch_server(server_args)
```

### 参数类
**文件**: `python/sglang/srt/server_args.py`
```python
# server_args.py:173-207
@dataclasses.dataclass
class ServerArgs:
    ...
    mem_fraction_static: Optional[float] = None  # CLI 可覆盖
    ...
```
- 默认为 `None`，后续由 `_handle_gpu_memory_settings()` 推导。
- 定义出现在官方调优文档 `docs/advanced_features/hyperparameter_tuning.md:31-39`。

---

## 阶段2: 显存分区启发式计算 (`server_args._handle_gpu_memory_settings`)

**文件**: `python/sglang/srt/server_args.py`
```python
# server_args.py:524-667 (节选)
def _handle_gpu_memory_settings(self, gpu_mem):
    """
    GPU memory capacity = model weights + KV cache pool + activations + cuda graph buffers
    mem_fraction_static = (weights + KV pool) / GPU memory capacity
    """
    ...
    if self.mem_fraction_static is None:
        reserved_mem = 512  # 常驻元数据
        reserved_mem += max(self.chunked_prefill_size, 2048) * 1.5  # 激活估算
        reserved_mem += self.cuda_graph_max_bs * 2                 # CUDA Graph 缓冲
        reserved_mem += self.tp_size * self.pp_size / 8 * 1024     # 并行附加
        if self.enable_dp_attention:
            reserved_mem += self.cuda_graph_max_bs * self.dp_size * 3
            if self.cuda_graph_max_bs > 300:
                reserved_mem += self.cuda_graph_max_bs * self.dp_size * 1.5
        if gpu_mem is not None and gpu_mem > 60 * 1024:
            reserved_mem = max(reserved_mem, 10 * 1024)
        if self.speculative_algorithm == "STANDALONE":
            reserved_mem += 6 * 1024
        elif self.speculative_algorithm not in (None, "NGRAM"):
            reserved_mem += 2 * 1024
        self.mem_fraction_static = round((gpu_mem - reserved_mem) / gpu_mem, 3)
```
**要点**
- `chunked_prefill_size` 与 `cuda_graph_max_bs` 先根据显存容量自适应，再反过来影响 reserved 预算。
- 预算单位为 MB：如 `chunked_prefill_size=8192` → 约 12 GB 激活保留。
- DP Attention、推理草稿 (EAGLE/Standalone) 会额外保留 GPU 内存以防 runtime 爆掉。
- 若 `gpu_mem` 未探测到，回退至 0.88 (`else 0.88`)。
- 多模态模型在 `adjust_mem_fraction_for_vlm()` (`server_args.py:3016-3051`) 再次减小，以给视觉前处理留空间。

---

## 阶段3: KV Cache 容量推导 (`model_runner.profile_max_num_token`)

**文件**: `python/sglang/srt/model_executor/model_runner.py`
```python
# model_runner.py:1248-1284
def profile_max_num_token(self, total_gpu_memory: int):
    available_gpu_memory = get_available_gpu_memory(...)
    num_layers = ...  # 依据模型结构
    if self.use_mla_backend:
        cell_size = (kv_lora_rank + qk_rope_head_dim) * num_layers * element_size
    else:
        cell_size = num_kv_heads * head_dim * num_layers * 2 * element_size
    rest_memory = available_gpu_memory - total_gpu_memory * (1 - self.mem_fraction_static)
    max_num_token = int(rest_memory * (1 << 30) // cell_size)
    return max_num_token
```
**含义**
- `available_gpu_memory` 为加载权重后调用 `torch.cuda.mem_get_info` 的实时值。
- `total_gpu_memory * (1 - mem_fraction_static)` 即预留给激活 + Graph 的空间，被扣除后剩余才是 KV 池预算。
- `cell_size` 反映单 token 在所有层的 KV 存储开销，尤其对 MLA 模型使用 LoRA 排布。

---

## 阶段4: KV Cache 实际分配流程 (`model_runner.init_memory_pool`)

**文件**: `python/sglang/srt/model_executor/model_runner.py`
```python
# model_runner.py:1380-1490 (节选)
def init_memory_pool(self, total_gpu_memory: int, ...):
    self.kv_cache_dtype = ...  # auto/fp8
    self.max_total_num_tokens = self.profile_max_num_token(total_gpu_memory)
    ...
    if self.max_total_num_tokens <= 0:
        raise RuntimeError("Not enough memory. Please try to increase --mem-fraction-static.")

    # 实例化不同形态的 KV 池
    if self.is_hybrid:
        self.token_to_kv_pool = SWAKVPool(...)
    elif self.is_hybrid_gdn:
        self.token_to_kv_pool = HybridLinearKVPool(...)
    else:
        self.token_to_kv_pool = MHATokenToKVPool(...)

    # 申请 allocator（可分页 / SWA / Ascend variants）
    self.token_to_kv_pool_allocator = ...

    logger.info(
        f"Memory pool end. avail mem={get_available_gpu_memory(...):.2f} GB"
    )
```
**流程**
1. 按 `mem_fraction_static` 计算最大 token 数；传入 CLI 的 `--max-total-tokens` 只起上限作用。
2. 选取 KV 池实现：分页 (`PagedTokenToKVPoolAllocator`)、滑窗 (`SWATokenToKVPoolAllocator`) 等。
3. 分配后打印 `avail mem`，即剩余动态空间。此值若低于 2-3 GB，后续极易 OOM。

---

## 阶段5: CUDA Graph 与 torch.compile 内存开销

### Graph 捕获初始化
**文件**: `python/sglang/srt/model_executor/cuda_graph_runner.py`
```python
# cuda_graph_runner.py:212-360 (节选)
class CudaGraphRunner:
    def __init__(self, model_runner):
        self.capture_bs, self.compile_bs = get_batch_sizes_to_capture(model_runner)
        ...
        with torch.device(self.device):
            self.input_ids = torch.zeros((self.max_num_token,), ...)
            self.req_pool_indices = torch.zeros((self.max_bs,), ...)
            self.seq_lens = torch.full((self.max_bs,), ...)
            self.out_cache_loc = torch.zeros((self.max_num_token,), ...)
            self.positions = torch.zeros((self.max_num_token,), ...)
            self.mrope_positions = torch.zeros((3, self.max_num_token), ...)
            self.next_token_logits_buffer = torch.zeros((self.max_num_token, vocab), ...)
```
- `max_bs = max(self.capture_bs)` 受 `--torch-compile-max-bs` 和 `cuda_graph_max_bs` 限制。
- 这些张量全部驻留 GPU，容量随批次线性增长，因此会在 `_handle_gpu_memory_settings()` 中提前预留 `cuda_graph_max_bs * 2` MB。

### torch.compile 的静态缓冲
- 当 `--enable-torch-compile` 设置后，`compile_bs` (≤ `torch_compile_max_bs`) 会以固定形状捕获图，torch Inductor 亦会生成额外 workspace；其空间亦被纳入上面 `reserved_mem` 预算。

---

## 阶段6: 运行期剩余显存与调优策略

### 监控剩余显存
**文件**: `python/sglang/srt/managers/scheduler.py`
```python
# scheduler.py:441-456
logger.info(
    f"max_total_num_tokens={self.max_total_num_tokens}, "
    f"chunked_prefill_size={server_args.chunked_prefill_size}, "
    f"max_prefill_tokens={self.max_prefill_tokens}, "
    f"max_running_requests={self.max_running_requests}, "
    f"context_len={self.model_config.context_len}, "
    f"{'available_cpu_mem' if self.device == 'cpu' else 'available_gpu_mem'}={avail_mem:.2f} GB"
)
```
- `avail_mem` 即权重 + KV 分配后仍可用显存。文档建议保持 5–8 GB (`docs/advanced_features/hyperparameter_tuning.md:41-49`)。

### 调优建议
- 如果 `avail_mem` > 10 GB：增加 `--mem-fraction-static` (步长 0.01) 或增大 `--max-total-tokens`，提升 KV 池容量。
- 如果频繁 OOM：降低 `--chunked-prefill-size`、`--cuda-graph-max-bs` 或调低 `--mem-fraction-static`。
- DP Attention / speculative decoding 激活时，关注额外预留项是否过大，可适当减小 `cuda_graph_max_bs` 以减少乘积项。

---

## 常见问题排查要点

| 现象 | 可能原因 | 排查文件/字段 |
| --- | --- | --- |
| 启动阶段 OOM | `reserved_mem` 估算不足；`mem_fraction_static` 设太高 | `server_args.py` 中 `reserved_mem` 打印；调低 CLI 值 |
| 解码阶段 `KV cache pool is full` | KV 池过小 | `scheduler.py` 日志的 `max_total_num_tokens`；增大 `mem_fraction_static` 或调低 `max_running_requests` |
| 捕获 CUDA Graph 时 OOM | `cuda_graph_max_bs` 过大 | `server_args.py:627-641` 预留；减小 `--torch-compile-max-bs` 或关闭 graph |
| 多模态模型显存不够 | VLM 额外预算不足 | `ServerArgs.adjust_mem_fraction_for_vlm()` 调整后的值；手动再降低 0.05 |

---

## 总结
- `--mem-fraction-static` 决定权重 + KV 池占用的显存份额，其余显存用于激活、Graph、Speculative 支撑结构。
- 默认值由 `_handle_gpu_memory_settings()` 根据显存容量和并行拓扑启发式推导；了解该过程有助于手动调参。
- KV 池实际大小通过 `profile_max_num_token()` -> `init_memory_pool()` 推导并分配，可通过日志验证。
- CUDA Graph、torch.compile、DP Attention 等特性会显著影响保留显存，需要视业务场景协同调整。
- 调优时建议关注：启动日志 `available_gpu_mem`、运行期 `KV cache pool is full` 报警，以及 CUDA Graph 捕获阶段的显存占用。合理设置可在保证稳定性的前提下提升吞吐与并发。
