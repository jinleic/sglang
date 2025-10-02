# SGLang Attention Backend Source Map

本文梳理 `--attention-backend` 选择为 **triton / flashinfer / trtllm_mla** 时，SGLang 内部涉及的文件、类与函数调用路径，便于快速定位代码与理解 GPU kernel 的执行流程。

---

## 0. 公共入口

1. **Backend 解析** – `python/sglang/srt/model_executor/model_runner.py:1710-1769`
   - `ModelRunner.init_attention_backend()` → `_get_attention_backend()` 根据 CLI 参数决定 decode/prefill backend。
   - `_get_attention_backend_from_str()` 调用 `ATTENTION_BACKENDS[name]` 构造器，之后由 `attn_backend_wrapper()` 做模型特定包装。
2. **注册表** – `python/sglang/srt/layers/attention/attention_registry.py`
   - 每个 backend 通过 `@register_attention_backend` 注册：
     - `create_triton_backend` (`:69-84`)
     - `create_flashinfer_backend` (`:16-36`)
     - `create_trtllm_mla_backend` (`:39-45`)

---

## 1. Triton Backend

### 1.1 构造与元数据
- **类定义**：`TritonAttnBackend` – `python/sglang/srt/layers/attention/triton_backend.py:54`
  - `__init__` (`:55-163`) 懒加载内核函数 `decode_attention_fwd`/`extend_attention_fwd`，准备 KV 索引缓冲、 deterministic 配置等。
  - `get_num_kv_splits()` (`:165-215`) 使用 Triton kernel `get_num_kv_splits_triton` 动态确定 KV 拆分。
  - `init_forward_metadata()` (`:217-324`) 基于 batch 信息构造 `kv_indptr`、`kv_indices`，并在启用 sliding-window 时调用 `update_sliding_window_buffer`。

### 1.2 前向路径
- `forward_extend()` (`:782-820`) – 用于 prefill/extend；调用 `extend_attention_fwd` Triton kernel。
- `forward_decode()` (`:821-871`) – decode 入口；将数据整理为 `[batch, head, d]` 后调用 `self.decode_attention_fwd`。

### 1.3 GPU Kernel（Triton 实现）
- **Decode 内核**：`python/sglang/srt/layers/attention/triton_ops/decode_attention.py`
  - `_fwd_kernel_stage1` (`:74-178`) – 单个 CTA 对一段 KV 块执行：
    1. 加载 `q` 与对应 `k` 瓦片；`tl.sum(q * k, 1)` 计算 `q ⋅ kᵀ`。
    2. 应用 `sm_scale`、可选 `tanh` softcap 和 XAI temperature。
    3. 做 log-sum-exp (`e_max/e_sum`)，并将 softmax 权重乘以 `v` 聚合到 `acc`。
    4. 写回 `Att_Out`（最终输出）与 `Att_Lse`（logsumexp）。
  - `_decode_att_m_fwd` (`:182-220`) – 配置网格、BLOCK 维度并 launch `_fwd_kernel_stage1`。
- **Extend 内核** 类似存放于 `triton_ops/extend_attention.py`（结构同 decode，只是支持 prefilling/full attention）。

---

## 2. FlashInfer Backend

### 2.1 构造与元数据
- **类定义**：`FlashInferAttnBackend` – `python/sglang/srt/layers/attention/flashinfer_backend.py:81`
  - `__init__` (`:84-235`) 完成：
    - Tensor Core/ deterministic 参数设置。
    - 初始化 FlashInfer Workspace、`BatchPrefill/DecodeWithPagedKVCacheWrapper` 等包装器。
    - 如果启用 CUDA graph，预创建 `kv_indptr`、`qo_indptr`、`kv_last_page_len` 等缓冲。
  - `init_forward_metadata()` (`:236-307`) 根据 `ForwardBatch` 调用 `FlashInferIndicesUpdaterDecode/Prefill`，将请求映射到 FlashInfer wrapper 所需的 paged KV 结构。
  - `init_forward_metadata_capture_cuda_graph()` / `init_forward_metadata_replay_cuda_graph()` (`:400-493`) – 捕获/回放 CUDA graph 时更新索引。

### 2.2 前向路径
- `forward_prefill()` (`:480-559`) – 支持 ragged/paged 组合；在分段场景通过 `merge_state` 合并 softmax 结果。
- `forward_decode()` (`:585-621`) – 取得 `decode_wrapper`，调用其 `forward` 接口。

### 2.3 FlashInfer 内核调用
- `BatchDecodeWithPagedKVCacheWrapper.forward(...)` 与 `BatchPrefillWithPagedKVCacheWrapper.forward(...)`（定义在 `flashinfer` Python 包中）最终调用 FlashInfer C++/CUDA 内核：
  - 这些内核同样一次性执行 `q ⋅ kᵀ → softmax → × v`，并处理 paged KV、tensor core、split tile 等细节。
  - FlashInfer 版本支持 Ragged (变长) 序列、sliding window，以及与 CUDA Graph 集成（需要 `decode_split_tile_size`、`prefill_split_tile_size` 等参数）。

---

## 3. TensorRT-LLM MLA Backend (`trtllm_mla`)

### 3.1 构造与工作区
- **类定义**：`TRTLLMMLABackend` – `python/sglang/srt/layers/attention/trtllm_mla_backend.py:71`
  - 继承 `FlashInferMLAAttnBackend`，在 `__init__` (`:74-131`) 中解析 MLA 专属维度（LoRA rank / NoPE / Rope 维度等）。
  - 维护 TRT-LLM 所需工作区 `workspace_buffer`、CUDA graph metadata（`decode_cuda_graph_kv_indices` 等）。
  - `_calc_padded_blocks()` (`:132-152`) & `_create_block_kv_indices()` (`:154-191`) 用 Triton kernel 构造满足 TRT-LLM 要求的 KV block 索引。

### 3.2 CUDA Graph & Metadata
- `init_cuda_graph_state()` (`:193-208`) – 预分配 decode 时的 block index。
- `init_forward_metadata_capture_cuda_graph()` 等函数对 CUDA Graph 捕获/回放做特化。

### 3.3 前向路径
- `forward_decode()` (`:520-559`) – 读取元数据后调用：
  ```python
  flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
      query=query,
      kv_cache=kv_cache,
      workspace_buffer=self.workspace_buffer,
      qk_nope_head_dim=self.qk_nope_head_dim,
      kv_lora_rank=self.kv_lora_rank,
      qk_rope_head_dim=self.qk_rope_head_dim,
      block_tables=metadata.block_kv_indices,
      seq_lens=forward_batch.seq_lens.to(torch.int32),
      max_seq_len=metadata.max_seq_len,
      bmm1_scale=q_scale * k_scale * layer.scaling,
  )
  ```
  - TensorRT-LLM Engine 在 GPU 上完成整个 Attention（包括 LoRA/NoPE 融合），输出 reshape 后返回。
- `forward_extend()` (`:561-648`) – 兼容 target verify / draft extend，必要时调用同一 TRT-LLM kernel 或回退到父类 FlashInfer 路径，并负责在 KV cache 中写入 MLA 结构（`set_mla_kv_buffer`）。

---

## 4. 补充：其它 Backend 入口（参考）

| Backend | 注册函数 | 实现类/函数 | 特性 |
| --- | --- | --- | --- |
| `torch_native` | `attention_registry.py:87-91` | `TorchNativeAttnBackend` (`python/sglang/srt/layers/attention/torch_native_backend.py`) | 直接调用 `torch.nn.functional.scaled_dot_product_attention`，依赖 cuBLAS。 |
| `flex_attention` | `attention_registry.py:94-98` | `TorchFlexAttnBackend` | 使用 PyTorch FlexAttention。 |
| `fa4` | `attention_registry.py:123-130` | `FlashAttentionBackend(..., fa_impl_ver=4)` | FlashAttention v4 API。 |
| `flashinfer_mla` | `attention_registry.py:101-105` | `FlashInferMLAAttnBackend` | MLA 版本的 FlashInfer 内核。 |
| `cutlass_mla` | `attention_registry.py:133-137` | `CutlassMLABackend` | 基于 CUTLASS 的 MLA kernel。 |

---

## 5. 调试 & 阅读建议

1. 从 `ModelRunner.attn_backend` 下断点，可观测不同 backend 的 `forward_*` 调用链。
2. Triton backend 的详细数学流程可直接阅读 `_fwd_kernel_stage1`，该函数展示了训练级别的 softmax + matmul 手工实现。
3. FlashInfer/TRT-LLM backend 依赖外部库；若需深入，可阅读对应 Python wrapper 或参考 FlashInfer/TensorRT-LLM 官方文档。
4. 所有 backend 最终都依赖于统一的 KV cache 管理（`forward_batch.token_to_kv_pool`），了解 `TokenToKVPool` / `PagedTokenToKVPoolAllocator` 的工作方式有助于定位数据放置位置。

---

**结论**：不同 backend 的差异主要在于 GPU kernel 的实现与数据打包方式。本文列出的文件和函数为排查性能或精度问题时的首选入口，可结合日志（例如 scheduler 打印的 backend 名称）快速定位执行路径。
