# SGLang Attention Backend Execution Paths

> **上下文命令**：`python3 -m sglang.launch_server ... --attention-backend <backend>`
>
> 本文梳理 SGLang 在不同注意力 backend（例如 `fa3`、`triton`、`flashinfer`、`trtllm_mla`）下如何选择并执行 GPU 侧的矩阵乘法/softmax 内核，同时列出其它常见 backend 的处理方式。

---

## 1. Backend 选择流程

1. **模型启动**时，`ModelRunner.init_attention_backend()` 调用 `_get_attention_backend()`，根据 CLI 参数 `--attention-backend`、`--decode-attention-backend`、`--prefill-attention-backend` 生成最终组合 (`python/sglang/srt/model_executor/model_runner.py:1710-1769`)。
2. `_get_attention_backend_from_str()` 会在 `ATTENTION_BACKENDS` 注册表中查找对应构造器，并通过 `attn_backend_wrapper` 针对 Hybrid GDN 等模型做包装 (`python/sglang/srt/model_executor/model_runner.py:1765-1769`)。
3. 注册表 (`python/sglang/srt/layers/attention/attention_registry.py:5-162`) 将文本名称映射到实际 backend 类，例如：
   - `fa3` → `FlashAttentionBackend`
   - `triton` → `TritonAttnBackend`
   - `flashinfer` → `FlashInferAttnBackend` / `FlashInferMLAAttnBackend`
   - `trtllm_mla` → `TRTLLMMLABackend`

因此，命令行开关只影响每层调用的具体 kernel 实现，逻辑输入 (`q`/`k`/`v` 和 KV cache) 不变。

---

## 2. FlashAttention v3 (`--attention-backend fa3`)

### 2.1 构造器与约束
- 注册函数 `create_flashattention_v3_backend` 要求 SM80/SM90 且非 MLA (`python/sglang/srt/layers/attention/attention_registry.py:108-120`)。
- 返回 `FlashAttentionBackend` 实例，`fa_impl_ver=3`。

### 2.2 GPU 内核调用
- Prefill/Decode 都会调用 `flash_attn_with_kvcache()` (`python/sglang/srt/layers/attention/flashattention_backend.py:774-792`)。
- 该函数来自 `sgl-kernel` 扩展 (`sgl-kernel/python/sgl_kernel/flash_attn.py:17-154`)，本质是 FlashAttention v3 的融合 kernel：
  1. 按 KV page table 加载一段 `k` 与 `v` 到寄存器/共享内存。
  2. 对每个 query 向量做 `dot(q, kᵀ)`，在寄存器中完成缩放与可选 softcap。
  3. 使用 log-sum-exp 实现 softmax，直接与 `v` 相乘累加生成输出，避免显式 materialize 注意力矩阵。
  4. 支持 paged KV、GQA/MQA、rotary embedding、num_splits 等增强功能。

### 2.3 乘法的实际形态
- Kernel 实际执行的是一个融合的 **Block GEMM + Softmax + Value GEMM**：矩阵乘法仍然按照 `q` × `kᵀ` 低层实现，但 fused kernel 将 softmax 与 `× v` 一并完成，以减少显存读写。

---

## 3. Triton Backend (`--attention-backend triton`)

### 3.1 构造器
- `create_triton_backend` 根据模型结构选择 `TritonAttnBackend` 或双稀疏变体 (`python/sglang/srt/layers/attention/attention_registry.py:69-84`)。

### 3.2 Triton Kernel 工作流程
- Decode 场景使用 `_fwd_kernel_stage1` 内核 (`python/sglang/srt/layers/attention/triton_ops/decode_attention.py:74-178`)：
  1. 每个 CTA 处理一个 `(batch, head, kv_split)`，从 KV cache 中加载一段 `k`/`v`。
  2. 使用 `tl.sum(q * k, 1)` 直接计算局部的 `q ⋅ kᵀ`，完成缩放及可选 `tanh` logit cap。
  3. 再计算 `p = softmax(qk)`，与 `v` 相乘累加到 `acc`。
  4. 将 `acc / e_sum` 写回输出，并记录 log-sum-exp (`Att_Lse`) 供后续合并。
- Triton backend 没有外部库依赖，矩阵乘法完全由自定义 kernel 驱动，因此可直接阅读/修改计算粒度与并行策略。

---

## 4. FlashInfer Backend (`--attention-backend flashinfer`)

### 4.1 构造器
- 依据 `use_mla_backend` 选择 `FlashInferAttnBackend` 或 `FlashInferMLAAttnBackend` (`python/sglang/srt/layers/attention/attention_registry.py:16-36`)。

### 4.2 内核调用
- FlashInfer backend 在 decode/prefill 时调用自带封装，例如：
  - `BatchPrefillWithPagedKVCacheWrapper`、`BatchDecodeWithPagedKVCacheWrapper` 等包装 FlashInfer 内核 (`python/sglang/srt/layers/attention/flashinfer_backend.py:400-465`)。
  - 对 MLA 模型，会统一通过 TensorRT-LLM IPlugin 接口 `flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla` (`python/sglang/srt/layers/attention/trtllm_mla_backend.py:520-555`)。
- FlashInfer 的 kernel 同样融合了 `qkᵀ→softmax→×v`，但内部实现依赖 FlashInfer 项目的 C++/CUDA 库，支持 Ragged attention、分块等复杂场景。

---

## 5. TensorRT-LLM MLA Backend (`--attention-backend trtllm_mla`)

### 5.1 构造器
- 注册器校验模型必须是 MLA (`python/sglang/srt/layers/attention/attention_registry.py:39-45`)，随后实例化 `TRTLLMMLABackend`。

### 5.2 执行流程
- `TRTLLMMLABackend.forward_decode()` 调用 TensorRT-LLM 集成的 FlashInfer kernel（`python/sglang/srt/layers/attention/trtllm_mla_backend.py:520-559`）：
  1. 构造 `bmm1_scale = q_scale * k_scale * softmax_scale` 匹配 TensorRT 的缩放约定。
  2. 执行 `flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(...)`，在 TensorRT 生成的引擎中完成整个注意力运算。
  3. 对输出 reshape 返回；KV cache 更新（包括 LoRA/NoPE 分量）在同一 backend 处理。
- 由于 kernel 来自 TensorRT builder，矩阵乘法的具体排布由 TensorRT 决定，但从调用参数上看仍然是 `query` × `kv_cache` 的 batched GEMM，结合 TensorRT 的高吞吐 shared-memory pipeline。

---

## 6. 其它常见 Backend 概览

| Backend | 代码入口 | 乘法实现特征 |
| --- | --- | --- |
| `torch_native` | `python/sglang/srt/layers/attention/torch_native_backend.py:17-140` | 直接调用 PyTorch `scaled_dot_product_attention`，底层走 cuBLAS / cuDNN，多步执行（GEMM + softmax + GEMM）。 |
| `flex_attention` | `python/sglang/srt/layers/attention/torch_flex_backend.py` | 依赖 PyTorch FlexAttention 生成的 kernel，自动融合常见操作。 |
| `fa4` | 同 `FlashAttentionBackend`，`fa_impl_ver=4` (`attention_registry.py:123-130`) | 使用 FlashAttention v4 API；不支持 KV cache in-place 更新、rotary 等。 |
| `cutlass_mla` | `python/sglang/srt/layers/attention/cutlass_mla_backend.py` | CUTLASS 实现的 MLA kernel，矩阵乘法通过 CUTLASS GEMM Template 完成。 |
| `dual_chunk_flash_attn` | `python/sglang/srt/layers/attention/dual_chunk_flashattention_backend.py` | 在 FlashAttention 基础上做 chunk 拆分以处理长序列。 |

---

## 7. 总结

- SGLang 通过注册表模式将 `--attention-backend` 文本映射到不同的注意力实现。所有 backend 都完成相同的数学计算（`q` × `kᵀ` → softmax → × `v`），差异在于使用的 kernel/库、是否融合、对 KV cache 的排布支持。
- `fa3`、`flashinfer`、`trtllm_mla` 等沿用高度融合的 CUDA kernel，矩阵乘法与 softmax 在同一 kernel 内执行；`triton` backend 则提供可读性较高的 Triton 实现；`torch_native` 等回退方案依赖 cuBLAS/SDPA。
- 选择 backend 时需关注 GPU 架构支持、模型类型（MLA vs 非 MLA）、以及是否需要 TensorRT/FlashInfer 等额外依赖。合理搭配能在保持正确性的同时最大化带宽利用与吞吐。 
