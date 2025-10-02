# DeepSeek V3.1 Terminus Launch Flow

## Source References
- `python/sglang/launch_server.py:20-28` – CLI entry point that wires argument parsing to server launch.
- `python/sglang/srt/server_args.py:3055-3089` – `prepare_server_args()` builds the `ServerArgs` object from CLI switches such as `--tp 8`, `--attention-backend trtllm_mla`, and `--enable-torch-compile`.
- `python/sglang/srt/utils.py:1066-1078` – `prepare_model_and_tokenizer()` resolves remote model/tokenizer assets before loading.
- `python/sglang/srt/entrypoints/http_server.py:1198-1278` – `launch_server()` configures FastAPI, tracing, and hands off to the engine bootstrapper.
- `python/sglang/srt/entrypoints/engine.py:754-901` – `_launch_subprocesses()` allocates IPC ports and spawns tokenizer, scheduler, and detokenizer components.
- `python/sglang/srt/managers/scheduler.py:2781-2870` – `run_scheduler_process()` instantiates the per-GPU scheduler and drives its event loop.
- `python/sglang/srt/managers/scheduler.py:227-520` – `Scheduler.__init__()` wires ZMQ channels, KV cache pools, and launches the tensor-parallel worker that will execute GPU kernels.
- `python/sglang/srt/managers/tp_worker.py:61-198` – `TpModelWorker.__init__()` wraps a `ModelRunner` on each tensor-parallel shard and exposes GPU-facing helpers.
- `python/sglang/srt/model_executor/model_runner.py:208-420` – `ModelRunner.__init__()` and `initialize()` set up NCCL, offloading, and memory pools before the model is constructed.
- `python/sglang/srt/model_executor/model_runner.py:722-845` – `ModelRunner.load_model()` loads weights (respecting `--trust-remote-code`), applies tensor-parallel sharding, and synchronises ranks.
- `python/sglang/srt/model_executor/model_runner.py:1688-1999` – Attention backend selection and `forward()` paths that execute prefill/decode steps and interact with CUDA graphs.
- `python/sglang/srt/model_executor/cuda_graph_runner.py:212-360` – `CudaGraphRunner.__init__()` captures CUDA graphs and drives `torch.compile` for batches up to `--torch-compile-max-bs`.
- `python/sglang/srt/layers/attention/trtllm_mla_backend.py:71-205` – `TRTLLMMLABackend` integrates FlashInfer/TRT-LLM kernels, workspaces, and KV index construction for MLA attention.
- `python/sglang/srt/entrypoints/http_server.py:479-515` – HTTP `/generate` endpoint that forwards requests into the runtime.
- `python/sglang/srt/managers/tokenizer_manager.py:367-411` – `TokenizerManager.generate_request()` tokenises input, handles streaming, and dispatches work to schedulers.
- `python/sglang/srt/managers/tokenizer_manager.py:809-844` – `_send_one_request()` sends tokenised payloads over ZMQ to the scheduler process.

## Execution Narrative

### 1. CLI Entry and Argument Preparation
1. Running `python3 -m sglang.launch_server` drops into `python/sglang/launch_server.py:20-28`, which immediately calls `prepare_server_args()` and then `launch_server()`.
2. `prepare_server_args()` (`python/sglang/srt/server_args.py:3055-3089`) merges CLI flags with optional config files, normalises booleans, and materialises a `ServerArgs` dataclass. Flags in the example command populate:
   - `tp_size=8`, coordinating eight tensor-parallel ranks.
   - `attention_backend="trtllm_mla"`, later used by the model runner to select the TRT-LLM MLA kernels.
   - `trust_remote_code=True`, allowing Hugging Face model files to execute custom `modeling_*.py` logic.
   - `enable_torch_compile=True` with `torch_compile_max_bs=16`, which constrains compilation/batch capture logic.

### 2. Server Bootstrap and Component Spawn
1. `launch_server()` (`python/sglang/srt/entrypoints/http_server.py:1198-1278`) installs tracing/metrics middleware, then delegates to `_launch_subprocesses()` to configure runtime components.
2. `_launch_subprocesses()` (`python/sglang/srt/entrypoints/engine.py:754-901`) performs environment validation, resolves model/tokeniser paths via `prepare_model_and_tokenizer()` (`python/sglang/srt/utils.py:1066-1078`), and allocates IPC endpoints (`PortArgs`).
3. For `tp_size=8`, it spawns eight scheduler processes (`run_scheduler_process`) plus a detokeniser process; the tokenizer manager stays in the main process to serve HTTP requests. Each scheduler process receives a distinct `gpu_id`, `tp_rank`, and MoE expert-parallel rank derived from the CLI arguments.

### 3. Scheduler Bring-up and Worker Launch
1. `run_scheduler_process()` (`python/sglang/srt/managers/scheduler.py:2781-2870`) sets process titles, CPU affinity, tracing, and enters the scheduler event loop after instantiating `Scheduler`.
2. Inside `Scheduler.__init__()` (`python/sglang/srt/managers/scheduler.py:227-520`):
   - Model metadata is captured in `ModelConfig.from_server_args`, preserving tensor-parallel (`tp_size=8`) and attention backend selections.
   - ZMQ sockets are set up to receive tokenised requests from the tokenizer manager and send results back.
   - A `TpModelWorker` is created on the assigned GPU. This is where NCCL groups and KV cache pools are prepared; the scheduler retains handles to memory allocators for later request scheduling.

### 4. ModelRunner Initialisation and Weight Loading
1. `TpModelWorker.__init__()` (`python/sglang/srt/managers/tp_worker.py:61-198`) wraps `ModelRunner`, feeding in the shard’s `tp_rank`, `gpu_id`, and distributed settings.
2. `ModelRunner.__init__()` (`python/sglang/srt/model_executor/model_runner.py:208-420`) establishes distributed groups (`init_torch_distributed`), configures optional offloading, and calls `initialize()` to ready memory pools and LoRA/double-sparsity support.
3. Weight loading happens inside `ModelRunner.load_model()` (`python/sglang/srt/model_executor/model_runner.py:722-845`):
   - A `LoadConfig` is assembled, respecting tensor-parallel slicing and download directories.
   - `get_model()` pulls model code/weights; with `trust_remote_code=True`, any repository-provided `modeling` modules are executed to instantiate DeepSeek-specific layers.
   - The model is wrapped in tensor-parallel utilities, quantisation hooks, and FP8 KV cache scaling if requested. All eight tensor-parallel ranks synchronise at a monitored barrier to ensure consistent load completion.

### 5. Attention Backend: TensorRT-LLM MLA
1. During `ModelRunner.initialize()`, attention kernels are set by `_get_attention_backend()`; because the CLI specified `trtllm_mla`, the registry returns `TRTLLMMLABackend`.
2. `TRTLLMMLABackend` (`python/sglang/srt/layers/attention/trtllm_mla_backend.py:71-205`) prepares FlashInfer/TRT-LLM workspaces, enforces block alignment constraints, and builds KV index tensors that map the runtime’s paged KV cache into TRT-LLM expectations. These buffers are reused during CUDA graph capture and decode steps.

### 6. torch.compile and CUDA Graph Integration
1. `CudaGraphRunner.__init__()` (`python/sglang/srt/model_executor/cuda_graph_runner.py:212-360`) is invoked from `ModelRunner.init_device_graphs()` once the model is ready.
2. With `enable_torch_compile=True`, the runner captures decode graphs for batch sizes up to `torch_compile_max_bs=16`, applies `torch.compile` for those batches, and preallocates graph input/output tensors on the GPU. This reduces launch overhead for steady-state decode loops while keeping larger dynamic batches on the eager path.

### 7. Request Lifecycle from HTTP to GPU
1. Incoming `/generate` HTTP calls (`python/sglang/srt/entrypoints/http_server.py:479-515`) are handed to `TokenizerManager.generate_request()`.
2. `TokenizerManager.generate_request()` (`python/sglang/srt/managers/tokenizer_manager.py:367-411`) normalises batching, tokenises text/image inputs (respecting multimodal processors), and forwards token ID batches via `_send_one_request()` (`python/sglang/srt/managers/tokenizer_manager.py:809-844`) over ZMQ to the appropriate scheduler.
3. The scheduler pulls requests, enqueues them into continuous batching structures, and when conditions are met, constructs `ModelWorkerBatch` objects that describe prefill/decode workloads destined for the GPU.
4. `TpModelWorker.forward_batch_generation()` invokes `ModelRunner.forward()` (`python/sglang/srt/model_executor/model_runner.py:1860-1999`). Depending on the stage:
   - Prefill/extend passes initialise attention metadata (with `TRTLLMMLABackend`) and either replay a captured CUDA graph or execute fused kernels generated by `torch.compile`.
   - Decode passes produce logits which are fed into sampling utilities; sampled token IDs are returned to the scheduler.
5. Results flow back through the detokenizer (if enabled) to the tokenizer manager, which streams tokens to the client and updates request state.

### 8. GPU Execution Details
1. Prefill kernels build KV cache entries using the TRT-LLM MLA backend, which interacts with FlashInfer utilities to populate block indices efficiently (`python/sglang/srt/layers/attention/trtllm_mla_backend.py:154-205`).
2. Decode iterations reuse cached metadata; when CUDA graphs are applicable, `CudaGraphRunner.replay()` runs pre-captured graphs that include both attention and MLP compute, benefiting from the `torch.compile`-optimised kernels.
3. KV cache memory is provided by the allocator initialised in `Scheduler.init_memory_pool_and_cache()`, ensuring sliding-window or hybrid cache policies honour the model’s context length and the CLI’s `kv_cache_dtype`/page size choices.

## End-to-End Summary
- The CLI command is parsed into `ServerArgs`, encoding tensor-parallel width, backend choices, and compilation flags.
- `launch_server()` spawns tokenizer, scheduler, and detokenizer components; each scheduler initialises a `ModelRunner` shard on its GPU, loads Dense/MLA weights with Hugging Face utilities, and synchronises NCCL groups.
- TRT-LLM MLA attention is wired through `TRTLLMMLABackend`, while `CudaGraphRunner` captures CUDA graphs and drives `torch.compile` for decode workloads up to batch size 16.
- Runtime requests travel from FastAPI → TokenizerManager → Scheduler → ModelRunner; prefill and decode steps run on the GPU using the TRT-LLM kernels and compiled graphs, with outputs streamed back to the client via the tokenizer/detokenizer managers.
