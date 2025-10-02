# SGLang DeepSeek-V3.1 服务器启动与推理完整流程分析

> **命令**: `python3 -m sglang.launch_server --tp 8 --model deepseek-ai/DeepSeek-V3.1-Terminus --attention-backend trtllm_mla --trust-remote-code --enable-torch-compile --torch-compile-max-bs 16`

---

## 目录

1. [命令行解析与参数准备](#阶段1-命令行解析与参数准备)
2. [多进程架构初始化](#阶段2-多进程架构初始化)
3. [分布式环境初始化](#阶段3-分布式环境初始化-tensor-parallelism)
4. [模型加载与权重分片](#阶段4-模型加载与权重分片)
5. [Attention Backend初始化](#阶段5-attention-backend初始化-trtllm_mla)
6. [KV Cache分配](#阶段6-kv-cache分配)
7. [CUDA Graph Capture](#阶段7-cuda-graph-capture)
8. [推理请求执行流程](#阶段8-推理请求执行流程)
9. [关键优化技术总结](#关键优化技术总结)

---

## 阶段1: 命令行解析与参数准备

### 入口点
**文件**: `python/sglang/launch_server.py`

```python
# python/sglang/launch_server.py:20-30
if __name__ == "__main__":
    server_args = prepare_server_args(sys.argv[1:])

    from sglang.srt.server_args import print_deprecated_warning
    print_deprecated_warning(MOVE_ENVS_WARN)

    try:
        launch_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
```

### 参数解析函数
**文件**: `python/sglang/srt/server_args.py`

```python
# server_args.py:3055-3085
def prepare_server_args(argv: List[str]) -> ServerArgs:
    """
    准备服务器参数，从命令行参数解析
    """
    # 检查配置文件
    if "--config" in argv:
        # 合并配置文件和CLI参数
        config_merger = ConfigArgumentMerger(boolean_actions=boolean_actions)
        argv = config_merger.merge_config_with_args(argv)

    # 解析参数
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args(argv)

    # 返回ServerArgs对象
    return ServerArgs.from_cli_args(args)
```

**关键参数类定义**:
```python
# server_args.py:140-200
@dataclasses.dataclass
class ServerArgs:
    # 模型配置
    model_path: str                    # deepseek-ai/DeepSeek-V3.1-Terminus
    tokenizer_path: Optional[str]

    # 并行配置
    tp_size: int = 1                   # --tp 8

    # Attention配置
    attention_backend: str = "flashinfer"  # --attention-backend trtllm_mla

    # 编译优化
    enable_torch_compile: bool = False     # --enable-torch-compile
    torch_compile_max_bs: int = 32        # --torch-compile-max-bs 16

    # 信任远程代码
    trust_remote_code: bool = False       # --trust-remote-code
```

**支持的Attention Backends**:
```python
# server_args.py:88-108
ATTENTION_BACKEND_CHOICES = [
    # Common
    "triton",
    "torch_native",
    "flex_attention",
    # NVIDIA specific
    "cutlass_mla",
    "fa3",
    "fa4",
    "flashinfer",
    "flashmla",
    "trtllm_mla",      # ← 我们使用的backend
    "trtllm_mha",
    "dual_chunk_flash_attn",
    # AMD specific
    "aiter",
    "wave",
    # Other platforms
    "intel_amx",
    "ascend",
]
```

---

## 阶段2: 多进程架构初始化

### HTTP Server启动
**文件**: `python/sglang/srt/entrypoints/http_server.py`

```python
# http_server.py:1198-1248
def launch_server(
    server_args: ServerArgs,
    pipe_finish_writer: Optional[multiprocessing.connection.Connection] = None,
    launch_callback: Optional[Callable[[], None]] = None,
):
    """
    启动SRT (SGLang Runtime) Server

    架构:
    - HTTP Server: FastAPI服务器，路由请求到引擎
    - Engine包含三个组件:
        1. TokenizerManager: 分词并发送请求到调度器
        2. Scheduler (子进程): 调度批次，执行forward，发送输出到Detokenizer
        3. DetokenizerManager (子进程): 解码输出token并返回给TokenizerManager

    IPC: 通过ZMQ库进行进程间通信
    """
    if server_args.tokenizer_worker_num > 1:
        port_args = PortArgs.init_new(server_args)
        port_args.tokenizer_worker_ipc_name = (
            f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}"
        )
        tokenizer_manager, template_manager, scheduler_info = _launch_subprocesses(
            server_args=server_args, port_args=port_args
        )
    else:
        tokenizer_manager, template_manager, scheduler_info = _launch_subprocesses(
            server_args=server_args,
        )
```

### 子进程启动
**文件**: `python/sglang/srt/entrypoints/engine.py`

```python
# engine.py:380-500
def _launch_subprocesses(
    server_args: ServerArgs,
    port_args: Optional[PortArgs] = None,
) -> Tuple[TokenizerManager, TemplateManager, SchedulerInfo]:
    """
    启动调度器和解码器子进程
    """
    # 准备模型和分词器
    model_path, tokenizer_path = prepare_model_and_tokenizer(
        server_args.model_path,
        server_args.tokenizer_path,
    )

    # 创建通信端口
    if port_args is None:
        port_args = PortArgs.init_new(server_args)

    # 启动调度器进程
    scheduler_procs = []
    for dp_rank in range(server_args.dp_size):
        proc = multiprocessing.Process(
            target=run_scheduler_process,  # ← 调度器进程入口
            args=(server_args, port_args, dp_rank),
        )
        proc.start()
        scheduler_procs.append(proc)

    # 启动解码器进程
    detoken_procs = []
    for dp_rank in range(server_args.dp_size):
        proc = multiprocessing.Process(
            target=run_detokenizer_process,  # ← 解码器进程入口
            args=(server_args, port_args, dp_rank),
        )
        proc.start()
        detoken_procs.append(proc)

    # 创建TokenizerManager (主进程)
    tokenizer_manager = TokenizerManager(server_args, port_args)

    return tokenizer_manager, template_manager, scheduler_info
```

### 进程架构图

```
┌─────────────────────────────────────────────────────────────┐
│                      Main Process                            │
│  ┌────────────────────┐      ┌──────────────────────┐       │
│  │  FastAPI Server    │──────│ TokenizerManager     │       │
│  │  (HTTP Endpoints)  │      │ - encode()           │       │
│  └────────────────────┘      │ - decode()           │       │
│                               └──────────┬───────────┘       │
└────────────────────────────────────────┬─┼───────────────────┘
                                         │ │
                          ZMQ IPC        │ │ ZMQ IPC
                          (port 30000)   │ │ (port 30002)
                                         ↓ ↓
┌─────────────────────────────────────────────────────────────┐
│              Scheduler Process (子进程)                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Scheduler                                           │   │
│  │  - event_loop_normal()                               │   │
│  │  - schedule_policy.get_next_batch()                  │   │
│  │  - model_runner.forward()      ← GPU计算在这里       │   │
│  └────────────────────────┬─────────────────────────────┘   │
└─────────────────────────────┼───────────────────────────────┘
                              │ ZMQ IPC (port 30001)
                              ↓
┌─────────────────────────────────────────────────────────────┐
│          DetokenizerManager Process (子进程)                 │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  - 接收output tokens                                  │   │
│  │  - tokenizer.decode()                                │   │
│  │  - 发送回TokenizerManager                            │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 阶段3: 分布式环境初始化 (Tensor Parallelism)

### Scheduler进程入口
**文件**: `python/sglang/srt/managers/scheduler.py`

```python
# scheduler.py:2400-2500
def run_scheduler_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    dp_rank: Optional[int] = None,
):
    """
    调度器进程主函数
    """
    setproctitle.setproctitle(f"sglang::scheduler_{dp_rank}")

    # 创建调度器
    scheduler = Scheduler(server_args, port_args, dp_rank)

    # 进入事件循环
    scheduler.event_loop_normal()
```

### Scheduler初始化与ModelRunner创建
**文件**: `python/sglang/srt/managers/scheduler.py`

```python
# scheduler.py:500-700
class Scheduler:
    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        dp_rank: Optional[int] = None,
    ):
        # 创建ModelRunner (TP workers)
        if server_args.enable_overlap:
            # 使用overlap模式 (prefill和decode并行)
            self.tp_worker = TpModelWorkerClient(...)
        else:
            # 标准模式
            self.tp_worker = TpModelWorker(
                server_args=server_args,
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                tp_size=server_args.tp_size,  # ← TP=8
                dp_size=server_args.dp_size,
                nccl_port=port_args.nccl_port,
            )
```

### TpModelWorker创建ModelRunner
**文件**: `python/sglang/srt/managers/tp_worker.py`

```python
# tp_worker.py:100-200
class TpModelWorker:
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        tp_size: int,
        ...
    ):
        # 创建模型配置
        self.model_config = ModelConfig(
            path=server_args.model_path,
            trust_remote_code=server_args.trust_remote_code,
            ...
        )

        # 创建ModelRunner (这里是核心)
        self.model_runner = ModelRunner(
            model_config=self.model_config,
            mem_fraction_static=mem_fraction_static,
            gpu_id=gpu_id,
            tp_rank=tp_rank,        # 0-7 for TP=8
            tp_size=tp_size,        # 8
            moe_ep_rank=moe_ep_rank,
            moe_ep_size=moe_ep_size,
            pp_rank=pp_rank,
            pp_size=pp_size,
            nccl_port=nccl_port,
            server_args=server_args,
        )
```

### ModelRunner初始化分布式环境
**文件**: `python/sglang/srt/model_executor/model_runner.py`

```python
# model_runner.py:205-292
class ModelRunner:
    """
    ModelRunner负责运行模型的forward passes
    """
    def __init__(
        self,
        model_config: ModelConfig,
        mem_fraction_static: float,
        gpu_id: int,
        tp_rank: int,      # 当前rank: 0-7
        tp_size: int,      # TP大小: 8
        moe_ep_rank: int,
        moe_ep_size: int,
        pp_rank: int,
        pp_size: int,
        nccl_port: int,
        server_args: ServerArgs,
        ...
    ):
        # 存储配置
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.device = server_args.device  # "cuda"

        # 初始化Torch分布式
        min_per_gpu_memory = self.init_torch_distributed()

        # 初始化模型、KV cache等
        self.initialize(min_per_gpu_memory)
```

### Torch分布式初始化
```python
# model_runner.py:615-720
def init_torch_distributed(self) -> float:
    """
    初始化PyTorch分布式环境
    返回: 每个GPU的最小可用内存
    """
    # 设置backend
    backend = "nccl" if self.device == "cuda" else "gloo"

    # 分布式初始化地址
    if self.server_args.dist_init_addr:
        dist_init_method = f"tcp://{self.server_args.dist_init_addr}"
    else:
        dist_init_method = f"tcp://127.0.0.1:{self.dist_port}"

    # 设置自定义all-reduce
    set_custom_all_reduce(not self.server_args.disable_custom_all_reduce)
    set_mscclpp_all_reduce(self.server_args.enable_mscclpp)

    # 初始化分布式环境
    init_distributed_environment(
        backend=backend,
        world_size=self.tp_size * self.pp_size,  # 8 * 1 = 8
        rank=self.tp_size * self.pp_rank + self.tp_rank,  # 0-7
        local_rank=self.gpu_id,
        distributed_init_method=dist_init_method,
        timeout=self.server_args.dist_timeout,
    )

    # 初始化模型并行
    initialize_model_parallel(
        tensor_model_parallel_size=self.tp_size,      # 8
        pipeline_model_parallel_size=self.pp_size,    # 1
        expert_model_parallel_size=self.moe_ep_size,  # 1
        duplicate_tp_group=self.server_args.enable_pdmux,
    )

    # 初始化DP attention (如果启用)
    initialize_dp_attention(
        server_args=self.server_args,
        model_config=self.model_config,
    )

    # 获取可用内存
    min_per_gpu_memory = get_available_gpu_memory(
        self.device,
        self.gpu_id,
        distributed=get_world_group().world_size > 1,
        cpu_group=get_world_group().cpu_group,
    )

    # TP内存平衡检查
    self.tp_group = get_tp_group()
    self.pp_group = get_pp_group()
    self.attention_tp_group = get_attention_tp_group()

    logger.info(
        f"Init torch distributed ends. "
        f"mem usage={(before_avail_memory - local_gpu_memory):.2f} GB"
    )
    return min_per_gpu_memory
```

**相关函数**:
**文件**: `python/sglang/srt/distributed/__init__.py`

```python
# distributed/__init__.py:100-200
def init_distributed_environment(
    backend: str,
    world_size: int,
    rank: int,
    local_rank: int,
    distributed_init_method: str,
    timeout: Optional[timedelta] = None,
):
    """
    初始化torch.distributed
    """
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend=backend,
            world_size=world_size,
            rank=rank,
            init_method=distributed_init_method,
            timeout=timeout or timedelta(minutes=30),
        )

    # 设置CUDA device
    if backend == "nccl":
        torch.cuda.set_device(local_rank)

def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
    ...
):
    """
    初始化模型并行组

    对于TP=8, PP=1, EP=1:
    - 创建1个world group: [0,1,2,3,4,5,6,7]
    - 创建1个TP group: [0,1,2,3,4,5,6,7]
    - 创建8个PP group: [0], [1], ..., [7]
    """
    # 创建TP group
    for i in range(pipeline_model_parallel_size):
        ranks = list(range(
            i * tensor_model_parallel_size,
            (i + 1) * tensor_model_parallel_size
        ))
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _TP_GROUP = group
```

---

## 阶段4: 模型加载与权重分片

### ModelRunner初始化函数
```python
# model_runner.py:302-400
def initialize(self, min_per_gpu_memory: float):
    """
    初始化模型、KV cache、attention backend等
    """
    server_args = self.server_args

    # Memory saver adapter
    self.memory_saver_adapter = TorchMemorySaverAdapter.create(
        server_args, self.device, self.gpu_id
    )

    # 加载模型
    self.load_model()

    # 初始化采样器
    self.init_sampler()

    # 初始化attention backend
    self.init_attention_backend()

    # 初始化KV cache
    self.init_memory_pool(
        total_gpu_memory=min_per_gpu_memory,
        available_gpu_memory=available_gpu_memory,
    )

    # 初始化CUDA graphs
    self.init_device_graphs()
```

### 模型加载
```python
# model_runner.py:722-840
def load_model(self):
    """
    加载模型权重
    """
    before_avail_memory = get_available_gpu_memory(self.device, self.gpu_id)
    logger.info(
        f"Load weight begin. "
        f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
    )

    # 设置单线程加载 (减少冲突)
    if self.device != "cpu":
        torch.set_num_threads(1)

    # 检查CUDA计算能力
    if self.device == "cuda":
        if torch.cuda.get_device_capability()[0] < 8:
            logger.info("Compute capability below sm80. Use float16")
            self.server_args.dtype = "float16"
            self.model_config.dtype = torch.float16

    set_cuda_arch()

    # 准备加载配置
    self.load_config = LoadConfig(
        load_format=self.server_args.load_format,  # "auto"
        download_dir=self.server_args.download_dir,
        model_loader_extra_config=self.server_args.model_loader_extra_config,
        tp_rank=self.tp_rank,  # 0-7
        ...
    )

    # 加载模型
    monkey_patch_vllm_parallel_state()
    monkey_patch_isinstance_for_vllm_base_layer()

    with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_WEIGHTS):
        self.model = get_model(
            model_config=self.model_config,
            load_config=self.load_config,
            device_config=DeviceConfig(self.device, self.gpu_id),
        )

    monkey_patch_vllm_parallel_state(reverse=True)
    monkey_patch_isinstance_for_vllm_base_layer(reverse=True)

    # 后处理
    get_offloader().post_init()

    # 加载KV cache缩放因子 (如果使用FP8)
    if self.server_args.kv_cache_dtype == "fp8_e4m3":
        if self.server_args.quantization_param_path is not None:
            if callable(getattr(self.model, "load_kv_cache_scales", None)):
                self.model.load_kv_cache_scales(
                    self.server_args.quantization_param_path
                )

    # 应用torchao量化 (如果配置)
    if self.server_args.torchao_config:
        apply_torchao_config_to_model(
            self.model,
            self.server_args.torchao_config,
            self.model_config.quantization,
        )

    # Torch TP (如果启用)
    if self.server_args.enable_torch_tp:
        self.apply_torch_tp()

    logger.info(
        f"Load weight end. "
        f"type={type(self.model).__name__}, "
        f"dtype={self.model_config.dtype}, "
        f"mem usage={(before_avail_memory - after_avail_memory):.2f} GB, "
        f"avail mem={after_avail_memory:.2f} GB"
    )
```

### get_model函数
**文件**: `python/sglang/srt/model_loader/__init__.py`

```python
# model_loader/__init__.py:21-31
def get_model(
    *,
    model_config: ModelConfig,
    load_config: LoadConfig,
    device_config: DeviceConfig,
) -> nn.Module:
    """
    加载模型
    """
    loader = get_model_loader(load_config)
    return loader.load_model(
        model_config=model_config,
        device_config=device_config,
    )
```

### 模型加载器
**文件**: `python/sglang/srt/model_loader/loader.py`

```python
# loader.py:200-400
def get_model_loader(load_config: LoadConfig) -> BaseModelLoader:
    """
    根据load_format返回对应的loader
    """
    if load_config.load_format == LoadFormat.AUTO:
        return DefaultModelLoader(load_config)
    elif load_config.load_format == LoadFormat.DUMMY:
        return DummyModelLoader(load_config)
    ...
    return DefaultModelLoader(load_config)

class DefaultModelLoader(BaseModelLoader):
    """
    默认模型加载器
    """
    def load_model(
        self,
        *,
        model_config: ModelConfig,
        device_config: DeviceConfig,
    ) -> nn.Module:
        # 1. 获取模型架构类
        model_class = get_model_architecture(model_config)[0]

        # 2. 设置默认dtype
        with set_default_torch_dtype(model_config.dtype):
            # 3. 实例化模型 (不加载权重)
            with torch.device(device_config.device):
                model = model_class(
                    config=model_config.hf_config,
                    cache_config=...,
                    quant_config=...,
                )

        # 4. 加载权重到模型
        model.load_weights(
            weights_iter=self.get_all_weights(model_config),
        )

        return model.eval()

    def get_all_weights(
        self,
        model_config: ModelConfig,
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        """
        迭代返回所有权重
        对于TP=8, 每个rank只加载1/8的权重
        """
        if self.load_config.load_format == LoadFormat.SAFETENSORS:
            # 使用safetensors格式
            for name, param in self._get_weights_from_safetensors(...):
                # TP切分逻辑在这里
                yield name, param
        elif self.load_config.load_format == LoadFormat.PT:
            # 使用pytorch格式
            ...
```

### 模型架构识别
**文件**: `python/sglang/srt/model_loader/utils.py`

```python
# model_loader/utils.py:50-150
def get_model_architecture(model_config: ModelConfig) -> Tuple[Type[nn.Module], str]:
    """
    从config.json中识别模型架构

    对于DeepSeek-V3:
    - architectures: ["DeepseekV3ForCausalLM"]
    - 返回对应的模型类
    """
    architectures = getattr(model_config.hf_config, "architectures", [])

    # 查找支持的架构
    for arch in architectures:
        model_cls = _get_model_architecture(model_config, arch)
        if model_cls is not None:
            return model_cls, arch

    raise ValueError(f"Unsupported architectures: {architectures}")

def _get_model_architecture(
    model_config: ModelConfig,
    arch: str
) -> Optional[Type[nn.Module]]:
    """
    根据架构名称返回模型类
    """
    # DeepSeek模型映射
    module_name = arch  # "DeepseekV3ForCausalLM"
    module_path = f"sglang.srt.models.{arch.lower()}"

    try:
        module = importlib.import_module(module_path)
        return getattr(module, module_name)
    except (ImportError, AttributeError):
        return None
```

### DeepSeek-V3模型定义
**文件**: `python/sglang/srt/models/deepseek*.py`

```python
# 模型文件结构 (示例, 具体文件名可能不同)
class DeepseekV3ForCausalLM(nn.Module):
    """
    DeepSeek-V3 带MLA的因果语言模型
    """
    def __init__(self, config, cache_config, quant_config):
        super().__init__()

        self.config = config

        # Embedding层
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )

        # Transformer层
        self.layers = nn.ModuleList([
            DeepseekV3DecoderLayer(config, cache_config, quant_config)
            for _ in range(config.num_hidden_layers)
        ])

        # 输出层
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
        )

    def load_weights(self, weights_iter):
        """
        加载权重, TP切分在这里发生
        """
        stacked_params_mapping = [...]
        params_dict = dict(self.named_parameters())

        for name, loaded_weight in weights_iter:
            # TP切分逻辑
            if "q_proj" in name or "k_proj" in name:
                # QKV投影按head维度切分
                shard_size = loaded_weight.shape[0] // tp_size
                loaded_weight = loaded_weight[
                    tp_rank * shard_size : (tp_rank + 1) * shard_size
                ]

            # 加载到模型
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)

class DeepseekV3DecoderLayer(nn.Module):
    """
    DeepSeek-V3 解码器层
    包含: MLA Attention + MoE FFN
    """
    def __init__(self, config, cache_config, quant_config):
        # MLA Attention
        self.self_attn = DeepseekV3Attention(
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
        )

        # MoE FFN
        self.mlp = DeepseekV3MoE(
            config=config,
            quant_config=quant_config,
        )

    def forward(self, hidden_states, ...):
        # Self-attention
        residual = hidden_states
        hidden_states = self.self_attn(hidden_states, ...)
        hidden_states = residual + hidden_states

        # MoE
        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
```

---

## 阶段5: Attention Backend初始化 (trtllm_mla)

### 初始化Attention Backend
```python
# model_runner.py:1050-1150
def init_attention_backend(self):
    """
    初始化attention backend
    """
    # 选择backend
    if self.server_args.attention_backend:
        backend_name = self.server_args.attention_backend  # "trtllm_mla"
    else:
        # 自动选择
        backend_name = self._auto_select_attention_backend()

    # 获取backend类
    backend_cls = ATTENTION_BACKENDS.get(backend_name)

    # 实例化backend
    self.attn_backend = backend_cls(
        model_runner=self,
        skip_prefill=self.server_args.skip_prefill_attention_backend,
        kv_indptr_buf=...,
        q_indptr_decode_buf=...,
    )

    logger.info(f"Attention backend: {backend_name}")
```

### Attention Backend注册表
**文件**: `python/sglang/srt/layers/attention/attention_registry.py`

```python
# attention_registry.py:50-150
ATTENTION_BACKENDS: Dict[str, Type[BaseAttentionBackend]] = {
    "triton": TritonAttnBackend,
    "torch_native": TorchNativeAttnBackend,
    "flashinfer": FlashInferAttnBackend,
    "flashmla": FlashMLAAttnBackend,
    "trtllm_mla": TRTLLMMLABackend,    # ← 我们使用的
    "trtllm_mha": TRTLLMBackend,
    "fa3": FA3Backend,
    "fa4": FA4Backend,
    ...
}

def attn_backend_wrapper(backend_name: str):
    """
    Decorator to register attention backend
    """
    def decorator(cls):
        ATTENTION_BACKENDS[backend_name] = cls
        return cls
    return decorator
```

### TRTLLM MLA Backend实现
**文件**: `python/sglang/srt/layers/attention/trtllm_mla_backend.py`

```python
# trtllm_mla_backend.py:70-200
class TRTLLMMLABackend(FlashInferMLAAttnBackend):
    """
    TensorRT-LLM MLA attention kernel from flashinfer

    MLA (Multi-head Latent Attention):
    - 使用低秩分解压缩KV cache
    - KV cache = KV_lora @ V_compressed
    - 大幅减少显存占用
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
        q_indptr_decode_buf: Optional[torch.Tensor] = None,
    ):
        super().__init__(
            model_runner,
            skip_prefill,
            kv_indptr_buf,
            q_indptr_decode_buf,
        )

        config = model_runner.model_config

        # 模型参数
        self.num_q_heads = config.num_attention_heads // get_attention_tp_size()
        self.num_kv_heads = config.get_num_kv_heads(get_attention_tp_size())
        self.num_local_heads = config.num_attention_heads // get_attention_tp_size()

        # MLA特定维度
        self.kv_lora_rank = config.kv_lora_rank      # 压缩秩
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.kv_cache_dim = self.kv_lora_rank + self.qk_rope_head_dim

        # Runtime参数
        self.scaling = config.scaling
        self.data_type = model_runner.kv_cache_dtype  # fp8_e4m3 or bf16
        self.q_data_type = model_runner.dtype
        self.page_size = model_runner.page_size
        self.req_to_token = model_runner.req_to_token_pool.req_to_token

        # Workspace分配 (128MB)
        self.workspace_size = DEFAULT_WORKSPACE_SIZE_MB * 1024 * 1024
        global global_zero_init_workspace_buffer
        if global_zero_init_workspace_buffer is None:
            global_zero_init_workspace_buffer = torch.zeros(
                self.workspace_size,
                dtype=torch.uint8,
                device=model_runner.device,
            )
        self.workspace_buffer = global_zero_init_workspace_buffer

        # CUDA graph state
        self.decode_cuda_graph_metadata = {}
        self.decode_cuda_graph_kv_indices = None
        self.forward_prefill_metadata: Optional[TRTLLMMLAPrefillMetadata] = None
        self.forward_decode_metadata: Union[TRTLLMMLADecodeMetadata, None] = None
```

### Prefill阶段的Attention
```python
# trtllm_mla_backend.py:400-500
def forward_prefill(
    self,
    q: torch.Tensor,              # [total_tokens, num_heads, head_dim]
    k: torch.Tensor,              # [total_tokens, num_kv_heads, kv_lora_rank + rope_dim]
    v: torch.Tensor,              # [total_tokens, num_kv_heads, v_head_dim]
    layer_id: int,
    forward_batch: ForwardBatch,
) -> torch.Tensor:
    """
    Prefill阶段的MLA attention
    使用FlashInfer的TRT-LLM kernel
    """
    # 获取metadata
    metadata = self.forward_prefill_metadata

    # 调用FlashInfer kernel
    if self.disable_chunked_prefix_cache:
        # 标准prefill
        output = flashinfer.trtllm_prefill_mla(
            q=q,
            kv_data=self.kv_pool[layer_id],
            cum_seq_lens=metadata.cum_seq_lens,
            max_seq_len=metadata.max_seq_len,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            scaling=self.scaling,
            workspace_buffer=self.workspace_buffer,
        )
    else:
        # Chunked prefill (分块处理长序列)
        output = flashinfer.trtllm_prefill_mla_chunked(...)

    return output
```

### Decode阶段的Attention
```python
# trtllm_mla_backend.py:600-700
def forward_decode(
    self,
    q: torch.Tensor,              # [batch_size, num_heads, head_dim]
    k: torch.Tensor,
    v: torch.Tensor,
    layer_id: int,
    forward_batch: ForwardBatch,
) -> torch.Tensor:
    """
    Decode阶段的MLA attention
    每次只生成一个token, 复用KV cache
    """
    # 获取metadata
    metadata = self.forward_decode_metadata

    # 调用TRT-LLM MLA decode kernel
    output = flashinfer.trtllm_decode_mla(
        q=q,                                    # [bs, num_heads, head_dim]
        kv_cache=self.kv_pool[layer_id],       # Paged KV cache
        block_kv_indices=metadata.block_kv_indices,  # Page table
        max_seq_len=metadata.max_seq_len,
        kv_lora_rank=self.kv_lora_rank,
        qk_rope_head_dim=self.qk_rope_head_dim,
        v_head_dim=self.v_head_dim,
        scaling=self.scaling,
        workspace_buffer=self.workspace_buffer,
    )

    return output
```

**FlashInfer TRT-LLM Kernel特点**:
1. **Fused操作**: QK^T, softmax, @V在一个kernel中
2. **Flash Attention 3**: 利用Hopper架构的Tensor Memory Accelerator (TMA)
3. **MLA压缩**: KV cache使用低秩分解, 节省50-70%显存
4. **Paged Attention**: 使用page table管理KV cache, 减少碎片

---

## 阶段6: KV Cache分配

### 初始化Memory Pool
```python
# model_runner.py:1100-1300
def init_memory_pool(
    self,
    total_gpu_memory: float,
    available_gpu_memory: float,
):
    """
    初始化KV cache和request-to-token映射池
    """
    # 计算可用于KV cache的内存
    if self.server_args.mem_fraction_static is not None:
        kv_cache_memory = total_gpu_memory * self.server_args.mem_fraction_static
    else:
        # 自动计算
        kv_cache_memory = available_gpu_memory * 0.9  # 90%用于KV cache

    # 创建ReqToTokenPool
    if self.is_hybrid:
        # Hybrid模型 (MLA + MHA混合)
        self.req_to_token_pool = HybridReqToTokenPool(...)
    else:
        # 标准模型
        self.req_to_token_pool = ReqToTokenPool(
            size=int(self.server_args.max_total_tokens / self.page_size),
            max_context_len=self.model_config.context_len + 4,
            device=self.device,
            use_records=self.use_records,
        )

    # 创建KV Pool
    if self.use_mla_backend:
        # MLA backend
        if self.device == "npu" and self.server_args.enable_swa:
            # NPU with SWA
            kv_pool_class = AscendMLAPagedTokenToKVPool
        else:
            # 标准MLA
            kv_pool_class = MLATokenToKVPool
    else:
        # MHA backend
        kv_pool_class = MHATokenToKVPool

    # 实例化KV pool
    self.token_to_kv_pool = kv_pool_class(
        num_layers=self.model_config.num_hidden_layers,
        num_kv_heads=self.model_config.get_num_kv_heads(get_attention_tp_size()),
        head_dim=self.kv_cache_dim,  # MLA: kv_lora_rank + rope_dim
        page_size=self.page_size,
        dtype=self.kv_cache_dtype,
        device=self.device,
    )

    # 创建allocator
    if self.server_args.enable_swa:
        # Sliding Window Attention
        self.token_to_kv_pool_allocator = SWATokenToKVPoolAllocator(...)
    else:
        # 标准Paged allocator
        self.token_to_kv_pool_allocator = PagedTokenToKVPoolAllocator(
            self.token_to_kv_pool,
            self.req_to_token_pool,
        )

    # RadixAttention (自动前缀缓存)
    if self.server_args.disable_radix_cache:
        self.tree_cache = None
    else:
        if self.server_args.enable_nan_detection:
            self.tree_cache = ChunkCache(
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool=self.token_to_kv_pool,
            )
        else:
            # 使用RadixCache (更高效)
            self.tree_cache = HiRadixCache(
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool=self.token_to_kv_pool,
                disable_chunked_prefix_cache=self.server_args.disable_chunked_prefix_cache,
            )

    logger.info(
        f"Memory pool initialized. "
        f"KV cache size: {kv_cache_memory:.2f} GB, "
        f"#pages: {self.token_to_kv_pool.size}"
    )
```

**相关数据结构**:

**文件**: `python/sglang/srt/mem_cache/memory_pool.py`

```python
# memory_pool.py:200-350
class MLATokenToKVPool:
    """
    MLA的Token到KV Pool映射

    KV cache存储格式:
    [num_layers, num_pages, page_size, num_kv_heads, kv_cache_dim]
    其中: kv_cache_dim = kv_lora_rank + qk_rope_head_dim
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,        # kv_lora_rank + rope_dim
        page_size: int,
        dtype: torch.dtype,
        device: str,
    ):
        # 计算总页数
        self.num_pages = self._calculate_num_pages(...)

        # 分配KV cache
        self.kv_data = [
            torch.empty(
                (self.num_pages, page_size, num_kv_heads, head_dim),
                dtype=dtype,
                device=device,
            )
            for _ in range(num_layers)
        ]

        # 空闲页列表
        self.free_page_ids = list(range(self.num_pages))

class ReqToTokenPool:
    """
    Request到Token的映射

    req_to_token[req_id] = [token_id_0, token_id_1, ..., token_id_n]
    token_id指向KV cache中的page
    """

    def __init__(self, size: int, max_context_len: int, device: str):
        # size: 最大request数
        # max_context_len: 最大上下文长度

        self.req_to_token = torch.empty(
            (size, max_context_len),
            dtype=torch.int32,
            device=device,
        )

        self.free_slots = list(range(size))
```

**RadixAttention树缓存**:

**文件**: `python/sglang/srt/mem_cache/hiradix_cache.py`

```python
# hiradix_cache.py:100-300
class HiRadixCache:
    """
    高性能RadixTree缓存, 自动共享相同前缀的KV cache

    示例:
    Request 1: "Hello, how are you?"
    Request 2: "Hello, how is the weather?"

    → "Hello, how " 的KV cache会被两个请求共享
    """

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool: BaseTokenToKVPool,
        disable_chunked_prefix_cache: bool = False,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool = token_to_kv_pool

        # Radix tree根节点
        self.root_node = TreeNode()

        # Token到Node的映射
        self.token_to_node = {}

    def match_prefix(
        self,
        token_ids: List[int],
    ) -> Tuple[TreeNode, int]:
        """
        匹配最长前缀
        返回: (匹配的节点, 匹配长度)
        """
        node = self.root_node
        matched_len = 0

        for i, token_id in enumerate(token_ids):
            if token_id in node.children:
                node = node.children[token_id]
                matched_len = i + 1
            else:
                break

        return node, matched_len

    def insert(
        self,
        token_ids: List[int],
        kv_indices: List[int],
    ):
        """
        插入新的token序列到radix tree
        """
        node = self.root_node

        for token_id, kv_idx in zip(token_ids, kv_indices):
            if token_id not in node.children:
                # 创建新节点
                new_node = TreeNode(
                    token_id=token_id,
                    kv_index=kv_idx,
                    parent=node,
                )
                node.children[token_id] = new_node
                node = new_node
            else:
                # 重用已有节点
                node = node.children[token_id]
                node.ref_count += 1
```

---

## 阶段7: CUDA Graph Capture

### 初始化Device Graphs
```python
# model_runner.py:1788-1823
def init_device_graphs(self):
    """
    Capture device graphs (CUDA/CPU)
    """
    self.graph_runner = None
    self.graph_mem_usage = 0

    if not self.is_generation:
        # CUDA graph只用于生成模型
        return

    if self.device != "cpu" and self.server_args.disable_cuda_graph:
        return

    if self.device == "cpu" and not self.server_args.enable_torch_compile:
        return

    tic = time.perf_counter()
    before_mem = get_available_gpu_memory(self.device, self.gpu_id)
    logger.info(
        f"Capture {'cpu graph' if self.device == 'cpu' else 'cuda graph'} begin. "
        f"This can take up to several minutes. "
        f"avail mem={before_mem:.2f} GB"
    )

    # 根据device选择graph runner
    graph_runners = defaultdict(
        lambda: CudaGraphRunner,
        {
            "cpu": CPUGraphRunner,
            "npu": NPUGraphRunner,
        },
    )
    self.graph_runner = graph_runners[self.device](self)

    after_mem = get_available_gpu_memory(self.device, self.gpu_id)
    self.graph_mem_usage = before_mem - after_mem
    logger.info(
        f"Capture {'cpu graph' if self.device == 'cpu' else 'cuda graph'} end. "
        f"Time elapsed: {time.perf_counter() - tic:.2f} s. "
        f"mem usage={self.graph_mem_usage:.2f} GB. "
        f"avail mem={after_mem:.2f} GB."
    )
```

### CUDA Graph Runner
**文件**: `python/sglang/srt/model_executor/cuda_graph_runner.py`

```python
# cuda_graph_runner.py:100-400
class CudaGraphRunner:
    """
    CUDA Graph runner
    预先capture常见batch size的执行图, replay时直接执行
    """

    def __init__(self, model_runner: ModelRunner):
        self.model_runner = model_runner
        self.graphs = {}  # {batch_size: CUDAGraph}
        self.input_buffers = {}
        self.output_buffers = {}

        # Torch compile设置
        self.compile_bs_range = None
        if model_runner.server_args.enable_torch_compile:
            self.compile_bs_range = list(range(
                1,
                model_runner.server_args.torch_compile_max_bs + 1  # 1-16
            ))

        # Capture graphs
        self.capture_graphs()

    def capture_graphs(self):
        """
        Capture CUDA graphs for different batch sizes
        """
        # 确定要capture的batch sizes
        if self.compile_bs_range:
            # Torch compile模式: capture所有bs
            batch_sizes = self.compile_bs_range
        else:
            # 标准模式: 只capture 2的幂次
            batch_sizes = [1, 2, 4, 8, 16, 24, 32]

        with freeze_gc(enable_cudagraph_gc=self.model_runner.server_args.enable_cudagraph_gc):
            for bs in tqdm.tqdm(batch_sizes, desc="Capturing graphs"):
                self._capture_one_graph(bs)

    def _capture_one_graph(self, batch_size: int):
        """
        Capture单个batch size的graph
        """
        # 准备dummy input
        forward_batch = self._prepare_dummy_batch(batch_size)

        # Warmup
        for _ in range(3):
            _ = self.model_runner.forward_decode(
                forward_batch,
                skip_attn_backend_init=True,
            )
        torch.cuda.synchronize()

        # Capture
        graph = torch.cuda.CUDAGraph()

        with model_capture_mode():
            with torch.cuda.graph(graph, pool=self._get_graph_pool()):
                output_buffers = self.model_runner.forward_decode(
                    forward_batch,
                    skip_attn_backend_init=True,
                )

        # 存储graph
        self.graphs[batch_size] = graph
        self.input_buffers[batch_size] = forward_batch
        self.output_buffers[batch_size] = output_buffers

        # Torch compile (如果启用)
        if self.compile_bs_range and batch_size in self.compile_bs_range:
            self._compile_graph(batch_size)

    def _compile_graph(self, batch_size: int):
        """
        使用torch.compile优化forward函数
        """
        # Monkey patch torch.compile
        monkey_patch_torch_compile()

        # Compile
        compiled_fn = torch.compile(
            self.model_runner.forward_decode,
            mode=self.model_runner.server_args.torch_compile_mode,  # "reduce-overhead"
            fullgraph=True,
            backend="inductor",
        )

        # 替换原函数
        self.compiled_forward_decode = compiled_fn

    def can_run(self, forward_batch: ForwardBatch) -> bool:
        """
        判断是否可以使用graph replay
        """
        batch_size = forward_batch.batch_size

        # 检查是否有对应的graph
        if batch_size not in self.graphs:
            return False

        # 检查其他条件 (序列长度等)
        return True

    def replay(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> LogitsProcessorOutput:
        """
        Replay CUDA graph
        """
        batch_size = forward_batch.batch_size

        # 获取对应的graph
        graph = self.graphs[batch_size]
        input_buffer = self.input_buffers[batch_size]
        output_buffer = self.output_buffers[batch_size]

        # 复制input到buffer
        self._copy_input_to_buffer(forward_batch, input_buffer)

        # Replay graph
        graph.replay()

        # 复制output
        output = self._copy_output_from_buffer(output_buffer)

        return output
```

**Torch Compile集成**:

**文件**: `python/sglang/srt/patch_torch.py`

```python
# patch_torch.py:50-150
def monkey_patch_torch_compile():
    """
    Monkey patch torch.compile以支持SGLang特定优化
    """
    original_compile = torch.compile

    def patched_compile(
        fn: Callable,
        mode: str = "default",
        fullgraph: bool = False,
        backend: str = "inductor",
        **kwargs
    ):
        """
        Patched torch.compile

        mode选项:
        - "default": 标准优化
        - "reduce-overhead": 减少Python开销
        - "max-autotune": 最大化kernel调优
        """
        if mode == "reduce-overhead":
            # 减少开销模式
            kwargs.update({
                "dynamic": False,
                "fullgraph": True,
            })
        elif mode == "max-autotune":
            # 最大调优模式
            kwargs.update({
                "dynamic": False,
                "fullgraph": True,
                "options": {
                    "max_autotune": True,
                    "max_autotune_gemm": True,
                },
            })

        return original_compile(fn, fullgraph=fullgraph, backend=backend, **kwargs)

    torch.compile = patched_compile
```

**Torch Compile优化内容**:
1. **Operator Fusion**: 将多个小算子融合成大kernel
   - 例如: LayerNorm + Linear → Fused kernel
2. **Memory Layout优化**: 自动选择最优的tensor layout
3. **Kernel自动调优**: 根据硬件特性选择最优kernel参数
4. **Python开销消除**: 将Python循环编译为C++代码

---

## 阶段8: 推理请求执行流程

### HTTP请求到达
**文件**: `python/sglang/srt/entrypoints/http_server.py`

```python
# http_server.py:700-800
@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    raw_request: Request,
):
    """
    OpenAI兼容的chat completion端点
    """
    return await raw_request.app.state.openai_serving_chat.handle_request(
        request, raw_request
    )

@app.post("/v1/completions")
async def completions(
    request: CompletionRequest,
    raw_request: Request,
):
    """
    OpenAI兼容的completion端点
    """
    return await raw_request.app.state.openai_serving_completion.handle_request(
        request, raw_request
    )

@app.post("/generate")
async def generate_request(obj: GenerateReqInput, raw_request: Request):
    """
    SGLang原生generate端点
    """
    # 参数验证
    obj.normalize_batch_and_arguments()
    obj.post_init()

    # 发送到TokenizerManager
    if obj.stream:
        # 流式响应
        async def generate_stream():
            async for chunk in tokenizer_manager.generate_request(
                obj, raw_request
            ):
                yield chunk

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
        )
    else:
        # 非流式响应
        response = await tokenizer_manager.generate_request(obj, raw_request)
        return ORJSONResponse(response)
```

### TokenizerManager处理
**文件**: `python/sglang/srt/managers/tokenizer_manager.py`

```python
# tokenizer_manager.py:200-400
class TokenizerManager:
    """
    分词管理器, 运行在主进程
    """

    def __init__(self, server_args: ServerArgs, port_args: PortArgs):
        self.server_args = server_args
        self.port_args = port_args

        # 加载分词器
        self.tokenizer = get_tokenizer(
            server_args.tokenizer_path or server_args.model_path,
            trust_remote_code=server_args.trust_remote_code,
        )

        # ZMQ socket (发送到scheduler)
        self.send_to_scheduler = self._create_zmq_socket(
            port_args.scheduler_input_ipc_name
        )

        # ZMQ socket (接收detokenizer输出)
        self.recv_from_detokenizer = self._create_zmq_socket(
            port_args.tokenizer_ipc_name
        )

    async def generate_request(
        self,
        obj: GenerateReqInput,
        request: Optional[Request] = None,
    ):
        """
        处理生成请求
        """
        # 1. Tokenization
        if obj.input_ids is None:
            input_ids = self.tokenizer.encode(obj.text)
        else:
            input_ids = obj.input_ids

        # 2. 准备请求
        tokenized_obj = TokenizedGenerateReqInput(
            rid=self._get_next_request_id(),
            input_ids=input_ids,
            sampling_params=obj.sampling_params,
            return_logprob=obj.return_logprob,
            ...
        )

        # 3. 发送到scheduler
        self.send_to_scheduler.send_pyobj(tokenized_obj)

        # 4. 等待响应
        if obj.stream:
            # 流式响应
            async for output in self._wait_for_response_stream(tokenized_obj.rid):
                yield output
        else:
            # 非流式响应
            output = await self._wait_for_response(tokenized_obj.rid)
            return output

    async def _wait_for_response(self, rid: str):
        """
        等待scheduler响应
        """
        while True:
            # 从detokenizer接收
            output = self.recv_from_detokenizer.recv_pyobj()

            if output.rid == rid:
                return output

            # 不是当前请求, 继续等待
            await asyncio.sleep(0.001)
```

### Scheduler调度循环
**文件**: `python/sglang/srt/managers/scheduler.py`

```python
# scheduler.py:1500-1800
class Scheduler:
    """
    调度器, 管理请求队列和批处理
    """

    def event_loop_normal(self):
        """
        标准事件循环
        """
        # 初始化
        self.last_batch = None
        self.running_batch: Optional[ScheduleBatch] = None
        self.waiting_queue: deque[Req] = deque()

        while True:
            # 1. 接收新请求
            recv_reqs = self.recv_requests()

            # 2. 处理新请求
            for recv_req in recv_reqs:
                if isinstance(recv_req, TokenizedGenerateReqInput):
                    # 创建Req对象
                    req = Req(
                        rid=recv_req.rid,
                        input_ids=recv_req.input_ids,
                        sampling_params=recv_req.sampling_params,
                        ...
                    )
                    self.waiting_queue.append(req)
                elif isinstance(recv_req, AbortReq):
                    # 终止请求
                    self._abort_request(recv_req.rid)

            # 3. 调度下一个batch
            schedule_batch = self.schedule_policy.get_next_batch(
                self.running_batch,
                self.waiting_queue,
                self.available_memory,
            )

            if schedule_batch is None:
                # 没有可调度的batch
                continue

            # 4. 准备ForwardBatch
            forward_batch = self._prepare_forward_batch(schedule_batch)

            # 5. 执行forward
            logits_output = self.tp_worker.forward_batch(forward_batch)

            # 6. Sampling
            next_token_ids = self._sample_next_tokens(
                logits_output,
                schedule_batch,
            )

            # 7. 更新batch状态
            self._update_batch_state(schedule_batch, next_token_ids)

            # 8. 发送完成的请求到detokenizer
            finished_reqs = schedule_batch.filter_finished()
            if finished_reqs:
                self.send_to_detokenizer(finished_reqs)

            # 9. 更新running_batch
            self.running_batch = schedule_batch if not schedule_batch.is_empty() else None

    def recv_requests(self) -> List[Union[TokenizedGenerateReqInput, AbortReq]]:
        """
        从ZMQ接收请求 (非阻塞)
        """
        recv_reqs = []

        while True:
            try:
                recv_req = self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
                recv_reqs.append(recv_req)
            except zmq.Again:
                # 没有更多请求
                break

        return recv_reqs
```

### 调度策略
**文件**: `python/sglang/srt/managers/schedule_policy.py`

```python
# schedule_policy.py:200-500
class SchedulePolicy:
    """
    调度策略: 决定下一个batch包含哪些请求

    关键算法: Continuous Batching
    - 不等待整个batch完成
    - 动态添加/移除请求
    - 最大化GPU利用率
    """

    def get_next_batch(
        self,
        running_batch: Optional[ScheduleBatch],
        waiting_queue: deque[Req],
        available_memory: int,
    ) -> Optional[ScheduleBatch]:
        """
        获取下一个要执行的batch
        """
        if running_batch is None:
            # 没有正在运行的batch, 创建新batch
            return self._create_new_batch(waiting_queue, available_memory)

        # 尝试向running batch添加新请求 (continuous batching)
        self._try_add_requests_to_batch(
            running_batch,
            waiting_queue,
            available_memory,
        )

        return running_batch

    def _create_new_batch(
        self,
        waiting_queue: deque[Req],
        available_memory: int,
    ) -> Optional[ScheduleBatch]:
        """
        创建新batch
        """
        if not waiting_queue:
            return None

        batch = ScheduleBatch(
            reqs=[],
            max_total_tokens=self.max_total_tokens,
        )

        # 贪心添加请求
        while waiting_queue:
            req = waiting_queue[0]

            # 检查是否有足够内存
            if self._can_add_request(batch, req, available_memory):
                waiting_queue.popleft()
                batch.add_request(req)
            else:
                break

        return batch if not batch.is_empty() else None

    def _try_add_requests_to_batch(
        self,
        batch: ScheduleBatch,
        waiting_queue: deque[Req],
        available_memory: int,
    ):
        """
        尝试向现有batch添加新请求 (continuous batching核心)
        """
        # Prefill adder: 决定是否可以添加prefill请求
        prefill_adder = PrefillAdder(
            tree_cache=self.tree_cache,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            max_batch_size=self.max_batch_size,
        )

        while waiting_queue:
            req = waiting_queue[0]

            # 尝试添加
            result = prefill_adder.try_add_request(batch, req, available_memory)

            if result == AddReqResult.SUCCESS:
                waiting_queue.popleft()
                batch.add_request(req)
            elif result == AddReqResult.NO_MEMORY:
                # 内存不足, 停止添加
                break
            else:
                # 其他原因, 跳过这个请求
                break
```

### Forward Batch准备
```python
# scheduler.py:1200-1400
def _prepare_forward_batch(
    self,
    schedule_batch: ScheduleBatch,
) -> ForwardBatch:
    """
    将ScheduleBatch转换为ForwardBatch
    """
    # 确定forward mode
    if schedule_batch.is_all_prefill():
        forward_mode = ForwardMode.PREFILL
    elif schedule_batch.is_all_decode():
        forward_mode = ForwardMode.DECODE
    else:
        forward_mode = ForwardMode.MIXED

    # 收集input_ids, positions等
    input_ids = []
    positions = []
    req_pool_indices = []
    seq_lens = []

    for req in schedule_batch.reqs:
        if req.is_prefill():
            # Prefill: 所有input tokens
            input_ids.extend(req.input_ids[req.filled_len:])
            positions.extend(range(req.filled_len, len(req.input_ids)))
        else:
            # Decode: 只有最后一个token
            input_ids.append(req.output_ids[-1])
            positions.append(len(req.input_ids) + len(req.output_ids) - 1)

        req_pool_indices.append(req.req_pool_idx)
        seq_lens.append(len(req.input_ids) + len(req.output_ids))

    # 转换为tensor
    forward_batch = ForwardBatch(
        input_ids=torch.tensor(input_ids, dtype=torch.int32, device=self.device),
        positions=torch.tensor(positions, dtype=torch.int64, device=self.device),
        req_pool_indices=torch.tensor(req_pool_indices, dtype=torch.int32, device=self.device),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=self.device),
        forward_mode=forward_mode,
        batch_size=len(schedule_batch.reqs),
        ...
    )

    return forward_batch
```

### TpModelWorker执行forward
**文件**: `python/sglang/srt/managers/tp_worker.py`

```python
# tp_worker.py:400-500
class TpModelWorker:
    """
    TP worker, 封装ModelRunner
    """

    def forward_batch(
        self,
        forward_batch: ForwardBatch,
    ) -> LogitsProcessorOutput:
        """
        执行batch的forward pass
        """
        # 调用ModelRunner
        logits_output, can_run_graph = self.model_runner.forward(
            forward_batch,
            skip_attn_backend_init=False,
            pp_proxy_tensors=None,
        )

        return logits_output
```

### ModelRunner.forward
**文件**: `python/sglang/srt/model_executor/model_runner.py`

```python
# model_runner.py:1941-2020
def forward(
    self,
    forward_batch: ForwardBatch,
    skip_attn_backend_init: bool = False,
    pp_proxy_tensors: Optional[PPProxyTensors] = None,
    reinit_attn_backend: bool = False,
    split_forward_count: int = 1,
) -> Tuple[Union[LogitsProcessorOutput, PPProxyTensors], bool]:
    """
    执行forward pass
    返回: (输出, 是否使用了cuda graph)
    """
    self.forward_pass_id += 1

    # 记录expert分布 (MoE)
    with get_global_expert_distribution_recorder().with_forward_pass(
        self.forward_pass_id,
        forward_batch,
    ):
        output = self._forward_raw(
            forward_batch,
            skip_attn_backend_init,
            pp_proxy_tensors,
            reinit_attn_backend,
            split_forward_count,
        )

    # EPLB (Expert Parallel Load Balancing)
    if self.eplb_manager is not None:
        self.eplb_manager.on_forward_pass_end()

    return output

def _forward_raw(
    self,
    forward_batch: ForwardBatch,
    skip_attn_backend_init: bool,
    pp_proxy_tensors: Optional[PPProxyTensors],
    reinit_attn_backend: bool = False,
    split_forward_count: int = 1,
) -> Tuple[Union[LogitsProcessorOutput, PPProxyTensors], bool]:
    """
    实际执行forward的函数
    """
    # 检查是否可以使用CUDA graph
    mode_check = (
        forward_batch.forward_mode.is_cpu_graph
        if self.device == "cpu"
        else forward_batch.forward_mode.is_cuda_graph
    )
    can_run_graph = bool(
        mode_check()
        and self.graph_runner
        and self.graph_runner.can_run(forward_batch)
    )

    if can_run_graph:
        # 使用CUDA graph replay (最快路径)
        ret = self.graph_runner.replay(
            forward_batch,
            skip_attn_backend_init=skip_attn_backend_init,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        return ret, can_run_graph

    # MLP sync (如果需要)
    if forward_batch.global_num_tokens_cpu is not None:
        forward_batch.prepare_mlp_sync_batch(self)

    # 根据mode调用不同的forward函数
    if forward_batch.forward_mode.is_decode():
        # Decode模式: 自回归生成
        ret = self.forward_decode(
            forward_batch,
            skip_attn_backend_init=skip_attn_backend_init,
            pp_proxy_tensors=pp_proxy_tensors,
        )
    elif forward_batch.forward_mode.is_extend():
        # Extend/Prefill模式: 处理input prompt
        ret = self.forward_extend(
            forward_batch,
            skip_attn_backend_init=skip_attn_backend_init,
            pp_proxy_tensors=pp_proxy_tensors,
        )
    elif forward_batch.forward_mode.is_split_prefill():
        # Split prefill: 分层处理
        ret = self.forward_split_prefill(
            forward_batch,
            reinit_attn_backend=reinit_attn_backend,
            forward_count=split_forward_count,
        )
    elif forward_batch.forward_mode.is_idle():
        # Idle: 空闲forward (pipeline parallelism)
        ret = self.forward_idle(forward_batch, pp_proxy_tensors=pp_proxy_tensors)
    else:
        raise ValueError(f"Invalid forward mode: {forward_batch.forward_mode}")

    return ret, can_run_graph
```

### Decode Forward (GPU计算核心)
```python
# model_runner.py:1865-1883
def forward_decode(
    self,
    forward_batch: ForwardBatch,
    skip_attn_backend_init: bool = False,
    pp_proxy_tensors=None,
) -> LogitsProcessorOutput:
    """
    Decode阶段的forward
    每次生成一个token, 复用KV cache
    """
    # 初始化attention backend metadata
    if not skip_attn_backend_init:
        self.attn_backend.init_forward_metadata(forward_batch)

    # 准备参数
    kwargs = {}
    if self.support_pp:
        kwargs["pp_proxy_tensors"] = pp_proxy_tensors

    # 调用模型forward
    return self.model.forward(
        forward_batch.input_ids,      # [batch_size]
        forward_batch.positions,      # [batch_size]
        forward_batch,
        **kwargs,
    )
```

### 模型Forward (DeepSeek-V3)
**文件**: `python/sglang/srt/models/deepseek*.py` (示例)

```python
class DeepseekV3ForCausalLM(nn.Module):
    """
    DeepSeek-V3 因果语言模型
    """

    def forward(
        self,
        input_ids: torch.Tensor,      # [batch_size] for decode
        positions: torch.Tensor,      # [batch_size]
        forward_batch: ForwardBatch,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> LogitsProcessorOutput:
        """
        模型forward pass

        计算流程:
        1. Embedding lookup
        2. 逐层Transformer
        3. LM head (输出logits)
        """
        # 1. Embedding
        hidden_states = self.embed_tokens(input_ids)  # [bs, hidden_dim]

        # 2. Transformer layers
        for i, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                positions,
                forward_batch,
                layer_id=i,
            )

        # 3. Final norm
        hidden_states = self.norm(hidden_states)

        # 4. LM head
        logits = self.lm_head(hidden_states)  # [bs, vocab_size]

        return LogitsProcessorOutput(
            next_token_logits=logits,
            next_token_logprobs=None,
            normalized_prompt_logprobs=None,
            input_token_logprobs=None,
            input_top_logprobs=None,
            output_top_logprobs=None,
        )

class DeepseekV3DecoderLayer(nn.Module):
    """
    DeepSeek-V3 Decoder Layer
    架构: MLA Attention + MoE FFN
    """

    def forward(
        self,
        hidden_states: torch.Tensor,  # [bs, hidden_dim]
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
    ) -> torch.Tensor:
        """
        单层forward
        """
        # 1. Self-Attention (MLA)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
            forward_batch=forward_batch,
            layer_id=layer_id,
        )

        hidden_states = residual + hidden_states

        # 2. MoE FFN
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states

        return hidden_states

class DeepseekV3Attention(nn.Module):
    """
    DeepSeek-V3 MLA Attention
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
    ) -> torch.Tensor:
        """
        MLA attention forward
        """
        # 1. QKV projection
        # MLA使用压缩的KV表示
        qkv = self.qkv_proj(hidden_states)  # [bs, qkv_dim]

        # 2. Split Q, K, V
        q, k, v = self._split_qkv(qkv)

        # 3. RoPE (Rotary Position Embedding)
        q, k = self.rotary_emb(q, k, positions)

        # 4. Attention computation
        # 调用TRT-LLM MLA backend
        attn_output = self.attn_backend.forward(
            q=q,
            k=k,
            v=v,
            layer_id=layer_id,
            forward_batch=forward_batch,
        )

        # 5. Output projection
        output = self.o_proj(attn_output)

        return output

class DeepseekV3MoE(nn.Module):
    """
    DeepSeek-V3 Mixture of Experts
    """

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        MoE forward

        流程:
        1. Router选择top-k experts
        2. 每个token分配到选中的experts
        3. Expert并行计算
        4. All-reduce聚合结果
        """
        batch_size, hidden_dim = hidden_states.shape

        # 1. Router
        router_logits = self.gate(hidden_states)  # [bs, num_experts]
        routing_weights, selected_experts = torch.topk(
            router_logits, self.top_k, dim=-1
        )  # [bs, top_k]

        routing_weights = F.softmax(routing_weights, dim=-1)

        # 2. Dispatch to experts
        final_hidden_states = torch.zeros_like(hidden_states)

        for expert_idx in range(self.num_experts):
            # 找到分配给这个expert的tokens
            expert_mask = (selected_experts == expert_idx)
            token_indices = expert_mask.nonzero(as_tuple=True)

            if len(token_indices[0]) == 0:
                continue

            # 3. Expert computation
            expert_input = hidden_states[token_indices[0]]
            expert_output = self.experts[expert_idx](expert_input)

            # 4. 加权累加
            weights = routing_weights[expert_mask]
            final_hidden_states[token_indices[0]] += expert_output * weights.unsqueeze(-1)

        # 5. All-reduce (如果使用EP)
        if self.ep_size > 1:
            torch.distributed.all_reduce(
                final_hidden_states,
                group=self.ep_group,
            )

        # 6. Shared expert (如果有)
        if self.shared_expert is not None:
            shared_output = self.shared_expert(hidden_states)
            final_hidden_states = final_hidden_states + shared_output

        return final_hidden_states
```

**GPU计算细节 (Decode阶段)**:

```
Input: [batch_size=8, hidden_dim=5120]
    ↓
┌─────────────────────────────────────────────────────┐
│ Layer 0                                              │
│                                                      │
│ ┌────────────────────────────────────────────┐      │
│ │ 1. MLA Attention                            │      │
│ │    - QKV projection [8, 5120] → [8, qkv_d] │      │
│ │      ↓ (TP=8: 每GPU 1/8 heads)              │      │
│ │    - RoPE                                   │      │
│ │    - TRT-LLM MLA Decode Kernel:             │      │
│ │      * 从KV cache读取 [8, max_len, kv_d]    │      │
│ │      * Fused QK^T + Softmax + @V            │      │
│ │      * 使用Tensor Cores加速                  │      │
│ │    - Output proj [8, attn_d] → [8, 5120]    │      │
│ └────────────────────────────────────────────┘      │
│    ↓ (Residual + LayerNorm)                         │
│                                                      │
│ ┌────────────────────────────────────────────┐      │
│ │ 2. MoE FFN                                  │      │
│ │    - Router [8, 5120] → [8, 256 experts]    │      │
│ │      ↓ Top-8 experts per token              │      │
│ │    - Expert dispatch                        │      │
│ │    - Expert 0-31: GPU 0                     │      │
│ │    - Expert 32-63: GPU 1                    │      │
│ │    - ... (EP=8)                             │      │
│ │    - Expert computation (parallel)          │      │
│ │    - All-reduce across EP group             │      │
│ │      ↓ NCCL通信                              │      │
│ │    - Shared expert (optional)               │      │
│ └────────────────────────────────────────────┘      │
│    ↓ (Residual + LayerNorm)                         │
└─────────────────────────────────────────────────────┘
    ↓
Layer 1-59 (重复)
    ↓
Final LayerNorm
    ↓
LM Head: [8, 5120] → [8, vocab_size=102400]
    ↓ (TP all-gather)
Output Logits: [8, 102400]
```

### Sampling
**文件**: `python/sglang/srt/layers/sampler.py`

```python
# sampler.py:100-300
class Sampler:
    """
    采样器: 从logits中采样下一个token
    """

    def __call__(
        self,
        logits: torch.Tensor,              # [batch_size, vocab_size]
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """
        采样下一个token

        支持的采样策略:
        - Greedy (argmax)
        - Top-p (nucleus sampling)
        - Top-k
        - Temperature
        - Beam search
        """
        batch_size, vocab_size = logits.shape
        next_token_ids = torch.empty(batch_size, dtype=torch.long, device=logits.device)

        for i, req in enumerate(forward_batch.reqs):
            # 获取采样参数
            sampling_params = req.sampling_params

            # 1. Temperature
            if sampling_params.temperature > 0:
                logits[i] = logits[i] / sampling_params.temperature

            # 2. Top-k
            if sampling_params.top_k > 0:
                logits[i] = self._apply_top_k(logits[i], sampling_params.top_k)

            # 3. Top-p
            if sampling_params.top_p < 1.0:
                logits[i] = self._apply_top_p(logits[i], sampling_params.top_p)

            # 4. 采样
            if sampling_params.temperature == 0:
                # Greedy
                next_token_ids[i] = torch.argmax(logits[i])
            else:
                # 概率采样
                probs = F.softmax(logits[i], dim=-1)
                next_token_ids[i] = torch.multinomial(probs, num_samples=1)

        return next_token_ids

    def _apply_top_k(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """
        Top-k filtering
        """
        values, indices = torch.topk(logits, top_k)
        min_value = values[-1]
        logits[logits < min_value] = -float("inf")
        return logits

    def _apply_top_p(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """
        Nucleus (top-p) filtering
        """
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # 找到累积概率超过top_p的位置
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False

        # 置零
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = -float("inf")

        return logits
```

### 发送到Detokenizer
**文件**: `python/sglang/srt/managers/scheduler.py`

```python
# scheduler.py:1900-2000
def send_to_detokenizer(self, finished_reqs: List[Req]):
    """
    发送完成的请求到detokenizer
    """
    for req in finished_reqs:
        output = BatchTokenIDsOutput(
            rid=req.rid,
            output_ids=req.output_ids,
            meta_info=req.meta_info,
            finished=req.finished,
        )

        self.send_to_detok.send_pyobj(output)
```

### Detokenizer处理
**文件**: `python/sglang/srt/managers/detokenizer_manager.py`

```python
# detokenizer_manager.py:200-400
class DetokenizerManager:
    """
    解码器管理器 (子进程)
    """

    def __init__(self, server_args: ServerArgs, port_args: PortArgs):
        # 加载分词器
        self.tokenizer = get_tokenizer(
            server_args.tokenizer_path or server_args.model_path,
            trust_remote_code=server_args.trust_remote_code,
        )

        # ZMQ sockets
        self.recv_from_scheduler = self._create_zmq_socket(
            port_args.detokenizer_ipc_name
        )
        self.send_to_tokenizer = self._create_zmq_socket(
            port_args.tokenizer_ipc_name
        )

    def event_loop(self):
        """
        事件循环
        """
        while True:
            # 1. 接收scheduler输出
            batch_output = self.recv_from_scheduler.recv_pyobj()

            # 2. Decode
            if batch_output.output_ids:
                output_text = self.tokenizer.decode(
                    batch_output.output_ids,
                    skip_special_tokens=True,
                )
            else:
                output_text = ""

            # 3. 发送回tokenizer
            result = GenerateReqOutput(
                rid=batch_output.rid,
                text=output_text,
                meta_info=batch_output.meta_info,
                finished=batch_output.finished,
            )

            self.send_to_tokenizer.send_pyobj(result)
```

### 返回HTTP响应
**文件**: `python/sglang/srt/managers/tokenizer_manager.py`

```python
# tokenizer_manager.py:500-600
async def _wait_for_response(self, rid: str) -> GenerateReqOutput:
    """
    等待最终响应
    """
    while True:
        # 接收detokenizer输出
        output = self.recv_from_detokenizer.recv_pyobj()

        if output.rid == rid:
            # 找到对应的响应
            if output.finished:
                # 请求完成
                return output

        # 继续等待
        await asyncio.sleep(0.001)

async def _wait_for_response_stream(self, rid: str):
    """
    流式响应
    """
    while True:
        output = self.recv_from_detokenizer.recv_pyobj()

        if output.rid == rid:
            yield output

            if output.finished:
                break

        await asyncio.sleep(0.001)
```

---

## 关键优化技术总结

### 1. Tensor Parallelism (TP=8)
**实现位置**: `python/sglang/srt/distributed/__init__.py`

**工作原理**:
- 将模型切分到8个GPU
- Attention heads切分: 每GPU负责1/8的heads
- MLP层切分: 每GPU负责1/8的hidden dimension
- 通信: All-reduce同步梯度 (推理时同步激活)

**性能提升**: 8x模型并行能力, 支持超大模型

---

### 2. TRT-LLM MLA Kernel
**实现位置**: `python/sglang/srt/layers/attention/trtllm_mla_backend.py`

**工作原理**:
- 使用FlashInfer库的TensorRT优化kernel
- MLA (Multi-head Latent Attention): 低秩分解压缩KV cache
- Fused kernel: QK^T, Softmax, @V在一个kernel中
- 利用Tensor Cores加速

**性能提升**: 2-3x attention计算速度, 50-70% KV cache内存节省

---

### 3. RadixAttention (自动前缀缓存)
**实现位置**: `python/sglang/srt/mem_cache/hiradix_cache.py`

**工作原理**:
- 使用Radix Tree自动检测和共享相同前缀
- 多个请求共享相同前缀的KV cache
- 显著减少内存使用

**性能提升**: 5-10x内存效率, 特别是批处理相似请求时

---

### 4. CUDA Graph
**实现位置**: `python/sglang/srt/model_executor/cuda_graph_runner.py`

**工作原理**:
- 预先record执行图
- Replay时直接执行, 无需重新调度
- 消除Python和CUDA driver开销

**性能提升**: 1.5-2x decode阶段速度

---

### 5. Torch Compile
**实现位置**: `python/sglang/srt/patch_torch.py`

**工作原理**:
- PyTorch 2.0 JIT编译
- Operator fusion: 融合多个小算子
- Kernel自动调优
- Memory layout优化

**性能提升**: 1.2-1.5x整体速度

---

### 6. Continuous Batching
**实现位置**: `python/sglang/srt/managers/schedule_policy.py`

**工作原理**:
- 动态batch: 不等待整个batch完成
- 请求完成后立即移出, 新请求立即加入
- 最大化GPU利用率

**性能提升**: 2-3x吞吐量

---

### 7. Paged Attention
**实现位置**: `python/sglang/srt/mem_cache/memory_pool.py`

**工作原理**:
- KV cache分页管理 (类似OS虚拟内存)
- 减少内存碎片
- 支持动态序列长度

**性能提升**: 提高内存利用率, 减少OOM

---

### 8. MoE Expert Parallelism
**实现位置**: `python/sglang/srt/models/deepseek*.py`

**工作原理**:
- Expert按GPU切分
- Router动态选择experts
- All-reduce聚合结果

**性能提升**: 支持256+ experts的超大MoE模型

---

## 完整时间线

### 服务器启动 (首次)
```
0s        python3 -m sglang.launch_server ...
          ↓ launch_server.py:20
1-3s      参数解析
          ↓ server_args.py:3055 prepare_server_args()
3-5s      多进程创建
          ↓ engine.py:380 _launch_subprocesses()
          ├─ Scheduler进程启动
          ├─ Detokenizer进程启动
          └─ TokenizerManager初始化
5-10s     分布式环境初始化
          ↓ model_runner.py:615 init_torch_distributed()
          ├─ NCCL初始化 (TP=8)
          └─ Model parallel groups创建
10-90s    模型加载
          ↓ model_runner.py:722 load_model()
          ├─ 模型架构识别
          ├─ 空模型实例化
          ├─ 权重加载 (TP切分)
          │  ├─ GPU 0: Expert 0-31
          │  ├─ GPU 1: Expert 32-63
          │  └─ ...
          └─ 模型移至GPU
90-120s   Attention backend初始化
          ↓ model_runner.py:1050 init_attention_backend()
          └─ TRTLLMMLABackend实例化
120-130s  KV cache分配
          ↓ model_runner.py:1100 init_memory_pool()
          ├─ MLATokenToKVPool创建 (~60GB)
          └─ RadixCache初始化
130-180s  CUDA graph capture
          ↓ cuda_graph_runner.py:100 capture_graphs()
          ├─ Capture bs=1,2,4,8,16
          └─ Torch compile (如果启用)
180s      HTTP服务器启动
          ✓ 服务器Ready!
```

### 单个推理请求流程
```
t=0ms     HTTP POST /generate
          ↓ http_server.py:750
t=1ms     FastAPI路由
          ↓ tokenizer_manager.py:200
t=2-5ms   Tokenization
          └─ input_ids = tokenizer.encode(prompt)
t=5ms     发送到Scheduler (ZMQ)
          ↓ scheduler.py:1500 event_loop_normal()
t=6ms     接收请求, 加入waiting_queue
          ↓ schedule_policy.py:200
t=7ms     调度: 创建ScheduleBatch
          ↓ scheduler.py:1200
t=8ms     准备ForwardBatch
          └─ input_ids, positions转tensor
t=10ms    Scheduler → TpModelWorker
          ↓ tp_worker.py:400
t=11ms    TpModelWorker → ModelRunner
          ↓ model_runner.py:1941 forward()

--- GPU计算开始 ---

t=12ms    判断是否使用CUDA graph
          ├─ Prefill: 不使用 (序列长度不定)
          └─ Decode: 使用graph replay

=== Prefill阶段 (首次, 假设256 tokens) ===

t=12-15ms Input Embedding
          └─ [256] → [256, 5120]

t=15-50ms Transformer Layers (60层)
          对于每层:
          ├─ MLA Attention (TRT-LLM kernel)
          │  ├─ QKV proj: 0.2ms
          │  ├─ RoPE: 0.1ms
          │  ├─ FlashAttention: 0.3ms
          │  └─ Output proj: 0.2ms
          ├─ MoE FFN
          │  ├─ Router: 0.05ms
          │  ├─ Expert dispatch: 0.1ms
          │  ├─ Expert compute (parallel): 0.2ms
          │  └─ All-reduce: 0.1ms
          └─ Total per layer: ~0.6ms
          Total: 60 * 0.6ms = 36ms

t=50-52ms LM Head
          └─ [256, 5120] → [256, 102400]

t=52ms    Sampling
          └─ next_token_id = sample(logits[-1])

=== Decode阶段 (自回归生成, 每个token) ===

t=53ms    CUDA Graph Replay (bs=1)
          ├─ Input: [1] (last token)
          ├─ Embedding: [1, 5120]
          ├─ 60 Layers (使用KV cache)
          │  └─ MLA Decode kernel: 非常快!
          └─ Output: [1, 102400]
          Total: ~2ms per token

t=55ms    Sampling
t=57ms    下一个token decode
...       (重复直到EOS或max_len)

--- GPU计算结束 ---

t=100ms   发送到Detokenizer
          ↓ detokenizer_manager.py:200
t=101ms   Detokenization
          └─ text = tokenizer.decode(output_ids)
t=102ms   发送回TokenizerManager
          ↓ tokenizer_manager.py:500
t=103ms   返回HTTP响应
          ✓ 完成!
```

### 性能指标 (估算, DeepSeek-V3 on 8xH100)
```
Prefill吞吐量: ~5000 tokens/s
Decode吞吐量: ~400 tokens/s (single batch)
端到端延迟 (256 in + 100 out): ~300ms
首token延迟 (TTFT): ~50ms
Token间延迟 (ITL): ~2-3ms (with CUDA graph)
峰值batch size: ~16-32 (取决于序列长度)
内存使用: ~70GB per GPU (model + KV cache)
```

---

## 参考文件索引

### 核心入口
1. `python/sglang/launch_server.py:20` - 主入口
2. `python/sglang/srt/server_args.py:3055` - 参数解析
3. `python/sglang/srt/entrypoints/http_server.py:1198` - HTTP服务器
4. `python/sglang/srt/entrypoints/engine.py:380` - 子进程启动

### 调度与批处理
5. `python/sglang/srt/managers/scheduler.py:500` - Scheduler类
6. `python/sglang/srt/managers/scheduler.py:1500` - 事件循环
7. `python/sglang/srt/managers/schedule_policy.py:200` - 调度策略
8. `python/sglang/srt/managers/schedule_batch.py` - Batch管理

### 模型执行
9. `python/sglang/srt/model_executor/model_runner.py:205` - ModelRunner类
10. `python/sglang/srt/model_executor/model_runner.py:615` - 分布式初始化
11. `python/sglang/srt/model_executor/model_runner.py:722` - 模型加载
12. `python/sglang/srt/model_executor/model_runner.py:1941` - Forward函数

### 模型加载
13. `python/sglang/srt/model_loader/__init__.py:21` - get_model
14. `python/sglang/srt/model_loader/loader.py` - ModelLoader
15. `python/sglang/srt/model_loader/utils.py:50` - 架构识别

### Attention Backend
16. `python/sglang/srt/layers/attention/attention_registry.py` - Backend注册表
17. `python/sglang/srt/layers/attention/trtllm_mla_backend.py:70` - TRT-LLM MLA
18. `python/sglang/srt/layers/attention/base_attn_backend.py` - Base Backend

### 内存管理
19. `python/sglang/srt/mem_cache/memory_pool.py:200` - KV Pool
20. `python/sglang/srt/mem_cache/hiradix_cache.py:100` - RadixCache
21. `python/sglang/srt/mem_cache/allocator.py` - Allocator

### CUDA Graph & Compile
22. `python/sglang/srt/model_executor/cuda_graph_runner.py:100` - CUDA Graph
23. `python/sglang/srt/patch_torch.py:50` - Torch Compile

### 分布式
24. `python/sglang/srt/distributed/__init__.py:100` - 分布式初始化
25. `python/sglang/srt/distributed/parallel_state.py` - 并行状态

### Token管理
26. `python/sglang/srt/managers/tokenizer_manager.py:200` - TokenizerManager
27. `python/sglang/srt/managers/detokenizer_manager.py:200` - DetokenizerManager

### 采样
28. `python/sglang/srt/layers/sampler.py:100` - Sampler

---

## 总结

这份文档详细追踪了SGLang DeepSeek-V3服务器从启动到推理的完整流程，包括:

1. **命令行到HTTP服务器**: 参数解析 → 多进程架构 → FastAPI启动
2. **分布式初始化**: NCCL → TP groups → 通信建立
3. **模型加载**: 架构识别 → 权重加载 → TP切分
4. **优化组件**: TRT-LLM MLA → RadixCache → CUDA Graph → Torch Compile
5. **推理流程**: HTTP请求 → Tokenizer → Scheduler → GPU计算 → Sampling → Detokenizer → 响应

每个步骤都标注了对应的**文件路径**和**函数名称**，方便深入源码学习。

**关键性能优化**使SGLang能够高效服务超大规模MoE模型(DeepSeek-V3 671B参数)，达到业界领先的推理速度。
