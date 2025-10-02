# SGLang 张量并行(TP)与专家并行(EP)详细指南

## 目录
1. [参数定义与含义](#参数定义与含义)
2. [Tensor Parallelism (TP) - 张量并行](#tensor-parallelism-tp---张量并行)
3. [Expert Parallelism (EP) - 专家并行](#expert-parallelism-ep---专家并行)
4. [TP vs EP 对比与选择](#tp-vs-ep-对比与选择)
5. [使用建议](#使用建议)
6. [代码实现细节](#代码实现细节)

---

## 参数定义与含义

### --tp-size / --tensor-parallel-size

**文件位置**: `python/sglang/srt/server_args.py:1517-1522`

```python
parser.add_argument(
    "--tensor-parallel-size",
    "--tp-size",
    type=int,
    default=ServerArgs.tp_size,  # 默认值为 1
    help="The tensor parallelism size.",
)
```

**含义**:
- **张量并行度** - 将模型的权重矩阵按列或行切分到多个GPU上
- 每个GPU只存储模型权重的一部分，但处理相同的输入数据
- 通过NCCL通信(AllReduce/AllGather)在GPU间同步中间结果

**默认值**: `1` (不使用TP)

---

### --ep-size / --expert-parallel-size

**文件位置**: `python/sglang/srt/server_args.py:2078-2083`

```python
parser.add_argument(
    "--ep-size",
    "--ep",
    type=int,
    default=ServerArgs.ep_size,  # 默认值为 1
    help="The expert parallelism size.",
)
```

**含义**:
- **专家并行度** - 专门用于MoE(Mixture of Experts)模型
- 将不同的专家(experts)分配到不同的GPU上
- 通过All-to-All通信在GPU间分发和收集tokens

**默认值**: `1` (不使用EP)

**重要约束** (`python/sglang/srt/server_args.py:911-914`):
```python
assert self.ep_size in [
    1,
    self.tp_size,
], "The expert parallel size must be 1 or the same as the tensor parallel size"
```

---

## Tensor Parallelism (TP) - 张量并行

### 1. 工作原理

TP将模型的线性层权重矩阵切分到多个GPU上:

- **ColumnParallelLinear**: 按列切分权重矩阵 `A = [A_1, A_2, ..., A_p]`
- **RowParallelLinear**: 按行切分权重矩阵

**图示** (以TP=4为例):

```
原始权重矩阵 (8192 x 28672):
┌──────────────────────────┐
│    完整权重矩阵 W         │
└──────────────────────────┘

ColumnParallelLinear切分 (按列):
┌──────┬──────┬──────┬──────┐
│ W_1  │ W_2  │ W_3  │ W_4  │  每个GPU: 8192 x 7168
│GPU 0 │GPU 1 │GPU 2 │GPU 3 │
└──────┴──────┴──────┴──────┘

RowParallelLinear切分 (按行):
┌─────────────────────────┐
│  W_1  (GPU 0)           │  每个GPU: 7168 x 8192
├─────────────────────────┤
│  W_2  (GPU 1)           │
├─────────────────────────┤
│  W_3  (GPU 2)           │
├─────────────────────────┤
│  W_4  (GPU 3)           │
└─────────────────────────┘
```

### 2. 初始化流程

**文件**: `python/sglang/srt/model_executor/model_runner.py:620-720`

#### Step 1: 设置分布式环境

```python
def init_torch_distributed(self):
    # 设置后端 (NCCL for CUDA)
    if self.device == "cuda":
        backend = "nccl"

    # 初始化分布式环境
    init_distributed_environment(
        backend=backend,
        world_size=self.tp_size * self.pp_size,
        rank=self.tp_size * self.pp_rank + self.tp_rank,
        local_rank=self.gpu_id,
        distributed_init_method=dist_init_method,
        timeout=self.server_args.dist_timeout,
    )
```

#### Step 2: 创建TP进程组

**文件**: `python/sglang/srt/distributed/parallel_state.py:1381-1459`

```python
def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
):
    # 计算TP组数量
    num_tensor_model_parallel_groups = world_size // tensor_model_parallel_size

    # 创建TP进程组
    # 例如: world_size=8, tp_size=4
    # Group 0: [0, 1, 2, 3]
    # Group 1: [4, 5, 6, 7]
    group_ranks = []
    for i in range(num_tensor_model_parallel_groups):
        ranks = list(
            range(i * tensor_model_parallel_size,
                  (i + 1) * tensor_model_parallel_size)
        )
        group_ranks.append(ranks)

    _TP = init_model_parallel_group(
        group_ranks,
        get_world_group().local_rank,
        backend,
        group_name="tp",
    )
```

**进程组拓扑** (world_size=8, tp_size=4):
```
TP Group 0: GPU 0 ←→ GPU 1 ←→ GPU 2 ←→ GPU 3
TP Group 1: GPU 4 ←→ GPU 5 ←→ GPU 6 ←→ GPU 7
```

### 3. 权重切分

**文件**: `python/sglang/srt/layers/linear.py:257-267`

#### ColumnParallelLinear 示例

```python
class ColumnParallelLinear(LinearBase):
    """
    将权重矩阵 A 按列切分: A = [A_1, ..., A_p]

    计算: Y = XA + b
    - X: 输入 (batch_size, input_size)
    - A: 权重 (input_size, output_size) -> 切分为 (input_size, output_size/tp_size)
    - Y: 输出 (batch_size, output_size/tp_size)

    如果 gather_output=True, 会在所有GPU间做AllGather收集完整输出
    """

    def __init__(self, input_size, output_size, ...):
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

        # 每个GPU只分配 output_size / tp_size 的权重
        self.output_size_per_partition = divide(output_size, self.tp_size)
```

**实际切分示例** (DeepSeek-V3 的 MLP层):

```python
# 原始: hidden_size=7168, intermediate_size=18432
# TP=4时每个GPU:

# gate_up_proj (ColumnParallel):
# - 输入: 7168
# - 输出: 18432 * 2 / 4 = 9216  (每个GPU)
self.gate_up_proj = ColumnParallelLinear(
    hidden_size=7168,
    output_size=18432 * 2,  # gate + up
    tp_size=4,
)

# down_proj (RowParallel):
# - 输入: 18432 / 4 = 4608  (每个GPU)
# - 输出: 7168 (需要AllReduce)
self.down_proj = RowParallelLinear(
    input_size=18432,
    output_size=7168,
    tp_size=4,
)
```

### 4. 通信模式

#### AllReduce (RowParallelLinear)

**文件**: `python/sglang/srt/distributed/communication_op.py`

```python
# GPU间同步求和
def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """
    所有GPU的输出求和

    Example (tp_size=4):
    GPU 0: [1, 2, 3]
    GPU 1: [4, 5, 6]      AllReduce      所有GPU: [10, 14, 18]
    GPU 2: [2, 3, 4]   ============>
    GPU 3: [3, 4, 5]
    """
    return torch.distributed.all_reduce(input_, group=get_tp_group())
```

#### AllGather (ColumnParallelLinear with gather_output=True)

```python
def tensor_model_parallel_all_gather(input_: torch.Tensor) -> torch.Tensor:
    """
    收集所有GPU的输出拼接

    Example (tp_size=4):
    GPU 0: [1, 2]
    GPU 1: [3, 4]        AllGather      所有GPU: [1, 2, 3, 4, 5, 6, 7, 8]
    GPU 2: [5, 6]     ============>
    GPU 3: [7, 8]
    """
    return torch.distributed.all_gather(input_, group=get_tp_group())
```

### 5. 内存使用

**单GPU显存计算** (DeepSeek-V3 为例):

```
总参数量: 671B
精度: BF16 (2 bytes/param)

不使用TP (tp_size=1):
- 权重: 671B * 2 bytes = 1342 GB
- KV Cache: ~200 GB (取决于batch size)
- 激活值: ~50 GB
总计: ~1592 GB  ❌ 单GPU放不下!

使用TP=8:
- 权重: 1342 / 8 = 167.75 GB
- KV Cache: 200 / 8 = 25 GB
- 激活值: 50 GB (每个GPU都需要)
- 通信开销: ~10 GB
总计: ~252.75 GB  ✅ H100 (80GB) 可以运行!
```

**内存检查** (`python/sglang/srt/model_executor/model_runner.py:702-716`):

```python
# 检查TP组内GPU内存是否均衡
if self.tp_size > 1:
    if min_per_gpu_memory < local_gpu_memory * 0.9:
        raise ValueError(
            "The memory capacity is unbalanced. Some GPUs may be occupied."
        )
```

---

## Expert Parallelism (EP) - 专家并行

### 1. 工作原理

EP专门用于MoE(Mixture of Experts)模型，如DeepSeek-V2/V3:

```
MoE层结构:
┌─────────────────────────────────┐
│  Router (gate)                  │  选择top-k个专家
└────────────┬────────────────────┘
             │
     ┌───────┴────────┐
     │   top_k=6      │
     │  从256个专家   │
     │  中选择6个     │
     └───────┬────────┘
             │
    ┌────────┴─────────┐
    │  Experts (256个)  │
    │  每个专家都是MLP  │
    └──────────────────┘
```

**问题**: 256个专家，每个专家的权重都很大，单GPU放不下所有专家

**解决方案**: 使用EP将不同专家分配到不同GPU

### 2. EP进程组创建

**文件**: `python/sglang/srt/distributed/parallel_state.py:1461-1504`

```python
def initialize_model_parallel(...):
    moe_ep_size = expert_model_parallel_size  # EP size
    moe_tp_size = tensor_model_parallel_size // moe_ep_size  # 剩余的TP size

    # 创建EP进程组
    # Example: world_size=8, tp_size=8, ep_size=8
    # 此时 moe_tp_size = 1

    if moe_ep_size == tensor_model_parallel_size:
        # EP复用TP的进程组
        _MOE_EP = _TP
    else:
        # 创建新的EP进程组
        # 假设 tp_size=8, ep_size=4, moe_tp_size=2
        # EP Group 0: [0, 2, 4, 6]  (每隔moe_tp_size取一个)
        # EP Group 1: [1, 3, 5, 7]
        group_ranks = []
        for i in range(num_tensor_model_parallel_groups):
            for j in range(moe_tp_size):
                st = i * tensor_model_parallel_size + j
                en = (i + 1) * tensor_model_parallel_size + j
                ranks = list(range(st, en, moe_tp_size))
                group_ranks.append(ranks)
```

**进程组拓扑示例** (tp_size=8, ep_size=8):

```
所有8个GPU组成一个EP组:
EP Group: [GPU 0, GPU 1, GPU 2, GPU 3, GPU 4, GPU 5, GPU 6, GPU 7]

专家分配 (假设256个专家):
GPU 0: Experts 0-31    (32个)
GPU 1: Experts 32-63   (32个)
GPU 2: Experts 64-95   (32个)
GPU 3: Experts 96-127  (32个)
GPU 4: Experts 128-159 (32个)
GPU 5: Experts 160-191 (32个)
GPU 6: Experts 192-223 (32个)
GPU 7: Experts 224-255 (32个)
```

### 3. DeepEP通信机制

**文件**: `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`

EP使用**All-to-All**通信模式，由DeepSeek开发的DeepEP库实现高效通信。

#### 两阶段通信

**Dispatch阶段** (分发tokens到对应的专家GPU):

```python
def dispatch_a(self, hidden_states, topk_idx, topk_weights):
    """
    Phase A: 准备通信布局

    Input:
    - hidden_states: (num_tokens, hidden_size)  # 所有tokens
    - topk_idx: (num_tokens, top_k)  # 每个token选择的专家ID

    计算:
    - 哪些tokens需要发送到哪个GPU
    - 每个GPU会接收多少tokens
    """
    buffer.get_dispatch_layout(topk_idx, num_experts)
```

```python
def dispatch_b(self):
    """
    Phase B: All-to-All通信

    将tokens分发到拥有对应专家的GPU

    Example (4 GPUs, 每个GPU 2个专家):
    GPU 0 (Experts 0-1):
      Token 1 选择 Expert 0 → 留在GPU 0
      Token 2 选择 Expert 5 → 发送到GPU 2
      Token 3 选择 Expert 1 → 留在GPU 0

    GPU 1 (Experts 2-3):
      Token 4 选择 Expert 2 → 留在GPU 1
      ...

    All-to-All后，每个GPU收到所有选择其专家的tokens
    """
    recv_hidden_states, recv_topk_idx, recv_topk_weights = \
        buffer.dispatch(hidden_states, topk_idx, topk_weights, ...)
```

**Combine阶段** (收集专家输出):

```python
def combine_a(self, hidden_states, topk_idx, topk_weights):
    """
    Phase A: 准备反向All-to-All

    每个GPU的专家处理完tokens后，需要将结果发送回原始GPU
    """
```

```python
def combine_b(self):
    """
    Phase B: All-to-All通信

    将专家输出发送回原始token所在的GPU

    最终每个GPU收到其原始tokens的所有专家输出
    """
    combined_output = buffer.combine(...)
    return combined_output
```

**完整流程图**:

```
原始GPU分布:
GPU 0: [Token 0, Token 1, Token 2, Token 3] → Router选择专家
       Token 0 → Expert 5 (GPU 1)
       Token 1 → Expert 2 (GPU 0)
       Token 2 → Expert 7 (GPU 1)
       Token 3 → Expert 1 (GPU 0)

GPU 1: [Token 4, Token 5, Token 6, Token 7]
       Token 4 → Expert 3 (GPU 0)
       ...

┌─────────────────────────────────────┐
│   Dispatch All-to-All              │
└─────────────┬───────────────────────┘
              │
专家GPU分布:
GPU 0 (Experts 0-3): [Token 1, Token 3, Token 4, ...] → 执行Expert计算
GPU 1 (Experts 4-7): [Token 0, Token 2, Token 6, ...]

┌─────────────────────────────────────┐
│   Combine All-to-All               │
└─────────────┬───────────────────────┘
              │
回到原始GPU:
GPU 0: [Token 0结果, Token 1结果, Token 2结果, Token 3结果]
GPU 1: [Token 4结果, Token 5结果, Token 6结果, Token 7结果]
```

### 4. MoE层实现

**文件**: `python/sglang/srt/models/deepseek_v2.py:475-949`

```python
class DeepseekV2MoE(nn.Module):
    def __init__(self, config, ...):
        if get_moe_a2a_backend().is_deepep():
            self.ep_size = get_moe_expert_parallel_world_size()
            self.num_experts = (
                config.n_routed_experts  # 256
                + ep_num_redundant_experts  # 额外的冗余专家
            )

            # 创建DeepEP dispatcher
            self.experts = DeepEPMoE(
                num_experts=self.num_experts,
                top_k=config.num_experts_per_tok,  # 6
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size,
                ...
            )

    def forward(self, hidden_states):
        # Step 1: Router选择专家
        topk_idx, topk_weights = self.gate(hidden_states)

        # Step 2: Dispatch tokens
        if self.ep_size > 1:
            self.experts.deepep_dispatcher.dispatch_a(
                hidden_states, topk_idx, topk_weights
            )
            dispatch_output = self.experts.deepep_dispatcher.dispatch_b()

        # Step 3: 专家计算
        expert_output = self.experts.moe_impl(dispatch_output)

        # Step 4: Combine结果
        if self.ep_size > 1:
            self.experts.deepep_dispatcher.combine_a(
                expert_output, topk_idx, topk_weights
            )
            final_output = self.experts.deepep_dispatcher.combine_b()

        return final_output
```

### 5. 内存使用

**DeepSeek-V3 MoE内存计算**:

```
配置:
- 专家数量: 256
- 每个专家参数: ~2.6B (intermediate_size=2048)
- 精度: BF16
- Top-K: 6

不使用EP (ep_size=1):
- 所有专家权重: 256 * 2.6B * 2 bytes = 1331 GB  ❌ 放不下!

使用EP=8:
- 每个GPU的专家: 256 / 8 = 32个
- 每个GPU权重: 32 * 2.6B * 2 bytes = 166.4 GB  ✅ 可行!
- All-to-All通信缓冲: ~20 GB
总计: ~186.4 GB per GPU
```

---

## TP vs EP 对比与选择

### 对比表

| 特性 | Tensor Parallelism (TP) | Expert Parallelism (EP) |
|------|------------------------|------------------------|
| **适用模型** | 所有Transformer模型 | 仅MoE模型 |
| **切分对象** | 权重矩阵 (QKV, MLP等) | 专家(Experts) |
| **通信模式** | AllReduce, AllGather | All-to-All |
| **通信频率** | 每层都需要通信 | 仅在MoE层通信 |
| **显存节省** | 线性: 1/tp_size | 线性: 1/ep_size (仅专家部分) |
| **计算负载** | 均衡分布 | 可能不均衡 (取决于Router) |
| **通信开销** | 高 (每层都有) | 中等 (仅MoE层) |
| **实现复杂度** | 简单 | 复杂 (需要DeepEP) |

### 何时使用TP

✅ **推荐使用TP的场景**:

1. **模型太大放不下单GPU**
   ```bash
   # 例如: Llama-70B, Qwen-72B
   python -m sglang.launch_server \
       --model meta-llama/Llama-2-70b \
       --tp 4  # 4个GPU分担模型权重
   ```

2. **需要更高的吞吐量**
   - TP可以增加计算并行度
   - 适合大batch size推理

3. **模型没有MoE结构**
   - 标准Transformer模型只能用TP

4. **GPU间带宽充足**
   - NVLink, NVSwitch等高带宽互连
   - TP通信频繁，需要高带宽

❌ **不推荐TP的场景**:

1. **小模型** (如7B以下)
   - 单GPU可以放下，TP反而增加通信开销

2. **GPU间带宽低**
   - PCIe连接的多GPU服务器
   - 通信成为瓶颈

### 何时使用EP

✅ **推荐使用EP的场景**:

1. **MoE模型且专家数量多**
   ```bash
   # DeepSeek-V3: 256个专家
   python -m sglang.launch_server \
       --model deepseek-ai/DeepSeek-V3 \
       --tp 8 \
       --ep 8  # EP=TP, 每个GPU 32个专家
   ```

2. **专家权重太大**
   - 单GPU无法容纳所有专家
   - EP按专家数量切分

3. **配合TP使用**
   - 通常 `ep_size = tp_size`
   - 在TP的基础上，MoE层额外使用EP

❌ **不推荐EP的场景**:

1. **非MoE模型**
   - EP仅对MoE有效

2. **专家数量少**
   - 如只有8个专家，单GPU可以放下

3. **ep_size != tp_size**
   - 当前实现要求 `ep_size ∈ {1, tp_size}`

### 组合使用建议

#### 场景1: DeepSeek-V3 (671B, 256专家)

```bash
# 推荐配置: 8xH100 (80GB each)
python -m sglang.launch_server \
    --model deepseek-ai/DeepSeek-V3 \
    --tp 8 \           # 8-way张量并行
    --ep 8 \           # 8-way专家并行
    --attention-backend trtllm_mla \
    --enable-torch-compile
```

**内存分析**:
```
每个GPU:
- 非MoE层权重 (TP切分): ~100 GB / 8 = 12.5 GB
- MoE层专家权重 (EP切分): ~1200 GB / 8 = 150 GB
- KV Cache: ~25 GB
- 激活值: ~40 GB
- 通信缓冲: ~20 GB
总计: ~247.5 GB

需要至少: 256 GB GPU (如H100 80GB需要更多优化)
```

#### 场景2: Qwen2-72B (非MoE)

```bash
# 推荐配置: 4xA100 (80GB each)
python -m sglang.launch_server \
    --model Qwen/Qwen2-72B \
    --tp 4  # 仅使用TP
```

**内存分析**:
```
每个GPU:
- 模型权重: 144 GB / 4 = 36 GB
- KV Cache: ~15 GB
- 激活值: ~10 GB
总计: ~61 GB  ✅ A100 80GB可用
```

#### 场景3: Mixtral-8x7B (MoE, 8专家)

```bash
# 推荐配置: 2xA100 (80GB each)
python -m sglang.launch_server \
    --model mistralai/Mixtral-8x7B-v0.1 \
    --tp 2
    # 不需要EP，8个专家单GPU可以放下
```

**内存分析**:
```
每个GPU:
- 非MoE层: ~15 GB / 2 = 7.5 GB
- 8个专家: ~75 GB (不切分)
- KV Cache: ~10 GB
总计: ~92.5 GB

如果显存不足，可以使用 --ep 2:
- 8个专家 / 2 = 4个专家/GPU: ~37.5 GB
总计: ~55 GB  ✅ 更节省
```

---

## 使用建议

### 1. 选择合适的TP/EP大小

**原则**:
1. **最小化通信**: TP/EP越大，通信开销越大
2. **满足显存**: 确保单GPU显存足够
3. **GPU数量**: tp_size * pp_size 必须整除GPU总数

**决策树**:

```
模型能放入单GPU?
├─ 是 → 不使用TP (tp=1)
└─ 否 →
    ├─ 是MoE模型?
    │   ├─ 是 → 使用TP+EP (ep=tp)
    │   └─ 否 → 仅使用TP
    └─ 选择最小的tp_size使得:
        model_memory / tp_size < gpu_memory * 0.7
```

### 2. 性能优化建议

**配置示例**:

```bash
# 8xH100服务器，运行DeepSeek-V3
python -m sglang.launch_server \
    --model deepseek-ai/DeepSeek-V3 \
    --tp 8 \
    --ep 8 \
    --attention-backend trtllm_mla \  # 使用MLA优化的attention
    --enable-torch-compile \          # 启用JIT编译
    --torch-compile-max-bs 16 \       # 编译batch size <= 16的kernel
    --cuda-graph-max-bs 256 \         # CUDA Graph最大batch
    --disable-radix-cache false \     # 启用prefix caching
    --chunked-prefill-size 8192       # Chunked prefill优化
```

**参数说明**:

| 参数 | 作用 | 推荐值 |
|------|------|--------|
| `--tp` | 张量并行度 | 根据GPU数量和模型大小 |
| `--ep` | 专家并行度 | MoE模型: 设为tp_size<br>非MoE: 1 |
| `--cuda-graph-max-bs` | CUDA Graph最大batch | TP<4: 256<br>TP>=4: 512 |
| `--chunked-prefill-size` | Prefill分块大小 | 4096-8192 |

### 3. 常见问题排查

#### 问题1: OOM (Out of Memory)

```
错误: torch.cuda.OutOfMemoryError: CUDA out of memory
```

**解决方案**:
1. **增加TP/EP大小**
   ```bash
   --tp 8  # 从4增加到8
   ```

2. **减小CUDA Graph大小**
   ```bash
   --cuda-graph-max-bs 128  # 从256减小
   ```

3. **减小KV Cache**
   ```bash
   --max-running-requests 256  # 限制并发请求
   ```

#### 问题2: 通信超时

```
错误: RuntimeError: NCCL timeout
```

**解决方案**:
1. **增加超时时间**
   ```bash
   --dist-timeout 3600  # 1小时
   ```

2. **检查网络**
   ```bash
   # 测试GPU间带宽
   nvidia-smi nvlink --status
   ```

3. **禁用自定义AllReduce**
   ```bash
   --disable-custom-all-reduce  # 使用NCCL标准实现
   ```

#### 问题3: EP约束错误

```
错误: AssertionError: The expert parallel size must be 1 or the same as the tensor parallel size
```

**解决方案**:
```bash
# 错误配置:
--tp 8 --ep 4  ❌

# 正确配置:
--tp 8 --ep 8  ✅
# 或
--tp 8 --ep 1  ✅
```

---

## 代码实现细节

### 1. TP组初始化调用链

```
python/sglang/srt/model_executor/model_runner.py:281
└─ init_torch_distributed()
   └─ python/sglang/srt/distributed/parallel_state.py:673
      └─ init_distributed_environment(world_size=tp*pp, rank=...)
         └─ torch.distributed.init_process_group(backend="nccl")

      └─ python/sglang/srt/distributed/parallel_state.py:681
         └─ initialize_model_parallel(tensor_model_parallel_size=tp_size)
            └─ 创建TP进程组 (lines 1422-1459)
            └─ 创建EP进程组 (lines 1461-1504)
            └─ 创建PP进程组 (lines 1506-1522)
```

### 2. ColumnParallelLinear前向传播

**文件**: `python/sglang/srt/layers/linear.py:257-600`

```python
class ColumnParallelLinear(LinearBase):
    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        # Step 1: 本地矩阵乘法 (只用本GPU的权重切片)
        # input: (batch, seq, input_size)
        # weight: (input_size, output_size/tp_size)
        # output: (batch, seq, output_size/tp_size)
        output_parallel = F.linear(input_, self.weight)

        if self.bias is not None:
            output_parallel = output_parallel + self.bias

        # Step 2: 如果需要gather完整输出
        if self.gather_output:
            # AllGather: 从所有GPU收集结果并拼接
            # output: (batch, seq, output_size)
            output = tensor_model_parallel_all_gather(output_parallel)
        else:
            output = output_parallel

        return output
```

### 3. RowParallelLinear前向传播

```python
class RowParallelLinear(LinearBase):
    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        # Step 1: 如果输入需要先切分
        if self.input_is_parallel:
            input_parallel = input_
        else:
            # Split输入到每个GPU
            # input: (batch, seq, input_size)
            # input_parallel: (batch, seq, input_size/tp_size)
            tp_rank = get_tensor_model_parallel_rank()
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.tp_size
            )
            input_parallel = splitted_input[tp_rank]

        # Step 2: 本地矩阵乘法
        # weight: (input_size/tp_size, output_size)
        # output: (batch, seq, output_size)
        output_parallel = F.linear(input_parallel, self.weight)

        # Step 3: AllReduce求和所有GPU的输出
        if self.reduce_results:
            output = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output = output_parallel

        return output
```

### 4. MoE EP调度

**文件**: `python/sglang/srt/models/deepseek_v2.py:895-949`

```python
class DeepseekV2MoE(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 创建状态对象
        state = MoEState(
            hidden_states_mlp_input=hidden_states,
            forward_batch=forward_batch,
        )

        # Step 1: Router - 选择专家
        self.op_gate(state)
        # state.topk_idx: (num_tokens, top_k)  每个token选哪些专家
        # state.topk_weights: (num_tokens, top_k)  每个专家的权重

        # Step 2: Dispatch Phase A - 准备通信布局
        if self.ep_size > 1:
            self.experts.deepep_dispatcher.dispatch_a(
                hidden_states=state.hidden_states_mlp_input,
                topk_idx=state.topk_idx_local,
                topk_weights=state.topk_weights_local,
                forward_batch=state.forward_batch,
            )

        # Step 3: Dispatch Phase B - All-to-All发送tokens
        if self.ep_size > 1:
            state.dispatch_output = self.experts.deepep_dispatcher.dispatch_b()
            # dispatch_output包含:
            # - hidden_states: 本GPU专家需要处理的所有tokens
            # - topk_idx, topk_weights: 对应的专家选择信息

        # Step 4: 专家计算
        state.hidden_states_experts_output = self.experts.moe_impl(
            dispatch_output=state.dispatch_output,
        )
        # 每个GPU的专家处理分配给它的tokens

        # Step 5: Combine Phase A - 准备反向通信
        if self.ep_size > 1:
            self.experts.deepep_dispatcher.combine_a(
                hidden_states=state.hidden_states_experts_output,
                topk_idx=state.dispatch_output.topk_idx,
                topk_weights=state.dispatch_output.topk_weights,
            )

        # Step 6: Combine Phase B - All-to-All收集结果
        if self.ep_size > 1:
            state.hidden_states_after_combine = \
                self.experts.deepep_dispatcher.combine_b()
            # 每个GPU收回其原始tokens的专家输出

        # Step 7: 合并共享专家输出 (如果有)
        if shared_output is not None:
            final_output = shared_output + \
                state.hidden_states_after_combine * self.routed_scaling_factor
        else:
            final_output = state.hidden_states_after_combine * \
                self.routed_scaling_factor

        return final_output
```

### 5. 关键文件索引

| 功能 | 文件路径 | 关键行号 |
|------|---------|---------|
| **参数定义** | | |
| --tp-size定义 | `python/sglang/srt/server_args.py` | 192, 1517-1522 |
| --ep-size定义 | `python/sglang/srt/server_args.py` | 299, 2078-2083 |
| EP约束检查 | `python/sglang/srt/server_args.py` | 911-914 |
| **TP初始化** | | |
| ModelRunner初始化 | `python/sglang/srt/model_executor/model_runner.py` | 208-291 |
| 分布式初始化 | `python/sglang/srt/model_executor/model_runner.py` | 620-720 |
| TP进程组创建 | `python/sglang/srt/distributed/parallel_state.py` | 1381-1459 |
| **EP初始化** | | |
| EP进程组创建 | `python/sglang/srt/distributed/parallel_state.py` | 1461-1504 |
| **TP实现** | | |
| ColumnParallelLinear | `python/sglang/srt/layers/linear.py` | 257-600 |
| RowParallelLinear | `python/sglang/srt/layers/linear.py` | 1187-1400 |
| AllReduce实现 | `python/sglang/srt/distributed/communication_op.py` | - |
| **EP实现** | | |
| DeepseekV2MoE | `python/sglang/srt/models/deepseek_v2.py` | 475-949 |
| DeepEPDispatcher | `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` | 251-500 |
| DeepEPMoE | `python/sglang/srt/layers/moe/ep_moe/layer.py` | 83-300 |

---

## 总结

### 快速决策指南

```
┌─────────────────────────────────────┐
│  选择TP还是EP?                      │
└────────────┬────────────────────────┘
             │
    ┌────────┴────────┐
    │  是MoE模型?     │
    └────────┬────────┘
             │
        ┌────┴────┐
        │   是    │   否
        │         │
        ↓         ↓
  ┌─────────┐  ┌──────────┐
  │TP + EP  │  │  仅TP    │
  │ep=tp    │  │  ep=1    │
  └─────────┘  └──────────┘

  TP大小选择:
  tp_size = min(
      gpu_count,
      ceil(model_memory / (gpu_memory * 0.7))
  )
```

### 最佳实践清单

- [ ] 计算模型显存需求
- [ ] 选择合适的tp_size
- [ ] MoE模型设置ep_size = tp_size
- [ ] 配置合适的cuda-graph-max-bs
- [ ] 启用torch compile (适合固定batch)
- [ ] 测试GPU间通信带宽
- [ ] 监控各GPU显存使用均衡性
- [ ] 使用混合精度(FP8/FP16)节省显存

### 性能调优检查表

| 优化项 | 检查点 | 推荐配置 |
|--------|--------|----------|
| **通信** | NVLink/NVSwitch可用? | 高带宽时增大TP |
| **显存** | 是否OOM? | 增大TP/EP或减小batch |
| **计算** | GPU利用率低? | 减小TP降低通信开销 |
| **延迟** | First token延迟高? | 启用chunked prefill |
| **吞吐** | 吞吐量不足? | 增大batch size |

---

**参考文档生成时间**: 2025-10
**SGLang版本**: main branch (latest)
**覆盖模型**: DeepSeek-V3, Qwen2, Llama, Mixtral等
