# SGLang `--tp` 与 `--ep` 并行策略详解与选型建议

> 适用版本：当前仓库。本文结合源码位置，说明 `--tp`（Tensor Parallel）与 `--ep`（Expert Parallel, MoE 专家并行）在 SGLang 中的含义、初始化流程、执行影响与使用建议。

---

## 一、参数定义与公共入口

- `--tp <N>`：张量并行大小（每层内权重/激活在 N 张 GPU 间切分，层内需 NCCL 通信）。
  - ServerArgs 字段：`tp_size`（python/sglang/srt/server_args.py:173）
- `--ep <M>`：MoE 专家并行大小（同一层的专家按 M 份跨 GPU 切分，token 经路由后跨卡发送到对应专家）。
  - ServerArgs 字段：`ep_size`（python/sglang/srt/server_args.py:299）

公共初始化流程：
- 进程拓扑与 rank 计算（启动子进程时）：
  - 计算 `moe_ep_rank = tp_rank // (tp_size // ep_size)`（python/sglang/srt/entrypoints/engine.py:803）
  - 每个调度子进程带上 `tp_rank`、`moe_ep_rank`（python/sglang/srt/managers/scheduler.py:233）
- 分布式组创建（模型构建前）：
  - `initialize_model_parallel(tensor_model_parallel_size=tp_size, expert_model_parallel_size=ep_size, ...)`（python/sglang/srt/model_executor/model_runner.py:741）

---

## 二、模型并行组如何创建

核心函数：`initialize_model_parallel`（python/sglang/srt/distributed/parallel_state.py:1383）按如下逻辑创建 3 类组：

- TP 组（张量并行组）：大小 `tp_size`，每组内对同一层做权重/激活切分，层内 All-Reduce/All-Gather（python/sglang/srt/distributed/parallel_state.py:1402）。
- MoE-EP 组（专家并行组）：大小 `ep_size`，用于 MoE 的 token 路由和专家归属（python/sglang/srt/distributed/parallel_state.py:1461-1482）。
- MoE-TP 组（专家内张量并行组）：大小 `moe_tp_size = tp_size / ep_size`，在一个专家内部继续做张量并行（python/sglang/srt/distributed/parallel_state.py:1482-1503）。

举例：`tp=8, ep=4`
- `moe_tp_size = 8 / 4 = 2`
- 会形成 2 个 MoE-EP 组，每组 4 卡（组内处理不同专家的 token）；以及 4 个 MoE-TP 组，每组 2 卡（在单个专家内做并行）。

运行时可在调度器进程名看到 `TPx EPy` 前缀（python/sglang/srt/managers/scheduler.py:2798-2804）。

---

## 三、`--tp` 的语义与影响（适用于所有模型）

- 语义：把同一层的大矩阵拆分到 `tp_size` 张卡上（行/列分片，依实现与层类型而异），每一层前向/后向需 NCCL 通信（如 All-Reduce, All-Gather）。
- 目的：
  - 降低单卡权重占用（近似缩小为 1/`tp_size`）。
  - 扩充可服务模型规模（或在相同模型下给 KV Cache 留出更多显存）。
- 成本：
  - 计算-通信重叠有限时会增加延迟；`tp_size` 越大，通信占比越高。
  - 需要高带宽互联（NVLink/NVSwitch/同机架 IB 更佳）。
- 代码参考：
  - TP 组创建（python/sglang/srt/distributed/parallel_state.py:1402）
  - 模型加载与并行初始化（python/sglang/srt/model_executor/model_runner.py:741）

---

## 四、`--ep` 的语义与影响（仅适用于 MoE 模型）

- 语义：把一层的 `num_experts` 专家均匀切到 `ep_size` 个专家并行组（EP 组）。每个 EP 组持有 `num_experts/ep_size` 个本地专家；路由器把 token 送到对应 EP 组执行专家 MLP，再回传聚合结果。
- 作用：
  - 单卡只需容纳一部分专家权重，极大降低 MoE 权重驻留压力；支持更大/更多专家数。
  - 结合 `moe_tp_size = tp/ep`，在每个本地专家内部继续用 TP 并行。
- 成本：
  - 每步产生 token 级跨卡 All-to-All（或 DeepEP 低延迟变体），对互联带宽/延迟敏感。
  - `num_experts` 不能太小，否则 `ep_size` 过大导致每组专家过少/不均衡。
- 代码参考：
  - EP/TP 组构造（python/sglang/srt/distributed/parallel_state.py:1461-1503）
  - 模型按 EP 切专家权重（python/sglang/srt/models/qwen3_vl_moe.py:266, 278）
  - MoE 专家路由/分发（python/sglang/srt/layers/moe/ep_moe/layer.py:370, 451）

---

## 五、选型建议：该用多大的 `--tp` / `--ep`？

### 5.1 非 MoE（Dense LLM）
- 优先考虑 `tp`，`ep` 固定为 1：
  - 小模型/大显存：`tp=1`（通信最少，延迟最低）。
  - 大模型/显存吃紧：逐步增大 `tp`（例如 2/4/8），以满足“权重 + KV Cache”不 OOM；注意通信开销增加。
- 单机 8 卡示例：70B 模型、80GB×8：`--tp 8 --ep 1`。

### 5.2 MoE 模型（如 Qwen3-MoE/DeepSeek-V3 等）
- 总原则：`tp` 负责切 dense/专家内部算子，`ep` 负责切专家集合。
- 约束与经验：
  - `ep` 只对 MoE 生效；非 MoE 设为 1 即可。
  - 建议 `ep_size` 整除 `tp_size`（源码以 `moe_tp_size = tp/ep` 构组，非整除会导致分组不合理）。
  - 尽量让 `num_experts % ep_size == 0`，避免最后一组专家数不均。
  - 跨卡通信敏感：多机或无 NVLink 的场景，`ep` 不宜过大；单机 NVSwitch 可更激进。
- 推荐模式：
  - `tp=8, ep=4`：每组持有 `1/4` 专家，专家内部 `moe_tp_size=2`，适合 8 卡单机 MoE。
  - `tp=8, ep=8`：每组 1 卡持专家（`moe_tp_size=1`），专家完全分散，All-to-All 压力最大，仅在 NVSwitch/近端 IB 上考虑。
  - `tp=4, ep=2`：All-to-All 压力较小，专家内部 `moe_tp_size=2`，适合中等规模。

### 5.3 实操校验
- 启动日志会打印进程前缀 `TPx EPy` 与可用显存；观察是否 OOM、是否出现 MoE 路由抖动/退避。
- 对 MoE：检查 `experts_per_ep` 是否均匀（python/sglang/srt/models/qwen3_vl_moe.py:278），必要时调整 `ep` 取值。

---

## 六、常见问题与排查

- Q：非 MoE 模型设置 `--ep > 1` 有效果吗？
  - A：没有。EP 仅在 MoE 层生效；Dense 模型请把 `ep` 设为 1。

- Q：`ep` 可以大于 `tp` 吗？
  - A：不建议。分组依赖 `moe_tp_size = tp/ep`，应保证 `ep` 整除 `tp`。

- Q：多机如何设置？
  - A：`world_size = tp * pp`（单实例），跨机通过 `nnodes/node_rank` 组网。TP/EP 组会跨机创建，通信对 IB/NVLink 要求较高。

---

## 七、关联源码一览（便于深入）

- 参数定义：`python/sglang/srt/server_args.py:173, 299`
- 进程 rank 计算：`python/sglang/srt/entrypoints/engine.py:803`
- 分布式并行初始化：`python/sglang/srt/model_executor/model_runner.py:741`
- 模型并行组构造：`python/sglang/srt/distributed/parallel_state.py:1383, 1461, 1482`
- MoE 专家装载/切分：`python/sglang/srt/models/qwen3_vl_moe.py:266, 278`
- MoE 路由/分发：`python/sglang/srt/layers/moe/ep_moe/layer.py:370, 451`
- 调度器进程前缀：`python/sglang/srt/managers/scheduler.py:2798-2804`

---

## 八、结论

- `--tp`：面向所有模型的层内切分手段，权衡“显存占用”与“通信开销”。
- `--ep`：仅对 MoE 生效，用更多 GPU 放下更多专家，代价是跨卡路由（All-to-All）；与 `tp` 组合为 `moe_tp_size = tp/ep`。
- 选型建议：在 Dense 模型优先调 `tp`；在 MoE 模型同时考虑 `tp` 与 `ep`，并遵循“`ep` 整除 `tp`、专家均匀分布、互联足够”三原则。
