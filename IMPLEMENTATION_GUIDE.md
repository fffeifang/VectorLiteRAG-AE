# VectorLiteRAG 具体实现讲解

本文档按“从请求进入到结果落盘”的顺序，讲解仓库中的关键实现，并把核心逻辑与论文 `2504.08930v3-2.pdf` 对齐。

## 1. 系统目标与总体结构

VectorLiteRAG 的核心目标是：

- 在固定 GPU 资源下，同时运行检索（ANN）和 LLM 推理
- 通过 CPU/GPU 混合分区索引满足检索 SLO
- 尽量保持 LLM 吞吐，扩大端到端 SLO 可达的请求率区间

代码分为两大阶段：

1. 离线阶段（Profiling + Partitioning + Index Split）
2. 在线阶段（Hybrid Retrieval + LLM Serving + Dispatcher）

入口在 `main.py`，按参数切换两大阶段。

## 2. 入口与运行模式

文件：`main.py`

- `main()` 根据 `--is_profiling` 选择离线或在线流程。
- 离线流程：
  - `profile(config)`：做延迟/命中率建模与分区点搜索
  - `extract(config)`：根据分区结果切分并落盘 GPU 子索引与映射表
- 在线流程：
  - `run_pipeline(config)`：启动 ANN 进程 + 多个 LLM 进程，跑完整 RAG 流水线

常见参数解析在 `vliterag/args.py`，配置汇总在 `vliterag/configs.py`。

## 3. 配置层实现

文件：`vliterag/configs.py`

`vLiteConfigs` 是系统“控制面”对象，主要职责：

- 读取模型/索引配置（`configs/models/*.json`，`configs/index.json`）
- 推导并发关系：
  - `llm_workers`
  - `ann_workers`（ded-gpu 模式）
- 推导压测请求规模：
  - `total_requests`
  - `warmup_requests`
- 生成结果路径与文件名（用于实验复现与对比）

关键函数：

- `resolve_mode_dependencies()`：处理 `cpu/all-gpu/ded-gpu/vlite` 下的 worker 分配
- `get_tput_ceiling()`：从配置读取裸 LLM 吞吐上限
- `get_model_slo()`：读取模型 prefill 侧 SLO 参考值
- `update_and_sweep()`：扫 arrival rate 用

## 4. 离线阶段：性能建模与分区

文件：`vliterag/profiler.py`

### 4.1 延迟建模（论文 Eq.1 对应）

`LatencyEstimator` 负责：

- 采集不同 batch size 下 ANN latency breakdown
- 建立 piecewise linear 回归模型
- 输出 CPU/GPU 两类回归参数与原始数据 CSV

核心接口：

- `collect_latency_data()`
- `run_regression_model()`
- `estimate_latency(batch_size, min_hitrate=None)`

### 4.2 尾查询命中率建模（论文 Eq.2 对应）

`HitRateEstimator` 负责：

- 根据训练查询统计 centroid 访问频率
- 构造 `ordered_centroids` 与 `cdf`
- 用 Beta 分布近似推导 batch 内最小命中率

核心接口：

- `collect_centroid_data()`
- `compute_min_hitrate(batch_size, exp_mean)`
- `hitrate_binarysearch()`（反解期望均值命中率）

### 4.3 分区点搜索（论文 Algorithm 1 对应）

关键函数：

- `partition_point_iteration()`
- `partitioning_point_search()`

逻辑概括：

1. 根据 `SLOsearch` 推导允许的检索时延预算
2. 结合 KV-cache 与 index memory 的竞争，估算当前可用吞吐
3. 对候选 batch size 估计命中率需求与分区比例
4. 二分迭代直到收敛，得到最佳 partitioning point

### 4.4 分区结果落盘

`save_partitioned_centroids()` 输出：

- `${slo}ms_cids_${nprobe}.npz`：热簇列表
- `${slo}ms_meta_${nprobe}.txt`：分区元信息（命中率、吞吐、向量占比）

## 5. 索引切分与映射表

文件：`vliterag/extractor.py`

`IndexSplitter` 负责把原始 IVF 索引切成可路由的子索引：

- 读取 `ivfpq.index`
- 按热簇集合分配到多个 GPU shard（round-robin + size balance）
- 生成 `.imap` 映射表（orig cid -> shard id + new cid）

关键函数：

- `partition_ivf()`：普通 VectorLite 分片
- `shard_ivf()`：另一路分片流程（兼容 hedrarag 模式）
- `extract()`：总控函数

## 6. 在线阶段：ANN + LLM 协同流水线

文件：`vliterag/runner.py`、`vliterag/engines.py`

### 6.1 进程与队列拓扑

`vLiteQueues` 里定义了核心通道：

- `searchQueue`：请求进入 ANN
- `retrieverQueue`：ANN top-k id -> 文档检索
- `llmQueue[i]`：第 i 个 LLM worker 的输入
- `outQueue[i]`：第 i 个 LLM worker 输出
- `statQueue`：ANN 侧统计信息

### 6.2 ANN 引擎

`ANNSEngine` 初始化时按模式选索引实现：

- `cpu/hedrarag` -> `ShardedIndex`
- `all-gpu/ded-gpu` -> `BaseGPUIndex`
- `vlite + slo>0` -> `PartitionedIndex`

运行时：

1. 批量取请求
2. 执行 ANN 搜索
3. 记录 breakdown/hitrate
4. 通过 callback 或直接回调把 doc ids 发到 `retrieverQueue`

### 6.3 文档检索与拼接

`DocRetriever` 读取 `retrieverQueue`：

- 默认使用 mock 文本（`random_prompt`）
- 如果启用文本库可走 LevelDB
- 结果发往对应 LLM worker 的 `llmQueue`

### 6.4 LLM 引擎

`LLMEngine` 基于 vLLM `AsyncLLMEngine`：

- 每个 worker 绑定自己 GPU 列表（按 rank + tp_size）
- 流式解码记录关键时间戳：
  - submit
  - prefill 首 token
  - completion
- 输出给 `outQueue`

### 6.5 结果汇总

`collect_results()` 汇合 ANN 与 LLM 结果并计算：

- `ttft`
- `e2e`
- ANN queue/batch/pending/search
- prefill/tpot 等

最后由 `vLiteResults` 落盘：

- `summary/*.csv`
- `raw/*_raw.parquet`

## 7. 路由器与动态 Dispatcher（论文关键点）

文件：`index/index_wrapper.py`

### 7.1 Router：按 shard 精细分发

`PartitionedIndex.route_queries()` 会把每个 query 的 centroid probe：

- 映射为目标 shard（或 CPU 回退路径）
- 为每个 shard 生成不同宽度的 `Iq/Dq`
- 避免“所有 shard 固定同 nprobe”的浪费

### 7.2 Dispatcher：提前释放已完成查询

`PartitionedIndex.dispatcher()` 独立线程：

1. 等待各 GPU shard 完成
2. 轮询 CPU callback 队列中的“已完整扫描 query”
3. 立即 merge+rerank 并下发给文档检索

作用：

- 减少批内头阻塞
- 降低 tail query 对整个批次的拖累
- 改善与 LLM 连续批处理的衔接

## 8. 数据预处理与索引构建

主要在 `database/` 与 `index/trainer.py`：

- `database/download.sh`：下载原始/预处理数据
- `database/embedding.py`：多 GPU 编码文本为向量
- `database/split_queries.py`：划分 train/test query 向量与索引
- `index/trainer.py`：
  - `build_ivf()`
  - `build_fs_from_ivf()`（FastScan 版本）
  - `find_groundtruth()`

## 9. 论文图复现到代码映射

文件：`analysis/plot.py`

- Figure 10：`plot_figure_10()`
- Figure 11：`plot_figure_11()`
- Figure 12：`plot_figure_12()`
- Figure 14：`plot_figure_14()`
- Figure 15：`plot_figure_15()`
- Figure 16：`plot_figure_16()`
- Figure 17：`plot_figure_17()`

脚本入口：`scripts/plotall.sh`

## 10. 实现上的注意点（读代码时建议优先关注）

1. `vlite` 模式是论文主路径：
- 先离线 profile/extract，再在线 run

2. `search_slo`、`arrival_rate`、`num_gpus`、`tp_size` 高度耦合：
- 改一个参数通常会影响分区点与可达吞吐

3. runtime 与 profiling 使用不同数据分布：
- profiling 使用训练查询统计
- serving 使用测试查询流（泊松到达）

4. 在线自适应重分片目前是预留接口：
- `PartitionedIndex._repartition_clusters()` 仍是 `pass`
- 当前代码重点在“离线分区 + 在线路由/调度”

---

如果你希望，我可以在这个文档后面再补两章：

- “按函数调用栈逐行走读（含时序图）”
- “当前代码中的已知问题与修复建议清单（按优先级）”
