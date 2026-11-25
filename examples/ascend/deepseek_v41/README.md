# 在 Ascend 910B 上运行 DeepSeek V4.1 Flash

这套适配在单机八张 Ascend 910B 上支持原始 **MXFP4** checkpoint 和 **QuaRot G32 INT4** checkpoint。路由专家以压缩格式常驻 NPU，由 Ascend C grouped GEMM 在计算时解码当前权重 tile。部署验证使用八张 64 GiB 的 910B3 和 1017 checkpoint。[模型权重](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) · [模型技术报告](https://arxiv.org/html/2609.19969)

## 适配脉络

### 权重驻留与算子实现

模型骨干有 552B 参数，Engram 另有 196B 参数。EP8 将每层 384 个路由专家均分，每张 NPU 驻留 48 个压缩专家。MoE 路由按专家归组 token。原始 checkpoint 的 grouped GEMM 将 MXFP4 权重和 scale 解码为 FP16 tile，再用 FP32 累加。QuaRot checkpoint 的 target 路由专家使用 G32 W4A8，target shared experts 使用 W8A8。[QuaRot G32 格式与导出](quarot-g32.md)

部分 dense 投影和原始 checkpoint 的 shared experts 在加载时将 FP8 权重转为沿 K 维分组的 INT8，每组 256 个元素。W8A16 GEMM 从显存读取压缩权重并逐 tile 反量化，输入激活和输出用 BF16。Engram 保留 FP8/E8M0 格式，两张大表由主机内存承载，合计约 189 GiB，节点内各 rank 共享。

### 长上下文与推测解码

在 910B 上，paged attention 的 QK/PV 由矩阵核计算，向量核处理 softmax、mask 和稀疏位置。主 KV 和滑窗 KV 用 BF16，Indexer 用 INT8 key 选择稀疏位置。CED 使用因果 encoder 的输出构建 decoder 的 global KV。Radix 命中时复用已缓存的全局状态，bounded SWA replay 恢复滑窗所需的局部状态。

Decode 使用 DSpark，每步最多提出五个草稿 token。Target Verify 和 Draft 都使用 NPU Graph，Prefill 走 eager。MoE 预填充按最多 8,192-token 窗口推进，设备端直接读取本卡命中的专家行数，免去逐层同步。Decode 的 EP8 输入使用 HCCL 聚合 BF16 token。固定几何下，本卡 shared expert 和 MoE `mhc_stats` 在辅助流上与聚合及路由专家计算重叠，在 MoE 返回前汇合。启用 [Peer 通信](peer-communication.md) 后，符合容量的 Decode 使用 TP2 组内聚合及融合 EP8 返回。

### Attention 和 MoE 分开并行

默认拓扑简写为 **`DP4(CP2/DP2) + EP8`**。Attention DP4 将八卡分成四个组，每组两卡。组内 Prefill 用 CP2 按序列切分 token，Decode 用 DP2 按请求分工。MoE EP8 跨八卡。

| Attention 配置 | Attention 请求分布 | 单组本地 prefill chunk | MoE |
| --- | --- | ---: | --- |
| DP8/TP1 | 八组，每条请求的 attention 和 KV 落在一张卡 | 4,096 | EP8 |
| `DP4(CP2/DP2)`（默认） | 四组，组内两卡共享 KV/状态镜像，以完整 query heads 交错执行 prefill、按请求对半执行 decode | 8,192 | EP8 |

DP8 把独立请求散到更多组。默认 DP4 在组内两卡使用完整 query heads，通过交错 token 或分配请求分担计算，并同步 KV 与相关状态。单请求长 prefill 的 attention 只使用所在组，MoE 由八卡共同计算。两个配置均使用 32,768-token 全局 chunk 预算。

### 主机扩展与设备核

CMake 在镜像构建时使用 GCC/G++ 13 编译主机扩展。DeepJIT 在启动和首次执行相应形状时，使用 CANN 编译 Ascend C 设备核。Compose 的 `deepjit-cache` 卷在重启时复用编译产物。实现目录和开发验证见[后端说明](../../../python/sglang/srt/hardware_backend/npu/dsv41/README.md)。

## 构建 Docker 镜像

在 Linux ARM64 主机的仓库根目录运行。准备 Ascend 910B 主机驱动、Docker Buildx、Compose v2、Git、`curl` 和 `sha256sum`，并确认 `npu-smi info` 能看到八张卡。模型磁盘空间约 476 GiB，主机内存还需覆盖 Engram 表、权重加载和运行开销。

```bash
bash examples/ascend/deepseek_v41/build-image.sh
```

脚本把当前 Git 提交打包为构建上下文，从公开发布源获取 ARM64 依赖并验证 SHA-256。默认依赖缓存为 `build/dsv41-deps`，可用 `DSV41_DEPS_DIR` 指向已准备好的同版本包。Dockerfile 使用 CANN 9.1.0、Python 3.12、PyTorch 2.10.0 和 `torch-npu` 2.10.0.post6，镜像标签默认为 `sglang-dsv41:910b`。`DSV41_IMAGE` 可覆盖标签，`--build-arg NATIVE_JOBS=4` 可限制主机扩展的并行编译数。[Dockerfile](Dockerfile) · [SGL Kernel NPU 依赖](https://github.com/sgl-project/sgl-kernel-npu/releases/tag/2026.9.0.post9) · [Triton Ascend 依赖](https://github.com/triton-lang/triton-ascend/releases/tag/v3.2.2)

## 启动与验证

通过 `DSV41_MODEL_DIR` 选择完整的 1017 checkpoint。两种模式使用同一镜像，SGLang 根据 checkpoint 配置加载对应的 target MoE。Compose 以只读方式挂载模型，默认监听 `127.0.0.1:31088`，启动使用 `DP4(CP2/DP2) + EP8`（`DSV41_ATTN_DP_SIZE=4`）。

```bash
export DSV41_MODEL_DIR=/path/to/DeepSeek-V4.1-Flash-1017
docker compose -f examples/ascend/deepseek_v41/compose.yaml up -d
docker compose -f examples/ascend/deepseek_v41/compose.yaml logs -f sglang
```

将 `DSV41_MODEL_DIR` 改为 QuaRot G32 checkpoint 目录，再运行 `up -d` 切换为 INT4。

```bash
export DSV41_MODEL_DIR=/path/to/DeepSeek-V4.1-Flash-1017-quarot-int4-g32
docker compose -f examples/ascend/deepseek_v41/compose.yaml up -d
```

用同一份 Compose 切换到 Attention DP8/TP1。若要使用传统 DP4/TP2，将 `SGLANG_OPT_DSV41_GROUP_CP2_DP2=0` 加入 Compose 服务的 `environment`。

```bash
DSV41_ATTN_DP_SIZE=8 docker compose -f examples/ascend/deepseek_v41/compose.yaml up -d
```

`/ready` 返回 HTTP 200 后执行生成和前缀复用检查。`DSV41_PORT` 自定义端口时，同步传给 smoke 脚本的 `--url`。

```bash
curl --fail http://127.0.0.1:31088/ready
docker compose -f examples/ascend/deepseek_v41/compose.yaml exec sglang \
  python3 examples/ascend/deepseek_v41/smoke.py --parallel-requests 4 --prefix-replay
```

## 跨会话复用 KV 前缀

叠加 [`compose.router.yaml`](compose.router.yaml) 后，每个 attention DP rank 发布本地 Radix 缓存的写入和淘汰事件。`sgl-router` 对文本请求查找可复用的最长 token 前缀，将请求定向到对应 rank。前缀未命中或命中长度相同时，默认选路由器在途请求最少的候选 rank。DP8/TP1、DP4(CP2/DP2) 和传统 DP4/TP2 共用此配置。[DP rank 路由说明](../../../sgl-router/REFERENCE.md#dp-rank-routing)

从仓库根目录构建路由器镜像，再用两份 Compose 文件启动。引擎和路由器共用 `DSV41_MODEL_DIR`。切换 checkpoint 时，沿用同一组 Compose 文件执行 `up -d`，让两个服务都更新模型挂载。所选 checkpoint 的 `tokenizer.json` 和 `config.json` 需可读。部署时可用 `DSV41_IMAGE` 指定已有引擎镜像。引擎在发布 KV 事件时会编译哈希扩展，使用这个 Dockerfile 构建的镜像包含扩展所需的 `libssl-dev`。

```bash
git archive HEAD docker/sgl-router.Dockerfile sgl-router rust | \
  docker buildx build --load --platform linux/arm64 -t sgl-router:dsv41-910b \
    -f docker/sgl-router.Dockerfile -

docker compose -f examples/ascend/deepseek_v41/compose.yaml \
  -f examples/ascend/deepseek_v41/compose.router.yaml up -d
```

将文本客户端的 OpenAI base URL 设为 `http://127.0.0.1:30000/v1`，模型名用 `deepseek-ai/DeepSeek-V4.1-Flash-1017`。图文请求可直接使用引擎 API `http://127.0.0.1:31088/v1`。路由器端口可用 `DSV41_ROUTER_PORT` 调整。KV 事件和回放端口只绑定回环地址，DP8 默认使用 `5557–5564` 和 `5567–5574`，各自的起始端口由 `DSV41_KV_EVENTS_PORT` 和 `DSV41_KV_REPLAY_PORT` 控制。

下面的检查在引擎 rank 1 预热一段 1,024-token 前缀，等待路由器收到 KV 事件，再经路由器发一条后缀不同的请求。两条请求各带 256 个后缀 token，为 bounded SWA replay 留出尾部计算空间。检查会核对实际执行 rank 和 `cached_tokens`。

```bash
curl --fail http://127.0.0.1:30000/readyz
docker compose -f examples/ascend/deepseek_v41/compose.yaml \
  -f examples/ascend/deepseek_v41/compose.router.yaml exec -T sglang \
  python3 - --router-url http://127.0.0.1:30000 \
  < examples/ascend/deepseek_v41/smoke.py
```

## 8×910B3 实测

SGLang 两套权重均在提交 `22f333e0d` 上通过 smoke 并完成测试。硬件为单机八张 64 GiB 910B3。SGLang 使用 1017 原始 MXFP4 或免校准 QuaRot INT4 G32 checkpoint，以及默认 `DP4(CP2/DP2) + EP8` 拓扑。vLLM-Ascend 对照使用 vLLM v0.33.0、Eco-Tech W4A8 checkpoint 和两套单机 EP8 拓扑。

SGLang 使用 32,768-token 全局 Prefill chunk、Radix、CED 和 bounded SWA replay。DSpark 使用五个草稿，Target Verify 和 Draft 都使用 NPU Graph，本地 batch 捕获至 32。用以下设置从仓库根目录启动受测配置，并按权重格式替换 `DSV41_MODEL_PATH`。

```bash
DSV41_MODEL_PATH=/path/to/checkpoint \
DSV41_ATTN_DP_SIZE=4 DSV41_MAX_RUNNING_REQUESTS=128 DSV41_MEM_FRACTION_STATIC=0.84 \
SGLANG_OPT_DSV41_NPU_PEER_COMM=1 \
  bash examples/ascend/deepseek_v41/serve.sh --random-seed 42 --decode-log-interval 1
```

vLLM-Ascend 使用 DSpark 五草稿和 `FULL_DECODE_ONLY` ACL Graph，Draft 走 eager。vLLM 的 DP8/TP1 在 4,096-token 批处理上限下冷 Prefill 遇到 OOM，表中使用 2,048-token 上限与 2.2 GB KV 缓存配置。

![8×910B3 的 32K Decode 并发曲线和 128K 单请求冷 Prefill 吞吐](benchmark-910b.png)

### 32K 输入的稳态 Decode

各配置逐档使用相同 token ID。输入由 28,672-token 共享前缀和 4,096-token 独立后缀组成。满批后运行 50 步预热和 60 步预运行，再测 60 步。采样使用 temperature 0.3、top-p 0.95、top-k 1024、top-5 logprobs 和 `ignore_eos=True`。vLLM DP8/TP1 在部分档位的实际前缀命中低于 87.5%，并发 128 时为 86.8%。

每格为**实测单步耗时（ms/step） · 折算单卡吞吐（tok/s/NPU）**。曲线按每步每请求固定生成 `1 + 5 × 0.75 = 4.75` 个 token 折算。每用户速度为步频乘以 4.75，单卡吞吐再乘整机并发并除以 8。[DeepInfer 稳态测量方法](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/examples/README.md#stage-decoupling-and-metrics)

| 整机并发 | SGLang INT4 G32 | SGLang MXFP4 | vLLM-Ascend DP8/TP1 | vLLM-Ascend TP8/DP1 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 29.49 · 20.13 | 31.90 · 18.62 | 59.22 · 10.03 | 66.08 · 8.98 |
| 2 | 31.85 · 37.28 | 35.13 · 33.81 | 60.72 · 19.56 | 65.17 · 18.22 |
| 4 | 34.66 · 68.52 | 40.59 · 58.52 | 61.37 · 38.70 | 61.65 · 38.52 |
| 8 | 39.46 · 120.36 | 46.74 · 101.62 | 61.17 · 77.66 | 63.23 · 75.13 |
| 16 | 46.23 · 205.51 | 56.67 · 167.63 | 63.61 · 149.35 | 65.80 · 144.38 |
| 32 | 56.10 · 338.66 | 68.55 · 277.17 | 69.74 · 272.45 | 91.49 · 207.68 |
| 64 | 68.47 · 555.00 | 80.82 · 470.19 | 71.99 · 527.88 | 114.94 · 330.59 |
| 128 | 84.99 · 894.20 | 99.41 · 764.50 | 91.69 · 828.88 | 181.23 · 419.36 |

并发 128 时，INT4 G32 为 84.99 ms/step，较 vLLM DP8/TP1 的 91.69 ms/step 低 6.70 ms。四列在该档的实际草稿接受率分别为 35.2%、36.1%、38.3%、37.2%，实测整机输出吞吐分别为 4,156、3,613、4,070、2,019 tok/s。

### 128K 单请求冷 Prefill

每套配置先用同长度输入预热并清空前缀缓存，再测两次全长未缓存的 131,072-token 输入，每次只生成一个 token。首 token 时延包含服务端调度和所有 Prefill chunk。输入吞吐用 131,072 除以首 token 时延的中位数。

| 配置 | 单卡权重（GiB） | 首 token 中位数（s） | 两次测量（s） | 输入 tok/s |
| --- | ---: | ---: | ---: | ---: |
| SGLang INT4 G32 | 44.44 | 8.333 | 8.306 / 8.360 | 15,729 |
| SGLang MXFP4 | 45.37 | 8.374 | 8.365 / 8.382 | 15,653 |
| vLLM-Ascend DP8/TP1 | 53.73 | 50.259 | 50.179 / 50.338 | 2,608 |
| vLLM-Ascend TP8/DP1 | 42.76 | 15.766 | 15.768 / 15.764 | 8,314 |
