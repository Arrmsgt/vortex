# vortex SDAA 适配手册

vortex 是 **Evo2 的推理引擎**（PyPI 包名 `vtx`）。`evo2` 仓库只是 912 行的薄壳，
StripedHyena2 的模型、长卷积算子、attention、生成循环全部在这里 ——
**Evo2 能不能在 SDAA 上跑，主要取决于本仓库**。

模型级记录（精度/性能对比、长序列上限、配卡建议）见 Evo2 仓库的
`sdaa/SDAA_ADAPTATION.md`。本文只讲 vortex 自身的迁移。

---

## 一、环境信息

| 项目 | 内容 |
|---|---|
| 机器 | 10.71.13.47 (node175)，龙芯 loongarch64，Loongson-3C6000/D 64 核 |
| 加速卡 | 8 × SDAA 物理卡（各 65536 MB，切成 4 个逻辑 device，每个 14.94 GiB） |
| 容器 | `tuyi_test` |
| Python | 3.12.13（`/home/py312`） |
| PyTorch | 2.12.0 + **torch_sdaa 3.3.1a6+git8b1ba89** |
| 底层栈 | sdaadriver / sdaart / tccl / tecodnn / tecoblas / tecocustom 3.3.0，tecorand 3.1.0，sdapti 1.8.0 |
| 本仓库 | `/data01/tuyilist/vortex`，分支 `adapt/sdaa` |
| 基线 | `main` @ `8b00afe`（Merge PR #77 triton-hc-kernels） |
| 版本核对 | 5 个改动文件的**原始内容与 `vtx 1.1.0` 逐字节一致**（用 pip wheel 验证） |
| 上游依赖 | `evo2`（`/data01/tuyilist/evo2`，editable 安装） |
| Triton | **未安装**（`ModuleNotFoundError: No module named 'triton'`） |

CUDA 对照：10.10.6.21，A100-PCIE-40GB × 1，torch 2.8.0+cu128，**triton 3.4.0 已装**，flash_attn 未装。

---

## 二、分析结论：自定义算子清单与启用状态

**vortex 是带自定义算子的**，而且不止一类 —— 这点必须说清楚，否则会误判工作量：

| 算子 | 位置 | 实现方式 | Evo2 7B 上是否启用 |
|---|---|---|---|
| HCS / HCM / HCL 融合卷积 | `ops/hcs_interface.py`、`hcm_interface.py`、`hcl_interface.py` | **Triton** | ❌ 否 |
| HCL 分块 `compute_filter` | `ops/hcl_interface.py` | **Triton** | ❌ 否 |
| Rotary 位置编码 | `ops/embedding/rotary.py` | **Triton** | ❌ 否 |
| Hyena-SE 前向 / 反向（含 TMA 变体） | `ops/hyena_se/*.py`（fwd/bwd/toeplitz 等） | **Triton** | ❌ 否 |
| Hyena-X 间接卷积 | `ops/hyena_x/triton_indirect_*.py` | **Triton** | ❌ 否 |
| causal_conv1d varlen | `ops/conv/local_causal_conv1d/causal_conv1d_varlen.py` | **Triton** | ❌ 否 |
| causal_conv1d 前/反向/update/step_fir | `ops/conv/csrc/*.cu`、`*.cpp`（5 个文件） | **CUDA C++ 扩展** | ❌ 否 |
| flash-attention 绑定 | `ops/attn_interface.py` | **CUDA 扩展（flash_attn_2_cuda）** | ❌ 否 |
| TMA 载入/存储辅助 | `ops/hyena_se/_*_tma.py`、`kernel_utils.py` | Triton + TMA（Hopper 专属） | ❌ 否 |

**Triton 虽然用 Python DSL 编写，但它同样是设备端 kernel** —— 需要由 Triton 编译器
lower 到目标后端的机器码（CUDA 走 PTX/CUBIN）。**SDAA 没有 Triton 后端**，所以它和
CUDA C++ 扩展在适配上是同一类问题，不能算「没有自定义算子」。

### 为什么这些算子全都没启用

根因不是 SDAA 缺 Triton 后端，而是**上游的一道硬 import**：

```python
# vortex/ops/__init__.py
from .attn_interface import (local_flash_attn_func, ...)   # 无 guard
# vortex/ops/attn_interface.py:7
import flash_attn_2_cuda as flash_attn_gpu                # 无 guard
```

`flash_attn_2_cuda` 在两端都没装 → **`vortex.ops` 整个包不可导入** →
连 `vortex.ops.triton_common` 都进不去 → 所有 `try: from vortex.ops... except ImportError`
一律落到 reference 实现。

在 A100 上实测确认（**triton 3.4.0 装着也一样**）：

```
vortex.ops.triton_common  -> 失败: ModuleNotFoundError No module named 'flash_attn_2_cuda'
vortex.ops.hcs_interface  -> 失败: ModuleNotFoundError No module named 'flash_attn_2_cuda'
```

运行时四个开关，两端**完全一致**：

| 开关 | SDAA | A100 |
|---|---|---|
| `engine.hcs_conv` | `None` | `None` |
| `engine.hcm_fft_conv` | `None` | `None` |
| `engine.hcl_fft_conv` | `None` | `None` |
| `model._hcl_compute_filter` | `None` | `None` |
| `rotary._HAS_TRITON_ROTARY` | `False` | `False` |

### 由此得出的适配决策

**不移植 kernel，全部走降级**。三条理由：

1. **上游已为每个算子备好 torch 参考实现**（`causal_conv1d_ref`、
   `apply_rotary_emb_torch`、`hcl_fft_conv` 的 eager FFT 版等），且 guard 结构已就位 ——
   Evo2 7B 的 32 层（27 个 HCL/HCM/HCS 卷积块 + 5 个 AttentionBlock）全部能跑在参考实现上。
2. **移植 Triton 需要后端支持**，SDAA 生态目前没有 Triton backend，写了也编译不出。
3. **CUDA C++ 扩展需用 `tecocustom` 重写**，而这几个算子在 Evo2 推理路径上
   本来就没被调用（仅 `hyena_se` 的 `use_fast_causal_conv1d=True` 时才走）。

**关键收益**：因为两端走的是同一条 torch 参考路径，**数值天然可比** —— 差异只来自
dtype 策略与后端 kernel 精度，不含「一边融合一边不融合」的路径差异。
反过来说，**性能差距（62–95×）里不包含融合算子的因素**，纯粹是后端
GEMM / FFT / 逐元素算子的效率差异，外加跨卡主机中转的代价。

### 本次仍需要处理的平台差异

算子之外，SDAA 与 CUDA 还有三处行为差异，这才是适配的主要工作：

| # | 差异 | 后果 |
|---|---|---|
| 1 | 不支持直接的 `deviceA → deviceB` 拷贝 | 跨物理卡直接报错；同卡内虽能拷但读到脏数据 |
| 2 | bf16 GEMM 的单项乘积按 **fp16** 精度算（上限 65504） | HCL 层激活到 1e12 → inf → NaN |
| 3 | 非规格化数被 flush 成 0（FTZ） | 极小的频谱分量丢失（非本次 NaN 主因） |

另外 `torch.sdaa.synchronize()` 无参**只同步当前卡**（与 CUDA 同语义，
但 CUDA 的 allocator 会记录 stream 使用、跨卡 copy 有隐式顺序保证，所以不显式同步也不出错）。

---

## 三、改动清单

5 个文件，+345 / −25。所有 SDAA 分支都由
`hasattr(torch, "sdaa") and torch.sdaa.is_available()` 守卫，
**CUDA 上自动跳过，同一份代码两端共用**（已验证 CUDA 侧数值零差异）。

### 3.1 `model/model.py` — 设备抽象

原代码硬编码 `f"cuda:{i}"` 和 `with torch.cuda.device(...)`。改为：

```python
def _device_str(index):
    forced = _os.environ.get("VORTEX_DEVICE", "")
    if forced == "cpu":
        return "cpu"
    return f"{forced.split(':')[0]}:{index}" if forced else f"cuda:{index}"

@contextmanager
def _device_ctx(device):
    ...   # torch.device(device)
```

- `VORTEX_DEVICE=sdaa` 时所有位置自动变成 `sdaa:i`
- `VORTEX_DEVICE_COUNT` 可显式指定卡数（不设则自动探测）
- 同步改了 `model/utils.py` 的默认设备辅助函数

### 3.2 `model/model.py` — `update_time()` 每次重建

原实现缓存时间向量、只在 L 变大时重建，否则截断：

```python
if self.t is None:            self.t = torch.arange(L, ...)
elif self.t.shape[-1] < L:    self.t = torch.arange(L, ...)
else:                         self.t = self.t[..., :L]
```

变长调用时会拿到错误切片（先用长序列、再用短序列就会错）。
改为每次重建（O(L)，代价可忽略）：

```python
self.t = torch.arange(L, device=device)[None, None]
```

### 3.3 `model/model.py` — 跨卡传输改走主机中转（核心改动）

SDAA **不支持直接的 `deviceA → deviceB` 拷贝**：

- 跨物理卡：`RuntimeError: SDAA Copy for different cards has not been supported`
- 同物理卡内的 4 个逻辑切片：能拷，但**完成信号记在源卡的队列上，
  目标卡的 `.item()` / `.cpu()` 不等待** → 读到上一轮的残留数据

最小复现（32 MB、2 个逻辑切片、4 轮）：

```python
for it in range(4):
    tag = float(it + 1)
    a = torch.full((N,), tag, dtype=torch.float32, device="sdaa:1")
    b = a.to("sdaa:0")                # 跨卡拷贝
    got.append(float(b[0].item()))    # 立刻读回主机
# 期望 [1, 2, 3, 4]，实际 [1, 2, 2, 3]
```

判定实验（说明问题在「直连拷贝」这条路径）：

| 做法 | 结果 |
|---|---|
| 不同步 | `[1, 2, 2, 3]` ❌ |
| 只同步**目标卡** | `[1, 2, 2, 3]` ❌ |
| 只同步**源卡** | `[1, 2, 3, 4]` ✅ |
| 逐卡同步 | `[1, 2, 3, 4]` ✅ |
| 不复用显存块 | `[1, 2, -0.0023, -0.0034]` ❌ 读到未写入的垃圾 |
| **走主机中转 `a.cpu().to(...)`** | `[1, 2, 3, 4]` ✅ |

改法：所有跨设备搬运统一走 `deviceA → CPU → deviceB`
（调用点两处：`cross_device_transfer()` 与 `forward()` 末尾搬回 0 号卡）：

```python
def _move_between_devices(x, dst):
    if not (hasattr(torch, "sdaa") and torch.sdaa.is_available()):
        return x.to(dst)                    # CUDA 保持直连
    if _os.environ.get("EVO2_COPY_VIA_HOST", "1") == "0":
        return x.to(dst)
    return x.cpu().to(dst)                  # 主机中转
```

效果（Evo2 7B，512bp）：

| 配置 | L128_0 | L512_0 | 512 耗时 |
|---|---|---|---|
| 直连拷贝 + 不同步 | −5.210476 | −6.118560 | 7.04 s ❌ |
| 直连拷贝 + 逐卡同步 | −1.326372 | −1.325365 | 4.32 s ✅ |
| **主机中转 + 不同步（默认）** | **−1.326372** | **−1.325365** | **5.46 s** ✅ |

代价：短序列慢约 25%，2048 以上持平。收益：**跨物理卡可用**
（4 逻辑卡 → 8/16 卡都能配起来，长序列上限从 5888 提到 7424 / 8192）。

同时保留 `_sync_sdaa_devices()` + `EVO2_SYNC_MODE`（`boundary`/`all`/`none`）
作为关闭主机中转时的兜底 —— **不能用无参 `synchronize()`，它只同步当前卡**。

### 3.4 `model/model.py` — 溢出层转 fp32（核心改动）

**SDAA 的 bf16 GEMM 把单项乘积按 fp16 精度算**，上限 65504，超出即 `inf`；
CUDA 是 fp32 累加，能到 3.4e38。实测阈值（`1×4096 @ 4096×1`，全 1 操作数）：

| 单项乘积 | SDAA | CUDA |
|---|---|---|
| 60000 | 2.45e8 ✅ | 2.45e8 ✅ |
| 65000 | 2.66e8 ✅ | 2.66e8 ✅ |
| **70000** | **inf ❌** | 2.87e8 ✅ |
| 100000 | **inf ❌** | 4.09e8 ✅ |

Evo2 的层输出实测量级：block27 = 756 → block28 = 2080 →
block29 = 7.3e5 → **block30 = 4.9e12**（CUDA 上同样是 4.88e12，说明巨值是模型固有特性）。
所以纯 bf16 必然溢出。

改法：`_wrap_fp32_forward(block)` 把指定 block 的参数转 fp32、forward 内部以
fp32 计算、输出转回原 dtype。三个关键点：

1. **哪些层**：`EVO2_FP32_IDXS` 指定；默认对 32 层的 Evo2 7B 取 **`29,30,31`**，
   且**仅在 SDAA 上生效**（CUDA 保持纯 bf16 —— 这一条是必须的，否则两端路径又不一致了）。
2. **加载豁免**：`load_checkpoint()` 在模型构造后会调用
   `to_bfloat16_except_pr_lc()` 把所有参数压回 bf16 —— 不加豁免会把 fp32 白做。
   已在 `_fp32_prefixes` 里跳过标了 `_fp32_compute` 的 block。
3. **层数是最小集**：实测只要 3 层，不是「所有 HCM/HCL」的 18 层：

| fp32 层配置 | L512 分数 | 512 耗时 |
|---|---|---|
| 0 层（纯 bf16） | nan | 3.86 s |
| 30,31 | nan | 5.63 s |
| **29,30,31（默认）** | **−1.325365**（CUDA −1.325327） | **4.54 s** |
| 28,29,30,31 | −1.325381 | 4.55 s |
| 27~31 | −1.325350 | 6.24 s |
| 全部 attn（5 层） | nan | 4.41 s |
| 所有 HCM/HCL（18 层） | −1.325228 | 7.62 s |

（保留 `EVO2_FP32_LAYERS` / `EVO2_FORCE_FP32` 两个旧开关便于对照实验）

### 3.5 `model/rotary.py` — 补上缺失的 import guard

`vortex/ops/__init__.py` 的硬 import（见第二章）导致整个 ops 包不可用，
但这处**原本没有 guard**：

```python
from vortex.ops.embedding.rotary import apply_rotary      # 裸 import，未 try
```

其余三处同类 import（`engine.py` 的 HCS/HCM/HCL、`model.py` 的
`_hcl_compute_filter`、`attention.py` 的 `local_flash_attn_*`）**上游已经包了
`try/except ImportError`** —— 所以 `import vortex` 在 SDAA 上能过，只有 rotary
这一处会炸。已补上 `try/except ImportError`，回落到 torch 参考实现
（`apply_rotary_emb_torch` / `apply_rotary_emb_qkv_torch`）。

另加 `VORTEX_ROTARY_FORCE_TORCH=1` 强制走参考实现 —— 用于与 CUDA 对标时
统一数值路径（A100 上 `_HAS_TRITON_ROTARY` 也是 False，所以默认本来就一致）。

### 3.6 `model/engine.py`、`model/generation.py` — 设备无关

- `engine.py`：`empty_cache` 改为跟随当前张量设备（+2/−1）
- `generation.py`：`memory_allocated` / 默认设备（+13/−12）

---

## 四、验证

### 4.1 验证矩阵

| # | 项目 | 方法 | 结果 |
|---|---|---|---|
| 1 | 版本基线 | 与 pip 下载的 `vtx 1.1.0` 逐文件 diff | ✅ 5 个文件原始内容完全一致 |
| 2 | 语法 | py312 `ast.parse` 全部 58 个 py 文件 | ✅ 0 错误 |
| 3 | 加载来源 | `import vortex; print(vortex.__file__)` | ✅ 指向仓库，不是 site-packages |
| 4 | **算子路径一致性** | 两端打印 `hcs_conv`/`hcm`/`hcl`/`_hcl_compute_filter`/`_HAS_TRITON_ROTARY` | ✅ 两端均为 `None/None/None/None/False` |
| 5 | SDAA 数值（15 条 + batch8） | 与适配前基线逐位比对 | ✅ 全部一致 |
| 6 | SDAA 4096 | 同上 | ✅ −1.300965 / −1.317691 / −1.394270 |
| 7 | **CUDA 回归（15 条 + batch8）** | A100 上跑同一份代码 | ✅ **最大差 0.000e+00** |
| 8 | `generate()` 冒烟 | prompt `"ACGT"×8`，生成 4 token | ✅ 输出 `"ACGT"`，Score −0.0513 |

第 4 项作用是把「数值可比」的前提钉死：**如果哪天装了 flash-attn 或解耦了
`ops/__init__.py`，Triton 融合路径会突然启用，数值路径就变了** —— 到时候必须重跑
全部精度对比，不能拿旧数据糊弄。

第 7 项是必须的：本次改动了 `cross_device_transfer()` 和 `forward()` 末尾的
跨设备搬运，必须确认 CUDA 路径没被影响 —— 实测零差异。

### 4.2 复现命令

```bash
source /opt/tecoai/setvars.sh
source /home/py312/bin/activate
export PYTHONPATH=/data01/tuyilist/vortex:/data01/tuyilist/evo2
export VORTEX_DEVICE=sdaa VORTEX_DEVICE_COUNT=8
export SDAA_VISIBLE_DEVICES=4,5,6,7,8,9,10,11
export TORCH_SDAA_BF16_CLIP=0

python - <<'EOF'
import vortex
print("加载来源:", vortex.__file__)      # 必须是 .../vortex/vortex/__init__.py
from evo2.models import Evo2
m = Evo2("evo2_7b_base", local_path="/data01/tuyilist/evo2_weights/evo2_7b_base.pt")
print(m.score_sequences(["ACGT" * 128], batch_size=1))
EOF
```

期望分数（512bp 用 `evo2_bench_seqs.json` 里的 L512_0）：**−1.325365424**。

### 4.3 换位置/升级后的自检

`vortex_src`（旧安装目录）和 site-packages 里的 `vtx` 都可能被误加载。
**唯一可靠的检查是打印 `vortex.__file__`**，然后再跑一遍数值回归：

```
L128_0  -1.326371789
L512_0  -1.325365424
L2048_0 -1.326315999
```

---

## 五、与 evo2 的配合

```
vortex（本仓库，推理引擎）  ←── 被依赖 ──  evo2（薄壳，tokenizer/scoring/入口）
        ↑ adapt/sdaa                        ↑ adapt/sdaa（2 文件 +60/−5）
```

- 两边**各自独立分支**，`PYTHONPATH` 同时指向两个仓库根目录
- vortex 改了 device/跨卡/精度策略，evo2 只改了「flash_attn 降级」和
  「读取结果前同步」，两者是互补的、不重叠
- evo2 侧 `scoring.py` 的 `_maybe_sync_devices()` 是为了让**用户不用手动同步**；
  vortex 侧的主机中转则从根上避免了脏读

部署顺序：先确认 vortex 能被正确 import（看 `vortex.__file__`），再跑 evo2。
evo2 只依赖 vortex 的公开 API（`StripedHyena`、`dotdict`、`load_checkpoint`、
`default_device`），本次改动没有变更任何函数签名，所以**上游 evo2 无需修改**。

---

## 六、已知限制

1. **跨卡必须走主机中转**。SDAA 不支持直连，这是平台约束不是本仓库的选择。
   代价：短序列慢约 25%。
2. **纯 bf16 跑不了 Evo2 7B**，必须保留 block 29/30/31 的 fp32，
   根因是 bf16 GEMM 的 fp16 乘积上限（65504）。这一条理想上应由框架修，
   模型侧的 fp32 只是绕过。
3. **`torch.sdaa.synchronize()` 无参只同步当前卡**，多卡场景必须逐卡调用。
   本仓库的 `_sync_sdaa_devices()` 已封装好。
4. **`TORCH_SDAA_BF16_CLIP=1` 不要用**：它把 NaN 变成 `-3.216747` 这类
   看着正常的错值，比 NaN 更难发现；正确中转后本就不需要。
5. **所有自定义算子（Triton ×6 组 + CUDA 扩展 + flash-attn）在两端都未启用**，
   跑的是 torch 参考实现。这不是适配选择，而是上游 `ops/__init__.py` 硬 import
   flash-attn 的副作用。**因此 62–95× 的性能差距里不含融合算子的因素。**

---

## 七、后续可优化项（未做，需评估）

启用融合算子需要**同时**满足以下条件，缺一不可：

| 条件 | 现状 | 说明 |
|---|---|---|
| ① 解耦 `vortex/ops/__init__.py` 对 flash-attn 的硬 import | 未做 | 上游问题，可提 issue；改法是把 `attn_interface` 挪进 `try/except` |
| ② SDAA 侧有 Triton 后端 | **不具备** | Triton 官方无 loongarch/SDAA backend；`import triton` 都不可用 |
| ③ CUDA 扩展用 `tecocustom` 重写 | 未做 | 只涉及 `causal_conv1d` 系列，且其仅在 `use_fast_causal_conv1d=True` 时使用 |
| ④ 重做全部精度对比 | 必须 | 换 kernel 就换数值路径，旧数据作废 |

现实结论：②是硬阻塞，短期无法解决。因此**融合算子这条路先不走**，
性能优化应优先看后端算力（GEMM/FFT 实现）而不是算子融合。

---

## 八、踩坑记录

1. **不要拿 HEAD 直接适配**。先确认基线版本和已安装的 `vtx` 一致，
   否则会把上游改动混进适配里。本次用 `pip download vtx==1.1.0` 解出 wheel
   逐文件 diff 验证，5 个目标文件全部一致才动手。
2. **`update_time` 的缓存是真 bug**，不是 SDAA 特有：先用长序列、再用短序列就会错。
   排查数值问题时它也会掩盖现象，建议先确认（本次已改成每次重建）。
3. **`to_bfloat16_except_pr_lc()` 会覆盖构造期的 dtype 设置**。
   给某些层提 fp32 时，必须同步改这个方法的豁免列表，
   否则 `load_checkpoint` 一跑就全变回 bf16（表现为「改了没生效」）。
4. **「分数偏低一个常数」先换算对数**。SDAA 修前是 −6.1186 ≈ `ln(512)`，
   即输出退化成全词表均匀分布；CUDA 是 −1.3253 ≈ `ln(4)`。
   一眼就能判断是「没输出信号」而不是「精度差一点」。
5. **`empty_cache()` 治不了这个坑**。它只归还缓存块，不等待未完成的拷贝。
6. **跨卡拷贝的完成信号归源卡**。同步目标卡无效、同步源卡有效 ——
   这条对定位其它多卡问题也有用。
7. **`set -u` + `setvars.sh` 会静默退出**。写验证脚本时别在 source 之前加 `set -u`，
   否则脚本什么都不输出、`exit=1`，很难查。
8. **别信「装了 triton 就会走 Triton 路径」**。A100 上 triton 3.4.0 在，
   但 `hcs_conv` 依然是 `None` —— 因为 `vortex.ops` 包的 `__init__` 先死在了
   flash-attn 上。判断某个算子到底跑没跑，**要打印运行时的 flag，不要看装了什么包**。

---

## 九、分支与提交

| 项目 | 内容 |
|---|---|
| 分支 | `adapt/sdaa`（基于 `main` @ `8b00afe`） |
| 改动 | 5 文件，+345 / −25 |
| 新增 | `sdaa/VORTEX_SDAA_ADAPTATION.md`（本文）、`sdaa/README.md`（索引） |
| 未提交 | 改动仍在工作区，等确认后再 commit |