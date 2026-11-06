# vortex — SDAA / 龙芯 适配

本目录是 vortex 在国产加速卡（SDAA，龙芯 loongarch64）上的适配产物。

| 文档 | 内容 |
|---|---|
| **[VORTEX_SDAA_ADAPTATION.md](VORTEX_SDAA_ADAPTATION.md)** | **完整迁移适配手册**：算子清单与启用状态、平台差异实测、5 个文件的逐项改动、验证矩阵、已知限制、踩坑记录 |

## 一句话结论

vortex 是 Evo2 的推理引擎（PyPI 包 `vtx`），Evo2 能否在 SDAA 上跑**主要取决于本仓库**。

**它是带自定义算子的**：6 组 Triton kernel（rotary / HCS·HCM·HCL 融合 / Hyena-SE / Hyena-X /
causal_conv1d varlen）、1 个 CUDA C++ 扩展（`ops/conv/csrc/`，5 个文件）、以及 flash-attn 绑定。
Triton 用 Python DSL 写，但同样要 lower 到设备机器码，**算自定义算子**。

不过这些算子**在 SDAA 和 A100 上都没启用** —— 根因是 `vortex/ops/__init__.py` 硬 import
`flash_attn_2_cuda`，整个 `ops` 包不可导入，全部落到上游备好的 torch 参考实现。
所以两端走同一条数值路径、结果可比；**62–95× 的性能差距里不含融合算子因素**。

本次适配的主要工作是三处平台差异：

1. 不支持 `deviceA → deviceB` 直连拷贝 → **跨卡搬运改走主机中转**
2. bf16 GEMM 的单项乘积按 fp16 精度算（上限 65504）→ **block 29/30/31 转 fp32**
3. `torch.sdaa.synchronize()` 无参只同步当前卡 → **逐卡同步封装**

## 快速自检

```bash
source /opt/tecoai/setvars.sh && source /home/py312/bin/activate
export PYTHONPATH=/data01/tuyilist/vortex:/data01/tuyilist/evo2
export VORTEX_DEVICE=sdaa VORTEX_DEVICE_COUNT=8 SDAA_VISIBLE_DEVICES=4,5,6,7,8,9,10,11
python -c "import vortex; print(vortex.__file__)"   # 必须指向本仓库，不是 site-packages
```

期望分数：L512_0 = **−1.325365424**（CUDA 参考 −1.325327158）。
模型级精度/性能对比见 `evo2/sdaa/SDAA_ADAPTATION.md`。

## 分支

`adapt/sdaa`，基于 `main` @ `8b00afe`（原始内容与 `vtx 1.1.0` 逐字节一致）。
改动 5 文件 +345/−25，CUDA 侧实测零差异（同一份代码两端共用）。