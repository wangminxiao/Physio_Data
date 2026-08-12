# MIMIC-III PLETH 4× 量程分裂:诊断与修复(2026-08-07)

## TL;DR

MIMIC-III store 里 39.7% 的记录(2,231/5,621)的 PLETH40 整体落在 [0, 4] 量程,
其余落在 [0, 1],两组相差恰好 4.003×。这不是抽取 bug,也不是生理差异,而是
**mimic3wdb-matched 源数据 `.hea` header 的标定缺陷**:12-bit 记录的 PLETH gain
没有随 ADC 位深从 1023 改成 4095。已于 2026-08-07 在 bedanalysis 上修复
(高幅组统一 ×1023/4095),修复后全店单峰、无记录超出 [0, 1.12]。
**ECG(II120)不受影响。** dream 上的同步副本尚未更新;MIMIC-III 的波形
归一化参数(global_affine)需要在修复后的数据上重估。

## 症状

对 store 做 per-record 窗口统计,per-window mean 和 std 都呈双峰:

| | mean | std | CV = std/mean |
|---|---|---|---|
| 低幅组(3,390 条) | 0.460 | 0.150 | 0.330 |
| 高幅组(2,231 条) | 1.824 | 0.618 | 0.341 |
| 比值 | 3.97× | 4.12× | 1.03× |

mean 和 std 同步放大、CV 不变 —— 纯乘性缩放的签名,排除生理解释
(灌注差异会改变搏动/基线比例,即改变 CV)。两组的上界分别是 ~1.02 和
~4.07,即一组 [0, 1]、一组 [0, 4]。log(std) 直方图两峰完全分离,中间一段空。

## 根因

回源头对 WFDB header(每组抽 40 条 + 原始 ADC 逐点验证),三种采集配置:

| 源配置 | PLETH gain | ADC 范围 | 物理值 = raw/gain | 结果 |
|---|---|---|---|---|
| 8-bit (fmt80) | 255 | 0–255 | [0, 1] | ✓ 正常 |
| 10-bit (fmt16) | 1023 | 0–1023 | [0, 1] | ✓ 正常 |
| **12-bit (fmt16)** | **仍是 1023** | **0–4095** | **[0, 4.003]** | ✗ 未随位深重标 |

4095/1023 = 4.003,正是观测到的比例。对照组:**II(ECG)的 gain 随位深
同步缩放了**(127 → 512 → 2046),所以 II120 单峰、干净 —— 且 PLETH 分组
完全不预测 II 幅度(两组 II std 中位数 0.158 vs 0.164),与该机制自洽。

关键澄清:

- **抽取管线是忠实的。** stage3 只有一条读取路径(`wfdb.rdrecord` 默认
  physical + `p_signal`),并在原始层验证 `(d_signal − baseline)/gain ==
  p_signal` 精确成立。任何按 WFDB 规范读物理值的工具都会得到同样的 4× 分裂。
- **量程按记录恒定。** 全店逐 entity 扫描,零混合病人 —— 每个 entity 内
  所有窗口同一量程,因此按记录乘一个常数即可完全修复。
- 此前怀疑的 gain=0 缺省值(→200)不成立:所有 header 都写了真实 gain。

## 修复

两层,同一天完成(commit `8610fa4`,含 `4bc2fd7`):

1. **Store 修复** `workzone/mimic3/fix_pleth_scale.py`:对
   `diag_out/group_high.txt` 里的 2,231 个 entity 把 `PLETH40.npy` ×1023/4095,
   统一到 NU [0, 1]。幂等(meta.json 写 `pleth_scale_fix` 标记 + 缩放前校验
   vmax>2)、原子写、可逆(精确因子记录在 meta,undo = 除回去)。
   运行结果:2,231/2,231 FIXED,零错误。
2. **管线免疫** `stage3_extract_waveforms.py`:NU 单位通道在读取时按
   `gain/(2^res − 1)` 归一 ADC 满量程 —— 8/10-bit 因子恒为 1(行为不变),
   12-bit 自动落回 [0, 1],重抽取不会再引入分裂。

**修复后验证**(全店重扫,`diag_out/mimic3_amp_scan_postfix.csv`):
log(std) 单峰;5,619 条记录 vmax>2 计数为 0,最大值 1.115(重采样振铃);
mean p50 = 0.458,IQR [0.442, 0.476]。

## 影响与遗留

- 此前**任何**在这份 store 上训练/评估的模型,其 PLETH 输入都混着两种相差
  4× 的尺度。这解释了 MIMIC-III 跨数据集迁移 eff_dim 4.88 反而低于被破坏
  对照(9.31)的反常结果:encoder 面对双尺度输入,表示塌缩是必然的。
- **遗留 1:** dream 副本(`dream:/projects/xhu40-cdsfm/physio_data/mimic3`)
  还是旧量程,需对这 2,231 个 entity 重推 `PLETH40.npy` + `meta.json`。
- **遗留 2:** 训练侧的 MIMIC-III 波形归一化参数(global_affine)是在旧数据
  上估的,须重估;涉及 MIMIC-III 的已有实验结论建议复跑。

## 产物索引

| 文件 | 内容 |
|---|---|
| `workzone/mimic3/diag_amp_scan.py` | per-record 幅度扫描(任意 store 可复用) |
| `workzone/mimic3/diag_headers*.py` | 源 header gain/res 对比 |
| `workzone/mimic3/diag_mixed_scan.py` | entity 内混合量程检测 |
| `workzone/mimic3/fix_pleth_scale.py` | store 修复(幂等) |
| `workzone/mimic3/diag_out/group_{low,high}.txt` | 两组 entity 名单(3,388 / 2,231) |
| `workzone/mimic3/diag_out/mimic3_amp_scan{,_postfix}.csv` | 修复前/后全店扫描(后者仅在 bedanalysis) |
