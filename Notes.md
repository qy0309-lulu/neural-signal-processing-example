# Neural Signal Preprocessing & Analysis Pipeline

## 项目概述

This project implements a systematic pipeline for neural signal preprocessing and exploratory analysis, including LFP signals and spike signals, using the public dataset **CRCNS HC-3 (Hippocampus)**.

Dataset access: [https://crcns.org/data-sets/hc/hc-3](https://crcns.org/data-sets/hc/hc-3)

The project follows a unified structure: **brief explanation + visualization**, presented as Jupyter Notebooks (all code attached).

---

## 01_metadata_analysis/

### 数据集overview

- HC-3 数据集包含大鼠海马及内嗅皮层 7736 个神经元 的神经记录，其中：
  - 锥体神经元（pyramidal neuron）：6100 个
  - 中间神经元（interneuron）：1132 个
  - 覆盖脑区：EC / CA1 / CA3 / DG

- 全数据集包含：
  - 442 个 session**
  - 14 种不同行为任务
  - 每个 session 的数据单独存储为一个 .tar.gz 文件
  - 提供元数据表格，可以对应查询细胞种类、发放率、行为任务等信息

### 数据集中核心文件说明

| `.res.N` | 第 N 个电极的峰电位时间戳 |
| `.spk.N` | 峰电位波形 |
| `.fet.N` | 排序所用特征值 |
| `.clu.N` | 每个 spike 的聚类编号 |
| `.eeg` | 低频 LFP 信号 |
| `.whl` | 大鼠位置坐标 |
| `.xml` | Neuroscope 配置（采样率、电极数等信息） |

### Spike Sorting

- 采用 **KlustaKwick + Kluster** 方法
- **特别注意**：该过程是对同一个 topdir 下的所有 session，单通道统一进行的。
  - 即同一个 `topdir` 下，不同 session 中同一个通道的 `cluster_id` 和 `cell_id` 是一一对应的。
  - 由 `(channel_id, cluster_id, topdir)` 即可唯一确定一个细胞。
  - 理解这一点对后续神经元级信号提取（cluster signal extraction）非常重要。

### 数据三级结构

- Session 级别** | session 长度（采集持续时间）、行为任务类型、动物编号、电极数量 |
- Electrode 级别** | 每个电极的通道数量、所在脑区 |
- Neuron 级别** | 细胞类型、`cluster_id`、发放率（firing rate） |

其他详细信息可查阅官方网站数据集介绍。

---

## 02_lfp_preprocessing/

典型处理流程：

### 1. Bad Channel 检测

通道质量评估通常包括：

- 均值（mean）、标准差（std）
- PSD 分析：
  - **高频比例**：异常高 → 噪声过大
  - **工频噪声（line noise）**
  - **1/f 特性**：检查 β 在 1.0–1.7 范围
- **海马体特殊检查**：自由活动时应出现清晰的 **θ 峰**（大鼠通常在 6–10 Hz）

### 2. 基线漂移（非必需）

- 本质是低频信号，也可能包含真实生理信息
- **是否去除**：取决于研究对象和分析频段
- [修正] 常用方法：**4阶 Butterworth 高通滤波**，去除 < 0.1 Hz 成分（有时更低至 0.01 Hz）。注意 0.5 Hz 会去除部分真实神经信号。

### 3. 去除工频噪声（50/60 Hz）

- [修正] 使用 **陷波滤波器（notch filter）**

### 4. 带通滤波（theta / gamma / ripple / beta）

常用经典频段及其功能关联：

| 频段 | 典型范围 | 功能关联 |
| --- | --- | --- |
| Delta | 1 – 4 Hz | 睡眠、静止、慢波活动 |
| Theta | 4 – 12 Hz | 导航、运动、空间记忆、海马编码 |
| Beta | 13 – 30 Hz | 感觉运动协调、长程通信 |
| Gamma | 30 – 100 Hz | 局部神经计算、信息整合 |
| Ripple | 100 – 250 Hz | 回放、记忆巩固、离线处理 |

> 注意：滤波边缘效应可能产生原本不存在的频段成分。所以应先通过 PSD 图确认目标频段存在，再进行滤波和分析。

### 5. 伪迹检测（Artifact Detection）

- 方法：Z-Score / MAD 阈值 / RMS 阈值
- 伪迹类型：
  - 工频噪声 / 坏通道
  - 运动伪迹（电极/电缆晃动）→ 大振幅低频波动
  - 肌肉伪迹（EMG，如咀嚼）→ 高频成分，会污染 gamma 等高频带
- 常用处理方法：
  - **阈值剔除（RMS / Z-Score）**

- CAR（共平均参考）：经典重参考方法，去除通道间共有噪声

### 6. 降采样（Downsampling）

- 根据后续使用需求决定
- HC-3 中的 LFP 已从原始 20 kHz 降采样至 **1.25 kHz**

---

## 03_spike_preprocessing/

主要介绍通过 计算 spike 发放率（firing rate, FR） 进行分析：

### 发放率计算

FR = N/T,(Hz) 单位时间内神经元发放的锋电位数量

计算步骤：

1. 选择合适的 bin 大小（10 / 50 / 100 / 200 ms）
   - bin 大小：观察神经活动的时间分辨率
   - 不同脑区、不同任务会选择不同 bin
2. 计算每个 bin 内的 spike 数量
3. 瞬时 FR = spike count / bin 
4. 群体 FR：对每个神经元的 FR 取平均（另一种常见方法是先计算群体 spike count，再除以 bin 长度，两种结果略有不同）
5. 还可按通道或细胞类型计算（如 HC-3 中将细胞分为 P / I 两类）

### 归一化（非必需）

- 常用方法：
  - Z-Score
  - Min-Max
  - Log 变换
  - （注意：按神经元归一化，而非按时间 bin）

- 适用场景：
  - Decoder 任务（尤其线性模型 / PCA / NN）
  - 群体动态分析（population-level dynamics analysis）：研究神经元群体活动的时间变化，寻找潜在特征（latent features）

### 用于筛选

- 通常过滤掉发放率过低的神经元（信息量太少）
- 本数据集提供了筛选高质量细胞的 `eDist` `RefracRatio` `RefracVio` 筛选标准

### 其他分析方式

- 滑动窗口计算 FR
- Spike Density Function（替代 hard binning）
  - 使用 Gaussian kernel 对 0/1 spike 序列进行卷积
  - 优点：更好地保留时间连续性
  - 缺点：过宽的 kernel 可能会过于平滑，模糊真实的时间动态，丢失快速变化的信息

---

## 04_behavior_preprocessing/

本数据集包含 14 种不同行为任务，包括运动型和静止型（详见描述文件）。

### 1. 为什么要平滑（Why smooth?）

- 行为数据采集方式：
  - Video tracking
  - LED tracking
  - Marker tracking

- 采集过程中存在 jitter 和 高频噪声，而计算速度、加速度（微分操作）会放大噪声

#### 常用平滑方法

| 方法 | 说明 |
|-----| -----| 
| Moving average | window = N × frames |
| Gaussian smoothing | Gaussian kernel，需设定 sigma |
| Kalman smoothing / Savitzky–Golay filter | 更高级的选项 |

>平滑的主要作用是**减少高频噪声**，但不同平滑方法的频率响应特性不同（例如移动平均是低通滤波，而 Savitzky–Golay 滤波可以在降噪的同时保留部分高频特征）。平滑会损失快速动作变化信息，**使用需谨慎。**

### 2. 常用统计分析方法（基础）

- **轨迹可视化（trajectory visualization）**
- **occupancy 图（occupancy map）** → 用于 place field 分析
- **速度分布（speed distribution）**

---

## 05_small_insights/

> "several useful insights FYR"

- 神经信号中的**时序信息非常重要**
- 单个神经元的活动非常多变，**寻找一群神经元之间的共性变化规律**更有意义
- 不同 subject、同一 subject 的不同 session 之间也可能存在很大差异