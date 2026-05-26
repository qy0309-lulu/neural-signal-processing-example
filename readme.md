本项目整理了笔者在入门学习神经信号的记录和思考，包括详细的代码和可视化解释，希望能帮助到有需要的人。
PS: 如果内容有误，请大家帮忙指正，欢迎交流！

this project represents a systematical pipeline of neural signal pre-processing and the process of exploratory analysis
including lfp signals and spike signals, using the public dataset crcns hc-3 (Hippocampus)

And you can get the dataset from https://crcns.org/data-sets/hc/hc-3

this project will follow a union structure: brief explanation + visualization
以jupyter notebook呈现， and all the code will be attached）

01_metadata_analysis/
    - overview: hc-3数据集包含大鼠海马及内嗅皮层7736个神经元的神经记录，其中pyramidal neuron 6100个，interneuron 1132个，涵盖不同脑区(EC/CA1/CA3/DG)。
      整个数据集共有442个session，14种不同的任务，每个session的数据单独储存在一个.tar.gz文件中
      附有元数据表格，可以查询细胞种类、发放率、行为任务等信息。
    - 核心文件信息: .res.N 第N个电极的峰电位时间戳 / .spk.N 峰电位波形 / .fet.N 排序所用特征值 / 
       .clu.N 每个spike的聚类编号 / .eeg 低频LFP信号 / .whl 大鼠位置坐标 / .xml Neuroscope配置（其中包含了采样率、电极数等信息）
    - spike sorting: 采用KlustaKwick + Kluster 方法
        #特别注意# 此过程是对同一个topdir下的所有session单通道统一进行的，
        即同一个topdir下，不同session中同一个通道，cluster_id 和 cell_id是一一对应的，channel_id, cluster_id, topdir 即可唯一确定一个细胞。
        理解这一点可帮助后续single neuron signal extraction 
    - 数据可分为三个级别：
      - session级别：session长度（数据采集的持续时间）；行为任务类型；动物编号；electrode数量
      - electrode级别：每个electrode的channel数量；所在脑区；
      - neuron级别：细胞类型、cluster_id、firing rate
    - 其他具体信息可查阅官方网站数据集的介绍

02_lfp_preprocessing/
a typical process:
   1. bad channel detect
      通道质量评估 channle quality evaluation 通常涉及：
      - mean、std
      - PSD：high-frequency ratio（异常高就是噪声太大）; line noise 
      - 1/f 检查β 1.0-1.7 
      - hippocampus特别 检查 θ 峰 （自由活动时有清晰的theta峰）
   2. 基线漂移（非必需）
      其本质是低频信号，也包含真实生理信息。根据研究对象、分析频段等具体决定是否去除
      - 但常用4阶butterworth high-pass filter， 去除<0.5Hz
   3. 去除工频噪声（50/60 Hz）
      norch filter
   4. band-pass filter (theta/gamma/ripple/beta) 
   通常提取经典频段，其频段范围即对应任务如下：
     | Frequency Band | Typical Range | Typical Functional Associations                            |
     |----------------|---------------|------------------------------------------------------------|
     | Delta          | 1 – 4 Hz      | Sleep, immobility, slow-wave activity                      |
     | Theta          | 4 – 12 Hz     | Navigation, locomotion, spatial memory, hippocampal coding |
     | Beta           | 13 – 30 Hz    | Sensorimotor coordination, long-range communication        |
     | Gamma          | 30 – 100 Hz   | local neural computation, inforamtion integration          |
     | Ripple         | 100 – 250 Hz  | Replay, memory consolidation, offline processing           |
   注意：滤波边缘效应会长生原本没有的频段，所以要先在PSD图中确认此频段存在，再进行滤波和分析
   5. artifact detection -> Z-Score / MAD threshold / RMS threshold
      - line noise / bad channel / movement_artifact / muscle artifact etc. 一切非自然的神经活动都是arifact
      - 这里去除的主要是movement导致电极、电缆晃动----大振幅低频波动；muscle artifact（EMG）:咀嚼/肌肉活动----高频成分/会污染gamma、high-freq band
      - 常用方法：CAR(channel average reference): 经典方法，去除通道共有的噪声; Threshold rejection(RMS/Z-Score) 
   6. downsampling 
   根据后续使用需求决定是否进行，hc-3数据集中LFP已下采样至1.25KHz(原本20KHz)

03_spike_preprocessing/
    这里主要介绍通过 计算spike firing rate 进行分析：
    - FR = N/T,(Hz) 单位时间内神经元发放的锋电位数量。其计算过程：
       1. 选取合适的bin（10/50/100/200 ms）
         - bin_size: 观察神经活动的窗口，类似分辨率
         - 不同脑区、不同task 会选择不同的bin
       2. 计算每个bin内的spike count 
       3. spike count / bin （瞬时fr）
          群体fr: 对每个神经元的fr取平均
                可以根据通道/细胞种类计算population fr，e.g. hc-3数据集中将细胞分成p（pyramidal neuron）/i（interneuron）两种
    - normalization （非必需）
        - 常用方法：Z-Score ; Min-max ; log transform （per neuron not per time-bin normalization）
        - decoder任务通常normalization，尤其是linear model / PCA / NN
        - population level dynamics analysis：研究神经元群体活动的时间变化，寻找latent feature，需要normalization
    - 用于筛选
        - 通常会对fr太低的neuron进行过滤，因为所含信息量太少；
        - 在本数据集提供了 每个细胞的 eDist/RefracRatio/RefracVio 和 criteria 以供筛选高质量细胞
    -其他分析方式
        - 滑动窗口计算fr
        - 除了hard binning还可以用spike density function 
            用Gaussian kernal卷积spike信号 0/1 序列（更好保留时间连续性，但也有可能制造虚假的dynamic信息）

04_behavior_preprocessing/
    This datasets involves 14 different behavioral tasks，包括运动型的也有静止的 (the detail can be found in the description file)
    1. " Why smooth? "
     - 行为数据的采集：video tracking / LED tracking / marker tracking
     - 采集过程中会有jitter、高频噪声，计算速度、加速度进行微分尤其放大噪声
     - smoothing方法：
        - moving average：window = N * frames
        - Gaussian smoothing：Gaussian kernal--sigma取值
        - * Kalman smoothing / Savitzky-Golay Filter 
     - smoothing的本质就是low pass filter，去除high-freq noise，也会损失快速动作变化信息，"使用需谨慎"
    2. 常用的统计分析方法：（basic）
     - trajectory visualization -> figs
     - occupancy map -> figs -> place field analysis
     - speed distribution -> figs


05_small_insights/
    "several uesful insights FYR"
    neural signal 中时序信息 IS IMPORTANT
    单个神经元的活动非常多变，寻找一群神经元之间的共性变化规律更有意义
    不同的subject、同一个subject的不同session也会有很大差别



06_exploratory_analysis/
    单个神经元：
      1. spike raster plot analysis 
      2. ISI inter-spike interval: refractory period 
      3. autocorrelation 
      4. most cells in Hippocampus is sparse firing (CA1/CA3/DG), about 0.1-3 Hz, 
         interneuron is higher, with fr at 10-50 Hz .
    群体神经元层面：
      1.
    LFP分析
      1. psd图
        - theta dominance （在相应频段能量最高）
        - oscillatory structure
      2. band power over time


