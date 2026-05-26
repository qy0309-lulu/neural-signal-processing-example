import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import matplotlib
matplotlib.use('Agg')

from hc3_utils import compute_morlet_wavelet, compute_stft

# ======================
# 1. 基础设置（只改这里）
# ======================
lfp_path = r"E:\crcns_hc3\ec012ec.11\ec012ec.189\ec012ec.189_processed_lfp.npy" # 改成你的npy路径
fs = 1250                     # 采样率

# 标准神经振荡频段
BANDS = {
    'theta': [4, 12],
    'gamma': [30, 80],
    'ripples': [80, 200]
}

# 绘图风格
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ======================
# 2. 加载LFP数据
# ======================
print("正在加载LFP数据...")
lfp = np.load(lfp_path)

# # 多通道取第一个通道
# if lfp.ndim > 1:
#     lfp = lfp[0]
#
# # 取前60秒（可改）
# duration = 60
# lfp = lfp[:int(fs * duration)]
# print(f"LFP长度: {len(lfp)/fs:.1f} 秒")

# ======================
# 3. PSD 功率谱
# ======================
def compute_psd(signal, fs, nperseg=2048):
    f, pxx = sig.welch(signal, fs=fs, nperseg=nperseg)
    return f, pxx

# 三图对比
# ======================
# 4. 三图对比：PSD | Morlet | STFT
# ======================
for name, (f_low, f_high) in BANDS.items():
    print(f"\n正在分析 → {name} ({f_low}-{f_high} Hz)")

    # 计算数据
    f_psd, pxx = compute_psd(lfp, fs)
    f_wav, t_wav, p_wav = compute_morlet_wavelet(lfp, fs, f_low, f_high)
    f_stft, t_stft, p_stft = compute_stft(lfp, fs)

    # 开始绘图：1行3列
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    # ---------- 子图1：PSD ----------
    ax = axes[0]
    ax.plot(f_psd, 10 * np.log10(pxx), color='#2a5a9b', linewidth=1)
    ax.set_title(f'{name} - PSD 功率谱')
    ax.set_xlabel('频率 (Hz)')
    ax.set_ylabel('功率 (dB/Hz)')
    ax.set_xlim(0, f_high + 30)
    ax.grid(alpha=0.3)

    # ---------- 子图2：Morlet 小波时频图 ----------
    ax = axes[1]
    vmin, vmax = np.percentile(p_wav, [5, 95])
    im = ax.imshow(p_wav,
                   aspect='auto', origin='lower',
                   extent=[t_wav.min(), t_wav.max(), f_wav.min(), f_wav.max()],
                   cmap='viridis', vmin=vmin, vmax=vmax)
    ax.set_title(f'{name} - Morlet 小波时频图')
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('频率 (Hz)')
    ax.set_ylim(f_low, f_high)
    plt.colorbar(im, ax=ax, shrink=0.8)

    # ---------- 子图3：STFT 时频图 ----------
    ax = axes[2]
    vmin, vmax = np.percentile(p_stft, [5, 95])
    im = ax.imshow(p_stft,
                   aspect='auto', origin='lower',
                   extent=[t_stft.min(), t_stft.max(), f_stft.min(), f_stft.max()],
                   cmap='viridis', vmin=vmin, vmax=vmax)
    ax.set_title(f'{name} - STFT 时频图')
    ax.set_xlabel('时间 (s)')
    ax.set_ylim(f_low, f_high)
    plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig(f'figures/{name}_2.png')
    plt.close()

print("\n✅ 全部分析完成：Theta / Gamma / Ripples")

import numpy as np
import scipy.signal as sig
from scipy.signal import hilbert


def bandpass_filter(signal, fs, low, high, order=4):
    """带通滤波，提取单一频段"""
    nyq = 0.5 * fs
    b, a = sig.butter(order, [low / nyq, high / nyq], btype='band')
    return sig.filtfilt(b, a, signal)


def compute_hilbert_features(filtered_signal, fs):
    """
    对滤波后的信号做Hilbert变换，返回三大核心特征
    """
    # 构造解析信号
    analytic_signal = hilbert(filtered_signal)

    # 1. 瞬时相位（-π ~ π）
    phase = np.angle(analytic_signal)

    # 2. 瞬时振幅包络
    amplitude = np.abs(analytic_signal)

    # 3. 瞬时频率（相位求导，单位Hz）
    inst_freq = np.diff(np.unwrap(phase)) / (2 * np.pi) * fs
    # 对齐长度（和原信号保持一致）
    inst_freq = np.append(inst_freq, inst_freq[-1])

    return phase, amplitude, inst_freq



# 1. 先滤波到Theta频段（4-12Hz）
theta_signal = bandpass_filter(lfp, fs, 4, 12)

# 2. Hilbert变换提取特征
theta_phase, theta_amp, theta_freq = compute_hilbert_features(theta_signal, fs)


