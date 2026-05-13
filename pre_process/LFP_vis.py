import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.time_frequency import tfr_array_morlet
from scipy.signal import welch
import matplotlib
matplotlib.use('TkAgg')

# ==============================
# 【设置中文显示（可选）】
# ==============================
plt.rcParams["font.family"] = ["Microsoft Yahei"]
plt.rcParams["axes.unicode_minus"] = False

# ==============================
# 1. 读取/准备数据
# 你只需要确保这两个变量正确
# ==============================
# 方式A：如果你有MNE对象（推荐，直接用）
# raw_original = 你的原始MNE数据
# raw_processed = 你的处理后MNE数据
DATA_lfp = r"E:\crcns_hc3\ec012ec.11\ec012ec.189\ec012ec.11.npy"
DATA_lfp_processed = r"E:\crcns_hc3\ec012ec.11\ec012ec.189\ec012ec.189_processed_lfp.npy"
lfp_origin_data = np.load(DATA_lfp)
sfreq_ori = 1250
lfp_processed_data = np.load(DATA_lfp_processed)
sfreq_pro = 625

n_channels_ori = lfp_origin_data.shape[0]
ch_names_ori = [f"ch{i + 1}" for i in range(n_channels_ori)]  # 通道名：ch1,ch2...
ch_types_ori = ["eeg"] * n_channels_ori  # LFP 统一用 eeg 类型
info_ori = mne.create_info(ch_names=ch_names_ori, sfreq=sfreq_ori, ch_types=ch_types_ori)

n_channels_pro = lfp_processed_data.shape[0]
ch_names_pro = [f"ch{i + 1}" for i in range(n_channels_pro)]  # 通道名：ch1,ch2...
ch_types_pro = ["eeg"] * n_channels_ori  # LFP 统一用 eeg 类型
info_pro = mne.create_info(ch_names=ch_names_pro, sfreq=sfreq_pro, ch_types=ch_types_pro)

# 方式B：如果你只有numpy数据（用这个）
raw_original = mne.io.RawArray(lfp_origin_data, info_ori)
raw_processed = mne.io.RawArray(lfp_processed_data, info_pro)

sfreq = int(raw_original.info['sfreq'])  # 采样率

# ==============================
# 2. 提取数据（取第1个通道画图）
# ==============================
orig_data = raw_original.get_data()[0]  # 原始
proc_data = raw_processed.get_data()[0] # 处理后
times = np.arange(len(orig_data)) / sfreq

# 为了看图清晰，只取前 5 秒（可改）
plot_sec = 5
plot_n_samples = int(plot_sec * sfreq)

orig_plot = orig_data[:plot_n_samples]
proc_plot = proc_data[:plot_n_samples]
times_plot = times[:plot_n_samples]

# ==============================
# 3. 创建大图：3行2列对比
# ==============================
fig, axes = plt.subplots(3, 2, figsize=(14, 10))

# ==============================
# 第一行：时域波形对比
# ==============================
axes[0,0].plot(times_plot, orig_plot, color='#1f77b4', linewidth=0.8)
axes[0,0].set_title('原始 LFP 时域波形', fontsize=12)
axes[0,0].set_ylabel('幅值 (V)', fontsize=10)
axes[0,0].grid(alpha=0.3)

axes[0,1].plot(times_plot, proc_plot, color='#ff4b5c', linewidth=0.8)
axes[0,1].set_title('处理后 LFP 时域波形', fontsize=12)
axes[0,1].grid(alpha=0.3)

# ==============================
# 第二行：PSD 功率谱密度对比
# ==============================
f_orig, pxx_orig = welch(orig_data, sfreq, nperseg=sfreq*2)
f_proc, pxx_proc = welch(proc_data, sfreq, nperseg=sfreq*2)

axes[1,0].plot(f_orig, 10*np.log10(pxx_orig), color='#1f77b4')
axes[1,0].set_title('原始 PSD', fontsize=12)
axes[1,0].set_ylabel('功率 (dB/Hz)', fontsize=10)
axes[1,0].set_xlim(0, 100)
axes[1,0].grid(alpha=0.3)

axes[1,1].plot(f_proc, 10*np.log10(pxx_proc), color='#ff4b5c')
axes[1,1].set_title('处理后 PSD', fontsize=12)
axes[1,1].set_xlim(0, 100)
axes[1,1].grid(alpha=0.3)

# ==============================
# 第三行：时频图对比 (小波/STFT)
# ==============================
freqs=np.arange(1, 80, 2)
# 原始时频图
orig_s = orig_plot[None, None, :]
tfr_o = tfr_array_morlet(orig_s, sfreq=sfreq, freqs=freqs, n_cycles=3, output='power')[0,0]

# 处理后时频图
proc_s = proc_plot[None, None, :]
tfr_p = tfr_array_morlet(proc_s, sfreq=sfreq, freqs=freqs, n_cycles=3, output='power')[0,0]

axes[2,0].imshow(tfr_o, aspect='auto', origin='lower',
                 extent=[times_plot.min(), times_plot.max(), freqs.min(), freqs.max()], cmap='jet')
axes[2,0].set_title('原始 时频图')
axes[2,0].set_ylabel('Freq (Hz)')

axes[2,1].imshow(tfr_p, aspect='auto', origin='lower',
                 extent=[times_plot.min(), times_plot.max(), freqs.min(), freqs.max()], cmap='jet')
axes[2,1].set_title('处理后 时频图')
axes[2,1].set_xlabel('Time (s)')

# ==============================
# 统一布局
# ==============================
plt.tight_layout()
plt.show()