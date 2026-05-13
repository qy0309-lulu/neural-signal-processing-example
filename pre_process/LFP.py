import os

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from tqdm import tqdm
import mne


def detect_bad_channels(raw, zscore_threshold=5.0):
    """简单坏通道检测（基于方差或幅度）"""
    # stds = np.std(data, axis=0)
    # bad_mask = stds > np.median(stds) * threshold
    # return bad_mask, stds
    data = raw.get_data()
    stds = np.std(data, axis=1)
    z = (stds - np.mean(stds)) / np.std(stds)
    bad_idx = np.where(np.abs(z) > zscore_threshold)[0]
    bad_chs = [raw.ch_names[i] for i in bad_idx]

    return bad_chs

def zscore_normalize(raw, per_channel=True):
    data = raw.get_data()
    if per_channel:
        # 按通道归一化：每个通道减去自身均值，除以自身标准差
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True)
    else:
        # 全局归一化：所有数据用同一个均值和标准差
        mean = np.mean(data)
        std = np.std(data)
    data_norm = (data - mean) / (std + 1e-10)  # 加小量防止除0
    raw_norm = raw.copy()
    raw_norm._data = data_norm
    return raw_norm

def lfp_process(data: np.ndarray, sfreq: float = 1250.0, l_freq: float = 4.0, h_freq: float = 200.0, use_CAR: bool = True, verbose: bool = True):

    total_steps = 6 + (1 if use_CAR else 0)
    pbar = tqdm(total_steps, desc="LFP Processing", unit="step")

    # 1. 转成 MNE Raw 对象（这就是最终结果）
    n_channels = data.shape[0]
    ch_names = [f"ch{i + 1}" for i in range(n_channels)]  # 通道名：ch1,ch2...
    ch_types = ["eeg"] * n_channels  # LFP 统一用 eeg 类型
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)
    pbar.set_postfix(当前步骤="转MNE格式", refresh=True)
    pbar.update(1)


    # 2. 坏通道检测（插值或剔除）
    # 方法一：直接剔除坏通道
    bad_channels = detect_bad_channels(raw, zscore_threshold=5.0)
    print("Detected bad channels:", bad_channels)
    raw.info['bads'] = bad_channels
    raw.drop_channels(bad_channels)
    # # 方法2：插值坏通道（推荐）
    # raw.interpolate_bads(reset_bads=True)
    pbar.set_postfix(当前步骤="坏通道处理", refresh=True)
    pbar.update(1)

    # 3. 陷波
    # 方法1：使用 mne 的 notch_filter
    raw.notch_filter(freqs=50, picks='all')
    pbar.set_postfix(当前步骤="50Hz陷波去噪", refresh=True)
    pbar.update(1)

    # 4. 带通滤波（1-200 Hz 或 4-200 Hz）
    raw.filter(l_freq=l_freq, h_freq=h_freq, picks='all', verbose=verbose)
    pbar.set_postfix(当前步骤="bandpass filter", refresh=True)
    pbar.update(1)

    if use_CAR:
        # 5. Common Average Reference (CAR) 平均重参考
        # --------------------------
        # 计算所有通道的平均信号，然后每个通道减去该平均信号
        raw.set_eeg_reference(ref_channels='average', verbose=verbose)
        pbar.set_postfix(当前步骤="CAR", refresh=True)
        pbar.update(1)

    # 6. Z-score 归一化（按通道或全局）
    # --------------------------
    raw_normalized = zscore_normalize(raw, per_channel=True)
    pbar.set_postfix(当前步骤="Z-Score Normalization", refresh=True)
    pbar.update(1)

    # 7. 下采样（例如降到 625 Hz）
    # --------------------------
    target_sfreq = 625
    raw_downsampled = raw_normalized.copy().resample(sfreq=target_sfreq, verbose=verbose)
    pbar.set_postfix(当前步骤="down sampling", refresh=True)
    pbar.update(1)

    pbar.close()
    print("===== 处理完成 =====")
    print("原始采样率:", sfreq)
    print("下采样后采样率:", raw_downsampled.info['sfreq'])
    print("最终数据形状:", raw_downsampled.get_data().shape)

    return raw_downsampled

if __name__ == "__main__":
    # BASE_PATH = "E:/crcns_hc3"  # ← 修改为你的数据集路径
    DATA_path_lfp = r"E:\crcns_hc3\ec012ec.11\ec012ec.189\ec012ec.11.npy"
    SAVE_PATH = "E:/crcns_hc3/ec012ec.11/ec012ec.189"

    data_lfp = np.load(DATA_path_lfp)
    # lfp_data_name = 'ec012ec.11/ec012ec.189_lfp.npy'
    # lfp_save_path = os.path.join(SAVE_PATH, lfp_data_name)
    # if lfp_save_path and not os.path.exists(lfp_save_path):
    #     os.makedirs(lfp_save_path)
    # np.save(SAVE_PATH, data_lfp)
    # print(f"lfp数据保存到{lfp_save_path}")

    raw_processd = lfp_process(data_lfp, sfreq=1250.0, l_freq=4.0, h_freq=200.0, use_CAR=False,verbose=False)

    processed_data = raw_processd.get_data()
    session_name = 'ec012ec.189_processed_lfp_2.npy'
    # _2: no CAR
    save_path = os.path.join(SAVE_PATH, session_name)
    save_dir = SAVE_PATH

    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(save_path, processed_data)
    print(f"数据保存到{SAVE_PATH}")
