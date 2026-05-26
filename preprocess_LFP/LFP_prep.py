import os
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from tqdm import tqdm
import mne
from scipy.signal import detrend


def detect_bad_channels(raw, zscore_threshold=5.0):
    """简单坏通道检测（基于方差或幅度）"""
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


def remove_baseline(raw, baseline=(0, 10), mode='mean'):
    """
    去除基线漂移
    :param raw: mne.Raw 对象
    :param baseline: 基线时段 (start, end)，单位：秒
    :param mode: 基线校正模式 - 'mean'(均值) / 'median'(中位数) / 'detrend'(线性去趋势)
    :return: 去基线后的 raw 对象
    """
    raw_baseline = raw.copy()
    data = raw_baseline.get_data()
    sfreq = raw_baseline.info['sfreq']

    if mode == 'detrend':
        # 线性去趋势（直接去除基线漂移）
        data = detrend(data, axis=1)
    else:
        # 基于基线时段的均值/中位数校正
        baseline_start = int(baseline[0] * sfreq)
        baseline_end = int(baseline[1] * sfreq)
        if baseline_end > data.shape[1]:
            baseline_end = data.shape[1]

        # 计算基线值
        if mode == 'mean':
            baseline_vals = np.mean(data[:, baseline_start:baseline_end], axis=1, keepdims=True)
        elif mode == 'median':
            baseline_vals = np.median(data[:, baseline_start:baseline_end], axis=1, keepdims=True)
        else:
            raise ValueError(f"不支持的基线模式: {mode}")

        # 去除基线
        data = data - baseline_vals

    raw_baseline._data = data
    return raw_baseline


def remove_artifacts(raw, method='zscore', threshold=5.0, replace='interpolate'):
    """
    去除伪迹（异常值）
    :param raw: mne.Raw 对象
    :param method: 伪迹检测方法 - 'zscore'(Z分数) / 'mad'(中位数绝对偏差)
    :param threshold: 异常值阈值
    :param replace: 伪迹替换方式 - 'interpolate'(插值) / 'nan'(置NaN) / 'zero'(置0)
    :return: 去伪迹后的 raw 对象
    """
    raw_artifact = raw.copy()
    data = raw_artifact.get_data()
    n_channels, n_times = data.shape

    # 1. 检测伪迹位置
    if method == 'zscore':
        # Z-Score
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True)
        z_scores = np.abs((data - mean) / (std + 1e-10))
        artifact_mask = z_scores > threshold
    elif method == 'mad':
        # 中位数绝对偏差法（更鲁棒）
        median = np.median(data, axis=1, keepdims=True)
        mad = np.median(np.abs(data - median), axis=1, keepdims=True)
        modified_z = 0.6745 * (data - median) / (mad + 1e-10)
        artifact_mask = np.abs(modified_z) > threshold
    else:
        raise ValueError(f"未编码此伪迹检测方法: {method}")

    # 2. 处理伪迹
    if replace == 'interpolate':
        # 线性插值替换伪迹
        for ch in range(n_channels):
            artifact_times = np.where(artifact_mask[ch])[0]
            if len(artifact_times) == 0:
                continue
            # 非伪迹位置
            clean_times = np.where(~artifact_mask[ch])[0]
            if len(clean_times) < 2:
                # 有效点太少，直接置0
                data[ch, artifact_times] = 0
                continue
            # 插值
            data[ch, artifact_times] = np.interp(artifact_times, clean_times, data[ch, clean_times])
    elif replace == 'nan':
        # 置为NaN（后续可填充）
        data[artifact_mask] = np.nan
    elif replace == 'zero':
        # 置为0
        data[artifact_mask] = 0
    else:
        raise ValueError(f"不支持的替换方式: {replace}")

    raw_artifact._data = data
    return raw_artifact


def lfp_process(data: np.ndarray, target_sfreq: float = 39.0625, sfreq: float = 1250.0, l_freq: float = 4.0, h_freq: float = 200.0,
                use_CAR: bool = True, remove_baseline_drift: bool = True, remove_artifacts_flag: bool = True,
                verbose: bool = False):
    """
    增强版LFP处理流程（新增去基线、伪迹去除）
    """
    # 调整总步骤数
    total_steps = 7 + (1 if use_CAR else 0)
    pbar = tqdm(total_steps, desc="LFP Processing", unit="step")

    # 1. 转成 MNE Raw 对象
    n_channels = data.shape[0]
    ch_names = [f"ch{i + 1}" for i in range(n_channels)]
    ch_types = ["eeg"] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)
    pbar.set_postfix(当前步骤="转MNE格式", refresh=True)
    pbar.update(1)

    # 2. 坏通道检测（插值或剔除）
    bad_channels = detect_bad_channels(raw, zscore_threshold=5.0)
    print("Detected bad channels:", bad_channels)
    raw.info['bads'] = bad_channels
    raw.drop_channels(bad_channels)
    pbar.set_postfix(当前步骤="坏通道处理", refresh=True)
    pbar.update(1)

    # 新增：3. 去除基线漂移
    if remove_baseline_drift:
        raw = remove_baseline(raw, baseline=(0, 10), mode='mean')  # 用前10秒做基线
        pbar.set_postfix(当前步骤="去除基线漂移", refresh=True)
        pbar.update(1)
    else:
        pbar.update(1)  # 保持进度条同步

    # 4. 陷波
    raw.notch_filter(freqs=60, picks='all')
    pbar.set_postfix(当前步骤="60Hz陷波去噪", refresh=True)
    pbar.update(1)

    # 5. 带通滤波
    raw.filter(l_freq=l_freq, h_freq=h_freq, picks='all', verbose=verbose)
    pbar.set_postfix(当前步骤="bandpass filter", refresh=True)
    pbar.update(1)

    # 6. 伪迹去除
    if remove_artifacts_flag:
        raw = remove_artifacts(raw, method='mad', threshold=5.0, replace='interpolate')
        pbar.set_postfix(当前步骤="伪迹去除", refresh=True)
        pbar.update(1)
    else:
        pbar.update(1)  # 保持进度条同步

    if use_CAR:
        # 7. Common Average Reference (CAR) 平均重参考
        raw.set_eeg_reference(ref_channels='average', verbose=verbose)
        pbar.set_postfix(当前步骤="CAR", refresh=True)
        pbar.update(1)


    # 8. 下采样
    target_sfreq = target_sfreq
    raw_downsampled = raw.copy().resample(sfreq=target_sfreq, verbose=verbose)
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
    # DATA_path_lfp = r"E:\crcns_hc3\ec012ec.11\ec012ec.188\ec012ec.188.npy"
    DATA_path_lfp = r"E:\crcns_hc3\ec013.28\ec013.410\ec013.410_LFP.npy"
    # load_hc_session.py中得到加载后的raw_lfp.npy
    SAVE_PATH = r"E:\crcns_hc3\ec013.28\ec013.410"

    # 加载数据
    data_lfp = np.load(DATA_path_lfp)

    # 增强版LFP处理（开启去基线和伪迹去除）
    raw_processd = lfp_process(
        data_lfp,
        sfreq=1250.0,
        target_sfreq = 625, #39.0625,
        l_freq=4.0,
        h_freq=200.0,
        use_CAR=True,
        remove_baseline_drift=True,  # 开启去基线
        remove_artifacts_flag=True,  # 开启伪迹去除
        verbose=False
    )

    # 保存处理后的数据
    processed_data = raw_processd.get_data()
    session_name = 'ec013.410_processed_lfp_625.npy'
    save_path = os.path.join(SAVE_PATH, session_name)
    save_dir = SAVE_PATH

    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(save_path, processed_data)
    print(f"数据保存到 {save_path}")