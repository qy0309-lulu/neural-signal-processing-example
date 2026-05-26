from pathlib import Path
from typing import Dict, Optional, List, Generator
import logging
from tqdm import tqdm
import numpy as np

from pre_process.load_one_session import load_hc3_session_full


# ====================== 你的原函数（保持不变）======================

def discover_all_sessions(base_path: str) -> List[Dict]:
    """
    自动发现 base_path 下所有 topdir 和对应的 session
    """
    base_path = Path(base_path)
    sessions = []

    # 遍历所有 top-level directory
    for topdir in base_path.iterdir():
        if not topdir.is_dir():
            continue

        # 查找该 topdir 下的所有 .tar.gz 文件
        tar_files = list(topdir.glob("*.tar.gz"))

        for tar_path in tar_files:
            # 提取 session name: ec012ec.189.tar.gz → ec012ec.189
            session_name = tar_path.stem.replace('.tar', '')

            sessions.append({
                'topdir': topdir.name,
                'session': session_name,
                'tar_path': tar_path
            })

    print(f"共发现 {len(sessions)} 个 session")
    return sessions


def load_all_sessions(
        base_path: str,
        load_spikes: bool = True,
        load_lfp: bool = False,  # LFP 数据很大，默认不加载
        load_position: bool = True,
        load_xml: bool = True,
        save_lfp: bool = True,
        electrodes: Optional[List[int]] = None,
        verbose: bool = False,
        max_sessions: Optional[int] = None
) -> Generator[Dict, None, None]:
    """
    批量加载所有 session 的生成器（推荐使用，避免内存爆炸）
    """
    sessions = discover_all_sessions(base_path)

    if max_sessions:
        sessions = sessions[:max_sessions]

    for info in tqdm(sessions, desc="加载 session"):
        try:
            data = load_hc3_session_full(
                base_path=base_path,
                topdir=info['topdir'],
                session=info['session'],
                electrodes=electrodes,
                load_spikes=load_spikes,
                load_lfp=load_lfp,
                load_position=load_position,
                load_xml=load_xml,
                save_lfp=save_lfp,
                verbose=verbose
            )
            data['tar_path'] = info['tar_path']  # 额外记录路径
            yield data

        except Exception as e:
            logging.error(f"加载 {info['topdir']}/{info['session']} 失败: {e}")
            continue


def bandpass_filter(data: np.ndarray, fs: float = 1250.0, lowcut=1.0, highcut=200.0, order=4):
    """零相位带通滤波 (推荐使用 filtfilt)"""
    nyq = fs / 2
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data, axis=0)

def notch_filter(data: np.ndarray, fs: float = 1250.0, freq=50.0, Q=30):
    """陷波滤波去除工频"""
    nyq = fs / 2
    b, a = signal.iirnotch(freq / nyq, Q)
    return signal.filtfilt(b, a, data, axis=0)

def common_average_reference(data: np.ndarray):
    """Common Average Reference (CAR)"""
    mean_across_channels = np.mean(data, axis=1, keepdims=True)
    return data - mean_across_channels

def detect_bad_channels(data: np.ndarray, threshold=5.0):
    """简单坏通道检测（基于方差或幅度）"""
    stds = np.std(data, axis=0)
    bad_mask = stds > np.median(stds) * threshold
    return bad_mask, stds


import numpy as np
import pywt
from scipy import signal
from scipy.fftpack import hilbert


# ------------------------------------------------------------------------------
# 1. STFT 短时傅里叶变换（时频分析，频率分辨率固定）
# ------------------------------------------------------------------------------
def compute_stft(sig, fs, nperseg=256, noverlap=128):
    """
    计算 STFT 时频图
    :param sig: 一维神经信号 (LFP/EEG)
    :param fs: 采样率
    :param nperseg: 每段长度
    :param noverlap: 重叠长度
    :return: f: 频率轴, t: 时间轴, Zxx: 时频功率
    """
    f, t, Zxx = signal.stft(sig, fs=fs, nperseg=nperseg, noverlap=noverlap)
    power = np.abs(Zxx) ** 2  # 取功率
    return f, t, power


# ------------------------------------------------------------------------------
# 2. Morlet 小波变换（时频分析，高频时间准、低频频率准，最适合神经信号）
# ------------------------------------------------------------------------------
def compute_morlet_wavelet(signal, fs, freq_min=1, freq_max=100, freq_step=2, n_cycles=5):
    """
    PyWavelets 版 Morlet 小波时频分析
    :param signal: 一维神经信号（LFP/EEG）
    :param fs: 采样率
    :param freq_min: 最低频率
    :param freq_max: 最高频率
    :param freq_step: 频率步长
    :param n_cycles: 小波周期数（推荐 3~7）
    :return: freqs: 频率轴, t: 时间轴, power: 时频功率(2D矩阵)
    """

    # === PyWavelets 核心：自动计算尺度，适配频率 ===
    # 生成需要的频率
    freqs = np.arange(freq_min, freq_max + 1, freq_step)

    # 关键：自己计算尺度（修复报错核心）
    # scales = pywt.scale2frequency('morl', freqs) ** -1 / fs
    scales = (fs * 2) / freqs

    # 'morl' = Morlet小波（神经信号最常用）
    cwt_matrix, freqs_calc = pywt.cwt(
        signal,
        scales=scales,  # 自动计算尺度
        wavelet='morl',  # Morlet小波
        sampling_period=1 / fs,  # 采样周期
        method='fft'  # 快速计算
    )

    # 时频功率（取模平方）
    power = np.abs(cwt_matrix) ** 2

    # 时间轴
    t = np.arange(len(signal)) / fs

    return freqs_calc, t, power

# ------------------------------------------------------------------------------
# 3. Hilbert 变换（提取瞬时振幅、相位、频率）
# ------------------------------------------------------------------------------
def compute_hilbert(signal):
    """
    Hilbert 变换，获取：
    1. 瞬时相位
    2. 瞬时振幅（包络）
    3. 瞬时频率
    """
    # # 暂时不用----去除NaN
    # valid = np.isfinite(signal)
    # if np.any(valid):
    #     x = np.interp(np.arange(len(signal)), np.where(valid)[0], signal[valid])
    # else:
    #     x = signal.copy()

    # Hilbert 变换
    analytic_signal = hilbert(signal)

    # 核心输出
    phase = np.angle(analytic_signal)  # 相位 (-π ~ π)
    amplitude = np.abs(analytic_signal)  # 振幅包络
    inst_freq = np.diff(np.unwrap(phase))  # 瞬时频率

    # 对齐长度
    inst_freq = np.append(inst_freq, inst_freq[-1])

    return phase, amplitude, inst_freq

# ====================== 使用示例 ======================
# if __name__ == '__main__':
#     BASE_PATH = "E:/crcns_hc3"  # ← 修改为你的数据集路径
#
#     # 方式1：生成器方式（推荐，节省内存）
#     print("开始批量加载...")
#     session_count = 0
#     total_units = 0
#
#     for data in load_all_sessions(
#             base_path=BASE_PATH,
#             load_lfp=True,  # 建议先不加载 LFP
#             verbose=True,
#             save_lfp=True,
#             max_sessions=1  # 设置数字可限制测试数量
#     ):
#         session_count += 1
#         n_units = len(data['spikes'])
#         total_units += n_units
#
#         print(f"\n[{session_count:03d}] {data['topdir']}/{data['session']:15s} "
#               f"→ {n_units:3d} units, "
#               f"position: {len(data['position']) if data['position'] is not None else 0}")
#
#         # 在这里可以做你的处理，例如保存到 hdf5、计算 firing rate 等
#         # lfp_process(data['lfp']['data'])
#
#     print(f"\n批量加载完成！共处理 {session_count} 个 session，总计 {total_units:,} 个神经元单元")