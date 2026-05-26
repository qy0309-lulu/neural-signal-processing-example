import numpy as np
from scipy.signal import butter, filtfilt, hilbert

# functions collection for LFP feature extraction
# instantaneous power for specific band
# ------------------------------------------------------------------------------
# 1. 带通滤波（零相位，论文标准）
# ------------------------------------------------------------------------------
def bandpass_filter(data, fs, low, high, order=4):
    """
    零相位带通滤波
    :param data: 原始 LFP 信号 (1D numpy array)
    :param fs: 采样率 (Hz)
    :param low: 低频截止
    :param high: 高频截止
    :param order: 滤波器阶数
    :return: 滤波后信号
    """
    nyq = 0.5 * fs
    low = low / nyq
    high = high / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, data)  # 零相位滤波 → 无延迟
    return filtered

# ------------------------------------------------------------------------------
# 2. Hilbert 变换 → 瞬时振幅
# ------------------------------------------------------------------------------
def get_instantaneous_amplitude(signal):
    """
    对单频信号做 Hilbert 变换，提取瞬时振幅（包络）
    """
    analytic_signal = hilbert(signal)
    amplitude = np.abs(analytic_signal)
    return amplitude

# ------------------------------------------------------------------------------
# 3. 瞬时振幅 → 瞬时功率
# ------------------------------------------------------------------------------
def amplitude_to_power(amplitude):
    """
    瞬时功率 = 瞬时振幅²
    这是信号处理 + 神经科学标准定义
    """
    power = amplitude ** 2
    return power

# ------------------------------------------------------------------------------
# 4. 提取指定频段的瞬时功率（主函数：一行调用）
# ------------------------------------------------------------------------------
def get_band_power_amp(lfp, fs, low, high):
    """
    从 LFP 时序中提取【某一频段】的瞬时功率时序
    标准流程：滤波 → Hilbert → 振幅 → 功率
    """
    filtered = bandpass_filter(lfp, fs, low, high)
    amp = get_instantaneous_amplitude(filtered)
    power = amplitude_to_power(amp)
    return power, amp
# ------------------------------------------------------------------------------
# 5. 整合函数：一次 Hilbert 得到 幅度 / 相位 / 功率（三个特征）
# # ------------------------------------------------------------------------------
# def get_band_hilbert_features(lfp, fs, low, high, unwrap_phase=True):
#     """
#     从LFP信号中提取指定频段的 Hilbert 三大特征
#     🔹 只做 1 次滤波 + 1 次 Hilbert 变换
#     🔹 同时输出：瞬时幅度、瞬时相位、瞬时功率
#     """
#     # 1. 带通滤波（只做一次）
#     filtered = bandpass_filter(lfp, fs, low, high)
#
#     # 2. Hilbert 变换（只做一次！）
#     analytic_signal = hilbert(filtered)
#
#     # 3. 一次性计算三个特征
#     instantaneous_amplitude = np.abs(analytic_signal)  # 幅度
#     instantaneous_phase = np.angle(analytic_signal)  # 相位
#     if unwrap_phase:
#         instantaneous_phase = np.unwrap(instantaneous_phase)  # 相位解缠绕
#     instantaneous_power = instantaneous_amplitude ** 2  # 功率（幅度平方）
#
#     return instantaneous_amplitude, instantaneous_phase, instantaneous_power

    # example for using:
    # theta_power = get_band_power(lfp, fs, low=4, high=12)
    # gamma_power = get_band_power(lfp, fs, low=30, high=80)

