"""
spike_firing_rate_toolbox.py
功能：针对单个通道的.clu/.res文件，筛选高质量神经元, 计算群体发放率, 计算群体发放率标准差
核心流程：
1. 读取.clu和.res文件解析spike时间与神经元聚类信息
2. 筛选高质量神经元（4重筛选条件）
3. 计算瞬时群体发放率（指定时间窗口）
输入：clu文件路径，采样率，时间窗口大小
输出：群体发放率数组、时间bins数组、高质量神经元数量、活跃神经元数量
"""

import numpy as np
from pathlib import Path


def load_spike_data(clu_file_path):
    """
    读取单个通道的.clu和.res文件，解析spike信息
    :param clu_file_path: str/Path, .clu文件路径
    :return:
        cluster_ids: np.array, 每个spike对应的神经元编号
        spike_times_sec: np.array, 所有spike时间（秒）
    """
    clu_file_path = Path(clu_file_path)
    # 构造对应的.res文件路径
    res_file_path = clu_file_path.name.replace('.clu.', '.res.')
    res_file_path = clu_file_path.parent / res_file_path

    # 读取.clu文件（跳过第一行簇总数）
    clu = np.loadtxt(clu_file_path, dtype=int, skiprows=1)
    cluster_ids = clu[1:] if len(clu) > 1 else np.array([])

    # 读取.res文件（spike采样点时间）
    res = np.loadtxt(res_file_path, dtype=int, skiprows=1)
    spike_times_sample = res

    return cluster_ids, spike_times_sample


def filter_good_units(cluster_ids, spike_times_sec):
    """
    筛选高质量神经元（4重筛选条件）
    :param cluster_ids: np.array, 每个spike对应的神经元编号
    :param spike_times_sec: np.array, 所有spike时间（秒）
    :return:
        all_good_units: list, 高质量神经元编号列表
        active_cell_num: int, 活跃神经元数量（平均发放率>0.5Hz）
    """
    all_good_units = []
    active_cell_num = 0

    for unit in np.unique(cluster_ids):
        if unit < 2:
            continue  # 跳过噪声/多单元

        # 提取当前单元的spike时间
        mask = cluster_ids == unit
        spike_times = spike_times_sec[mask]
        n_spikes = np.sum(mask)

        # # 筛选1：至少50个spike
        # if n_spikes < 50:
        #     continue
        #
        # # 筛选2-3：ISI相关指标
        # if len(spike_times) < 2:
        #     continue
        # isi = np.diff(spike_times)
        # isi_ms = isi * 1000
        #
        # # 筛选2：ISI < 2ms 比例 < 0.01
        # refrac_viol = np.sum(isi_ms < 2) / len(isi)
        # if refrac_viol >= 0.01:
        #     continue
        #
        # # 筛选3：R2/10 < 0.2
        # r2 = np.sum(isi_ms < 2)
        # r10 = np.sum(isi_ms < 10)
        # refrac_ratio = r2 / r10 if r10 > 0 else 1.0
        # if refrac_ratio >= 0.2:
        #     continue

        # 筛选4：isolation distance >14（预留接口，可补充.fet文件解析逻辑）
        # eDist = compute_isolation_distance(...)
        # if eDist <= 14:
        #     continue

        # # 计算当前单元平均发放率，统计活跃神经元
        # bins, rate = get_firing_rate(spike_times)
        # if rate.mean() > 0.5:
        #     active_cell_num += 1

        all_good_units.append(unit)

    return all_good_units


def get_firing_rate(spike_times_sec, bin_size=0.1):
    """
    计算binned瞬时发放率（spikes/second）
    :param spike_times_sec: np.array, 单个/群体spike时间（秒）
    :param bin_size: float, 时间窗口大小（秒），默认0.1s(100ms)
    :return:
        bins: np.array, 时间bins数组
        rate: np.array, 对应每个bin的发放率（Hz）
    """
    if len(spike_times_sec) == 0:
        return np.array([]), np.array([])

    t_total = spike_times_sec.max()
    bins = np.arange(0, t_total + bin_size, bin_size)
    counts, _ = np.histogram(spike_times_sec, bins=bins)
    rate = counts / bin_size  # 转换为spikes per second
    return bins, rate

def calculate_firing_rate_std(population_rate):
    """
    计算群体发放率在时间维度上的标准差（STD）
    :param population_rate: np.array, 群体瞬时发放率数组（Hz）
    :return:
        rate_std: float, 时间维度发放率标准差；若输入为空则返回0
    """
    if len(population_rate) == 0:
        return 0.0
    return np.std(population_rate, ddof=1)  # ddof=1 计算样本标准差（更贴合实验数据统计）


def compute_channel_population_rate(clu_file_path, fs=20000.0, bin_size=0.1):
    """
    主函数：计算单个通道的群体神经元发放率
    :param clu_file_path: str/Path, 单个通道的.clu文件路径
    :param fs: float, 采样率（Hz），默认20000Hz
    :param bin_size: float, 时间窗口大小（秒），默认0.1s
    :return:
        population_rate: np.array, 群体瞬时发放率数组（Hz）
        rate_std: float, 群体发放率时间维度标准差（Hz）
        bins: np.array, 时间bins数组（秒）
        good_unit_num: int, 高质量神经元数量
        active_unit_num: int, 活跃神经元数量（平均发放率>0.5Hz）
    """
    default_good_unit_num = 0
    # 1. 加载spike数据
    cluster_ids, spike_times_sample = load_spike_data(clu_file_path)
    if len(cluster_ids) == 0 or len(spike_times_sample) == 0:
        print(f"⚠️ {clu_file_path} 数据为空")
        return np.array([]), np.array([]), 0, 0

    # 转换spike时间为秒
    spike_times_sec = spike_times_sample / fs

    # 2. 筛选高质量神经元
    all_good_units = filter_good_units(cluster_ids, spike_times_sec)
    good_unit_num = len(all_good_units)
    # print(f"✅ {clu_file_path.name} 包含高质量神经元 {good_unit_num} 个")
    # print(f"✅ {clu_file_path.name} 包含活跃神经元 {active_unit_num} 个")

    # 3. 计算群体发放率（仅高质量神经元）
    if not all_good_units:
        print(f"⚠️ {clu_file_path} 无高质量神经元，群体发放率为空")
        return np.array([]), np.array([]), np.array([]), good_unit_num

    sua_mask = np.isin(cluster_ids, all_good_units)
    pop_spikes = spike_times_sec[sua_mask]
    bins, population_rate = get_firing_rate(pop_spikes, bin_size)

    # 4. 计算群体发放率时间维度STD
    rate_std = calculate_firing_rate_std(population_rate)

    pop_rate_mean = population_rate.mean() if len(population_rate) > 0 else 0
    print(f"📊 {clu_file_path} 群体发放率均值：{pop_rate_mean:.2f} Hz")
    print(f"📊 {clu_file_path} 群体发放率时间维度STD：{rate_std:.2f} Hz\n")

    if all_good_units is None:
        good_unit_num = 0

    return population_rate, rate_std, bins, good_unit_num




