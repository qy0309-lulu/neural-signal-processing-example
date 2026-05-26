import os

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# -------------------------- 复用核心工具函数（与之前完全兼容） --------------------------
def load_spike_data(clu_file_path: Path | str) -> Tuple[np.ndarray, np.ndarray]:
    """
    读取单个通道的.clu和.res文件，解析spike信息
    :param clu_file_path: .clu文件路径
    :return: cluster_ids(每个spike对应的神经元编号), spike_times_sample(所有spike采样点时间)
    """
    clu_file_path = Path(clu_file_path)
    # 自动匹配同目录下的.res文件
    res_file_name = clu_file_path.name.replace('.clu.', '.res.')
    res_file_path = clu_file_path.parent / res_file_name

    # 读取.clu文件（跳过第一行簇总数）
    clu_data = np.loadtxt(clu_file_path, dtype=int, skiprows=1)
    cluster_ids = clu_data[1:] if len(clu_data) > 1 else np.array([])

    # 读取.res文件（spike采样点时间）
    res_data = np.loadtxt(res_file_path, dtype=int, skiprows=1)
    spike_times_sample = res_data

    return cluster_ids, spike_times_sample


def get_firing_rate(
        spike_times_sec: np.ndarray,
        bin_size: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算binned瞬时发放率（spikes/second）
    :param spike_times_sec: 单个/群体spike时间（秒）
    :param bin_size: 时间窗口大小（秒），默认0.1s(100ms)
    :return: bins(时间bins数组), rate(对应每个bin的发放率Hz)
    """
    if len(spike_times_sec) == 0:
        return np.array([]), np.array([])

    t_total = spike_times_sec.max()
    bins = np.arange(0, t_total + bin_size, bin_size)
    counts, _ = np.histogram(spike_times_sec, bins=bins)
    rate = counts / bin_size  # 转换为spikes per second
    return bins, rate


# -------------------------- 本次适配topdir的核心新增/调整函数 --------------------------
def build_cell_type_mapping(
        csv_file_path: Path | str,
        key_columns: List[str] = ["topdir", "ele", "clu"],
        cell_type_column: str = "cellType"
) -> Dict[Tuple, str]:
    """
    从CSV文件构建神经元细胞类型映射字典（唯一键：topdir+ele+clu）
    :param csv_file_path: CSV文件路径
    :param key_columns: 作为唯一键的列名，默认["topdir", "ele", "clu"]
    :param cell_type_column: 细胞类型列名，默认"cellType"
    :return: 映射字典，key为(key_columns对应值的元组)，value为细胞类型
    """
    # 读取CSV文件
    df = pd.read_csv(csv_file_path)
    # 数据清洗：去除空值、去重（避免同一key对应多个类型）
    df_clean = df.dropna(subset=key_columns + [cell_type_column]).drop_duplicates(subset=key_columns)
    # 构建映射字典
    mapping = {}
    for _, row in df_clean.iterrows():
        # 按key_columns顺序构建元组key
        key = tuple(str(row[col]).strip() for col in key_columns)
        mapping[key] = str(row[cell_type_column]).strip()
    print(f"✅ 已构建细胞类型映射，共包含 {len(mapping)} 个神经元，唯一键：{key_columns}")
    return mapping


def classify_neurons_by_type(
        clu_file_path: Path | str,
        cell_type_mapping: Dict[Tuple, str],
        fs: float = 20000.0,
        key_columns: List[str] = ["topdir", "ele", "clu"]
) -> Dict[str, np.ndarray]:
    """
    加载spike数据，按细胞类型对神经元进行分类，返回各类型的spike时间
    :param clu_file_path: .clu文件路径
    :param cell_type_mapping: 细胞类型映射字典
    :param fs: 采样率（Hz），默认20000Hz
    :param key_columns: 映射字典的键列，默认["topdir", "ele", "clu"]
    :return: 分类字典，key为细胞类型，value为该类型所有神经元的spike时间（秒）
    """
    # 1. 从路径提取topdir和ele
    file_path = clu_file_path

    # 拆分路径所有部分
    parts = os.path.normpath(file_path).split(os.sep)
    # 提取topdir
    topdir = parts[2]  # 直接拿到 ec013.28
    # 提取最后的数字
    file_name = os.path.basename(file_path)  # 拿到 ec013.395.clu.1
    ele = file_name.split('.')[-1]  # 按 . 分割，取最后一段 → 1

    # 2. 加载spike数据
    cluster_ids, spike_times_sample = load_spike_data(clu_file_path)
    if len(cluster_ids) == 0 or len(spike_times_sample) == 0:
        print(f"⚠️ {clu_file_path} 无有效spike数据")
        return {}
    # 转换spike时间为秒
    spike_times_sec = spike_times_sample / fs

    # 3. 按细胞类型分类
    type_spike_dict: Dict[str, List[np.ndarray]] = {}
    # 遍历所有唯一神经元
    for unit_id in np.unique(cluster_ids):
        if unit_id < 2:
            continue  # 跳过噪声/多单元（神经元编号<2为无效值）
        # 构建映射key（与CSV的key_columns顺序完全一致）
        key = tuple(
            [topdir, ele, str(unit_id)] if key_columns == ["topdir", "ele", "clu"] else [topdir, ele, str(unit_id)])
        # 查细胞类型
        if key not in cell_type_mapping:
            print(f"⚠️ 神经元 {key} 未在映射字典中找到，跳过")
            continue
        cell_type = cell_type_mapping[key]
        # 提取该神经元的spike时间
        unit_mask = cluster_ids == unit_id
        unit_spikes = spike_times_sec[unit_mask]
        # 加入对应类型
        if cell_type not in type_spike_dict:
            type_spike_dict[cell_type] = []
        type_spike_dict[cell_type].append(unit_spikes)

    # 4. 合并每个类型的所有spike时间
    merged_type_dict: Dict[str, np.ndarray] = {}
    for cell_type, spike_list in type_spike_dict.items():
        merged_type_dict[cell_type] = np.concatenate(spike_list) if spike_list else np.array([])
        print(f"✅ 细胞类型 {cell_type}: 共 {len(spike_list)} 个神经元，{len(merged_type_dict[cell_type])} 个spike")

    return merged_type_dict


def compute_firing_rate_by_cell_type(
        cell_type_mapping: Dict[Tuple, str],  # 外部传入映射字典
        clu_file_path: Path | str,
        fs: float = 20000.0,
        bin_size: float = 0.1,
        key_columns: List[str] = ["topdir", "ele", "clu"],
        fixed_bins=None
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    主函数：计算单个.clu文件各细胞类型的瞬时发放率（适配topdir路径匹配）
    流程：从.clu路径提取topdir/ele → 神经元分类 → 计算发放率
    :param cell_type_mapping: 细胞类型映射字典（外部构建后传入）
    :param clu_file_path: .clu文件路径
    :param fs: 采样率（Hz），默认20000Hz
    :param bin_size: 时间窗口大小（秒），默认0.1s(100ms)
    :param key_columns: 映射字典的键列，默认["topdir", "ele", "clu"]
    :return:
        rate_dict: 字典，key为细胞类型，value为对应瞬时发放率数组（Hz）
        bins: 时间bins数组（秒），所有类型共用同一时间轴
    """
    print("=" * 50)
    print(f"开始处理文件：{clu_file_path}")
    print("=" * 50)

    # 1. 按细胞类型分类神经元（直接使用外部传入的映射字典）
    type_spike_dict = classify_neurons_by_type(clu_file_path, cell_type_mapping, fs, key_columns)
    if not type_spike_dict:
        print("❌ 无有效分类数据，发放率计算终止")
        return {}, np.array([])

    # 2. 计算各类型瞬时发放率
    rate_dict: Dict[str, np.ndarray] = {}
    # 先获取全局时间轴（所有spike的最大时间，保证所有类型时间轴统一）
    bins = fixed_bins
    if fixed_bins is None:
        all_spikes = np.concatenate(list(type_spike_dict.values()))
        t_total = all_spikes.max()
        bins = np.arange(0, t_total + bin_size, bin_size)

    # 逐个计算发放率
    for cell_type, spikes in type_spike_dict.items():

        clean_type = cell_type.strip().upper()
        if clean_type in ["N", "NA", "NONE"]:
            continue

        counts, _ = np.histogram(spikes, bins=bins)
        rate = counts / bin_size
        rate_dict[cell_type] = rate
        print(f"\n📊 细胞类型 {cell_type} 统计结果：")
        print(f"  - 平均发放率：{rate.mean():.2f} Hz")
        print(f"  - 发放率时间维度STD：{rate.std(ddof=1):.2f} Hz")
        print(f"  - 最大发放率：{rate.max():.2f} Hz")
        print(f"  - 时间bin数量：{len(rate)}")

    print("\n" + "=" * 50)
    print(f"🎉 处理完成！共计算 {len(rate_dict)} 种细胞类型的发放率")
    print("=" * 50)

    return rate_dict, bins


# -------------------------- 使用示例（支持多.clu文件批量处理） --------------------------
if __name__ == "__main__":
    # 1. 核心配置（必须根据你的实际情况修改）
    CSV_FILE = r"E:\crcns-hc3-metadata-tables\hc3-metadata-tables\hc3-cell.csv"  # 替换为你的CSV文件路径
    # 支持单个/多个.clu文件处理（示例：添加多个文件路径到列表）
    CLU_FILE_LIST = [
        r"E:\crcns_hc3\ec013.28\ec013.395\raw\ec013.395.clu.1",  # 第一个.clu文件
        # r"E:\crcns_hc3\ec013.28\ec013.396\raw\ec013.396.clu.1",  # 第二个.clu文件（按需添加）
        # r"E:\crcns_hc3\ec013.28\ec013.397\raw\ec013.397.clu.1",  # 第三个.clu文件（按需添加）
    ]
    FS = 20000.0  # 采样率（Hz）
    BIN_SIZE = 0.0256  # 时间窗口大小（秒），25.6ms

    # 2. 一次性构建映射字典（多文件复用，无需重复构建）
    cell_type_mapping = build_cell_type_mapping(
        csv_file_path=CSV_FILE,
        key_columns=["topdir", "ele", "clu"],
        cell_type_column="cellType"
    )

    # 3. 批量处理所有.clu文件
    all_results = {}  # 存储所有文件的结果：key=文件路径，value=(rate_dict, time_bins)
    for clu_file in CLU_FILE_LIST:
        firing_rate_dict, time_bins = compute_firing_rate_by_cell_type(
            cell_type_mapping=cell_type_mapping,  # 传入预构建的映射字典
            clu_file_path=clu_file,
            fs=FS,
            bin_size=BIN_SIZE,
        )
        all_results[clu_file] = (firing_rate_dict, time_bins)

    # 4. 结果查看示例
    print("\n" + "=" * 60)
    print("📋 所有文件最终结果概览：")
    print("=" * 60)
    for clu_file, (rate_dict, time_bins) in all_results.items():
        print(f"\n📄 文件：{clu_file}")
        for cell_type, rate in rate_dict.items():
            print(f"  - {cell_type}: 平均发放率 {rate.mean():.2f} Hz，共 {len(rate)} 个时间点")