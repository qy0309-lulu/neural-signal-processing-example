import os
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List

# 导入细胞类型映射函数（保持原有依赖）
from fea_extract.feature_utils_cellType import build_cell_type_mapping


def get_spike_times(res_file: str, fs: int) -> np.ndarray:
    """读取res文件，返回归一化后的spike时间（秒）"""
    res = np.loadtxt(res_file, dtype=int)
    return res / fs


def get_session_global_time_range(raw_dir: Path, fs: int = 20000) -> tuple[float, float]:
    """
    获取单个session的全局时间范围（所有电极的spike最早/最晚时间）
    :param raw_dir: session的raw目录路径
    :param fs: 采样率
    :return: (全局起始时间, 全局结束时间)
    """
    all_spike_times = []
    clu_files = sorted(raw_dir.glob("*.clu.*"))

    for clu_path in clu_files:
        res_path = raw_dir / clu_path.name.replace('.clu.', '.res.')
        if not res_path.exists() or os.path.getsize(res_path) == 0:
            continue
        spike_times = get_spike_times(str(res_path), fs)
        all_spike_times.extend(spike_times)

    if not all_spike_times:  # 空session默认时间范围
        return 0.0, 1.0

    all_spike_times = np.array(all_spike_times)
    return all_spike_times.min(), all_spike_times.max()


def compute_unit_firing_rates(
        clu_file: str,
        res_file: str,
        global_t_start: float,
        global_t_end: float,
        fs: int = 20000,
        bin_size: float = 0.0256
) -> tuple[List[np.ndarray], np.ndarray]:
    """
    计算单个电极上所有有效神经元的放电率（基于全局时间轴）
    :param clu_file: .clu文件路径
    :param res_file: .res文件路径
    :param global_t_start: session全局起始时间
    :param global_t_end: session全局结束时间
    :param fs: 采样率
    :param bin_size: 时间窗大小（秒）
    :return: (神经元放电率列表, 有效神经元ID列表)
    """
    # 读取聚类标签和spike时间
    cluster_labels = np.loadtxt(clu_file, dtype=int)[1:]  # 跳过第一行（聚类数）
    spike_times = get_spike_times(res_file, fs)

    # 构建全局统一时间bins
    bins = np.arange(global_t_start, global_t_end + bin_size, bin_size)

    # 筛选有效神经元（cid>1，排除噪音聚类）
    valid_cids = [cid for cid in np.unique(cluster_labels) if cid > 1]
    firing_rates = []

    for cid in valid_cids:
        # 提取单个神经元的spike时间并计算放电率
        unit_spikes = spike_times[cluster_labels == cid]
        fr, _ = np.histogram(unit_spikes, bins=bins)
        firing_rates.append(fr / bin_size)  # 放电率 = 每个bin的spike数 / bin大小

    return firing_rates, np.array(valid_cids)


def process_single_session(
        session_path: Path,
        cell_type_mapping: dict,
        topdir_name: str,
        bin_size: float = 1
) -> List[Dict]:
    """
    处理单个session，提取所有神经元的放电率和细胞类型
    :param session_path: session目录路径
    :param cell_type_mapping: 细胞类型映射字典
    :param topdir_name: 顶级目录名称（用于细胞类型匹配）
    :param bin_size: 时间窗大小
    :return: 神经元信息列表（包含电极、cid、放电率、细胞类型）
    """
    raw_dir = session_path / "raw"
    if not raw_dir.exists():
        return []

    # 解析采样率
    fs = 20000
    # 获取session全局时间范围
    global_t_start, global_t_end = get_session_global_time_range(raw_dir, fs)

    units = []
    clu_files = sorted(raw_dir.glob("*.clu.*"))

    for clu_path in clu_files:
        # 解析电极号
        ele = int(clu_path.name.split('.')[-1])
        res_path = raw_dir / clu_path.name.replace('.clu.', '.res.')

        if not res_path.exists() or os.path.getsize(res_path) == 0:
            continue

        # 计算当前电极的神经元放电率
        fr_list, cids = compute_unit_firing_rates(
            clu_file=str(clu_path),
            res_file=str(res_path),
            global_t_start=global_t_start,
            global_t_end=global_t_end,
            fs=fs,
            bin_size=bin_size
        )

        # 匹配细胞类型并整理结果
        for cid, fr in zip(cids, fr_list):
            key = (topdir_name, ele, cid)
            units.append({
                "ele": ele,
                "cid": int(cid),
                "firing_rate": fr,
                "cell_type": cell_type_mapping.get(key, "unknown")
            })

    return units


def merge_session_neurons(
        base_path: str,
        topdir: str,
        csv_file: str,
        bin_size: float = 0.0256
) -> Dict:
    """
    合并单个topdir下所有session的神经元数据，生成特征矩阵
    :param base_path: 基础路径
    :param topdir: 顶级目录名称
    :param csv_file: 细胞类型csv文件路径
    :param bin_size: 时间窗大小
    :return: 包含特征矩阵、特征名、神经元ID、细胞类型的字典
    """
    topdir_path = Path(base_path) / topdir
    session_dirs = sorted([d for d in topdir_path.iterdir() if d.is_dir()])
    if not session_dirs:
        raise ValueError(f"未找到session目录: {topdir_path}")

    # 构建细胞类型映射
    cell_type_mapping = build_cell_type_mapping(
        csv_file_path=csv_file,
        key_columns=["topdir", "ele", "clu"],
        cell_type_column="cellType"
    )

    # 处理所有session，收集神经元数据
    session_units = []
    session_lengths = []  # 每个session的时间步长度
    for sess_dir in session_dirs:
        units = process_single_session(sess_dir, cell_type_mapping, topdir, bin_size)
        session_units.append(units)
        # 记录当前session的时间步长度（无神经元则为0）
        session_lengths.append(len(units[0]["firing_rate"]) if units else 0)

    total_timesteps = sum(session_lengths)
    print(f"📏 所有session总时间步长度: {total_timesteps}")

    # 按神经元(ele, cid)合并跨session的放电率
    neuron_dict = {}
    for sess_idx, units in enumerate(session_units):
        sess_len = session_lengths[sess_idx]
        for unit in units:
            neuron_key = (unit["ele"], unit["cid"])
            if neuron_key not in neuron_dict:
                neuron_dict[neuron_key] = {
                    "cell_type": unit["cell_type"],
                    "sess_indices": [],
                    "firing_rates": []
                }
            neuron_dict[neuron_key]["sess_indices"].append(sess_idx)
            neuron_dict[neuron_key]["firing_rates"].append(unit["firing_rate"])

    if not neuron_dict:
        raise RuntimeError("❌ 未提取到有效神经元数据")

    # 构建完整的特征序列（填充跨session的空缺）
    final_sequences = []
    neuron_keys = []
    cell_types = []

    for neuron_key, info in neuron_dict.items():
        full_seq = np.zeros(total_timesteps)
        current_pos = 0

        for sess_idx in range(len(session_dirs)):
            sess_len = session_lengths[sess_idx]
            if sess_idx in info["sess_indices"]:
                # 填充当前session的放电率
                fr_idx = info["sess_indices"].index(sess_idx)
                fr = info["firing_rates"][fr_idx]
                # 兼容长度异常（边缘情况）
                if len(fr) != sess_len:
                    print(f"⚠️ 警告: 神经元{neuron_key} session{sess_idx}长度不匹配，已填充")
                    fr = np.pad(fr[:sess_len], (0, max(0, sess_len - len(fr))), mode='constant')
                full_seq[current_pos:current_pos + sess_len] = fr
            current_pos += sess_len

        final_sequences.append(full_seq)
        neuron_keys.append(neuron_key)
        cell_types.append(info["cell_type"])

    # 构建特征矩阵（时间步 × 神经元数）
    feature_matrix = np.column_stack(final_sequences)

    # 计算p/i群体放电率并追加到特征矩阵
    p_idx = [i for i, ct in enumerate(cell_types) if ct == 'p']
    i_idx = [i for i, ct in enumerate(cell_types) if ct == 'i']
    p_pop = np.sum(feature_matrix[:, p_idx], axis=1) if p_idx else np.zeros(total_timesteps)
    i_pop = np.sum(feature_matrix[:, i_idx], axis=1) if i_idx else np.zeros(total_timesteps)

    # 合并群体放电率到特征矩阵
    final_feature_matrix = np.hstack([
        feature_matrix,
        p_pop.reshape(-1, 1),
        i_pop.reshape(-1, 1)
    ])

    # 构建特征名称
    feature_names = [f"neuron_ele{ele}_clu{cid}" for ele, cid in neuron_keys]
    feature_names += ["p_population", "i_population"]

    # 整理神经元ID信息
    ele_ids = np.array([k[0] for k in neuron_keys])
    cluster_ids = np.array([k[1] for k in neuron_keys])

    print(f"\n✅ 特征矩阵形状 (时间步 × 特征数): {final_feature_matrix.shape}")
    print(f"✅ {topdir}中cluster数: {len(neuron_keys)}")

    return {
        "feature_matrix": final_feature_matrix,
        "feature_names": np.array(feature_names),
        "ele_ids": ele_ids,
        "cluster_ids": cluster_ids,
        "cell_types": np.array(cell_types)
    }


def save_features(result: Dict, save_dir: str):
    """保存特征矩阵及相关信息"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    # 保存特征矩阵（核心输出）
    np.save(save_dir / "final_feature_matrix.npy", result["feature_matrix"])
    # 保存特征名称
    with open(save_dir / "feature_names.txt", "w") as f:
        f.write("\n".join(result["feature_names"]))
    # 保存神经元ID
    np.savez(save_dir / "neuron_ids.npz",
             ele_ids=result["ele_ids"],
             cluster_ids=result["cluster_ids"])

    print(f"\n✅ 结果已保存至: {save_dir}")


if __name__ == '__main__':
    # 配置参数（保持原输入逻辑）
    BASE_PATH = "E:/crcns_hc3"
    TOPDIR = "ec013.28"
    CSV_FILE = r"E:\crcns-hc3-metadata-tables\hc3-metadata-tables\hc3-cell.csv"
    BIN_SIZE = 1  # 时间窗选择

    # 核心流程：处理数据→生成特征矩阵→保存
    feature_result = merge_session_neurons(BASE_PATH, TOPDIR, CSV_FILE, BIN_SIZE)
    # save_features(feature_result, Path(BASE_PATH) / TOPDIR)

    # 输出最终特征矩阵形状
    X = feature_result["feature_matrix"]
    print(f"\n🎯 最终可用特征矩阵 shape: {X.shape}")