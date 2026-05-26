import os
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Optional, List
# 导入工具函数
from fea_extract.feature_utils_cellType import build_cell_type_mapping, compute_firing_rate_by_cell_type
from feature_utils_spike import compute_channel_population_rate

def load_spike_population_rates(
        base_path: str,
        topdir: str,
        session: str,
        electrodes: Optional[List[int]] = None,
        verbose: bool = True
) -> Dict:
    """
    仅提取 Spike 信号 → 计算每个通道的群体发放率
    输入：base_path、topdir、session
    输出：字典 → channel_rates：{通道号: 群体发放率数组}
    """
    # 数据目录
    raw_dir = Path(base_path) / topdir / session / "raw"
    if not raw_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {raw_dir}")

    if verbose:
        print(f"加载 Spike 数据：{topdir}/{session}")

    # 结果结构（仅保留通道发放率）
    result = {
        "topdir": topdir,
        "session": session,
        "sampling_rate": 20000,
        "channel_rates": {},       # 核心输出：通道 → 发放率
        "channel_rates_std": {},   # 标准差
        "channel_bins": {}         # 时间 bins
    }


    # 遍历所有 .clu 文件计算通道级群体发放率
    clu_files = sorted(raw_dir.glob("*.clu.*"))
    bin_size = 0.0256  # 25.6ms 时间窗口

    for clu_path in clu_files:
        # 解析通道号
        try:
            channel = int(clu_path.stem.split('.')[-1])
        except:
            print(f"× No valid .clu file")
            continue

        # 通道过滤
        if electrodes is not None and channel not in electrodes:
            continue

        # 检查对应 res 文件
        res_path = raw_dir / clu_path.name.replace(".clu.", ".res.")
        if not res_path.exists():
            print(f"× No valid .res file")
            continue

        # 计算群体发放率
        pop_rate, rate_std, bins = compute_channel_population_rate(
            clu_file_path=str(clu_path),
            fs=result["sampling_rate"],
            bin_size=bin_size
        )

        # 保存结果
        result["channel_rates"][channel] = pop_rate
        result["channel_rates_std"][channel] = rate_std
        result["channel_bins"][channel] = bins

    if verbose:
        print(f"完成！共提取 {len(result['channel_rates'])} 个通道的群体发放率")

    return result


# --------------------------
# 使用示例
# --------------------------
if __name__ == "__main__":
    BASE_PATH = "E:/crcns_hc3"  # 修改为你的数据集路径
    TOPDIR = "ec013.28"  # 输入topdir名称
    SESSION = "ec013.406"  # 输入session名称
    CSV_FILE = r"E:\crcns-hc3-metadata-tables\hc3-metadata-tables\hc3-cell.csv"

    # 通道级firing rate
    spike_result = load_spike_population_rates(
        base_path=BASE_PATH,
        topdir=TOPDIR,
        session=SESSION,
        verbose=True
    )

    # 查看输出
    print("\n==== 各通道群体发放率 ====")
    for ch, rate in spike_result["channel_rates"].items():
        print(f"通道 {ch}：发放率长度 = {len(rate)}")

