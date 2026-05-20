import os
import tarfile
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Optional, List


def load_hc3_session_full(
        base_path: str,
        topdir: str,
        session: str,
        load_spikes: bool = True,
        load_lfp: bool = True,
        load_position: bool = True,
        load_xml: bool = True,
        save_lfp: bool = True,
        verbose: bool = True
) -> Dict:
    """
    完整版 hc3 session 加载函数（适配解压后的文件读取）
    """
    base_path = Path(base_path)
    extract_dir = base_path / topdir / session / "raw"

    if not extract_dir.exists():
        raise FileNotFoundError(f"解压后的文件夹不存在: {extract_dir}")

    if verbose:
        print(f"正在加载: {topdir}/{session}")
        print(f"读取路径: {extract_dir}")

    data = {
        'topdir': topdir,
        'session': session,
        'spikes': {},
        'lfp': None,
        'position': None,
        'xml_info': None,
        'sampling_rate': 20000,  # 默认 spike 采样率 20kHz
        'lfp_sampling_rate': 1250  # .eeg 默认 1.25kHz
    }

    # ========== 核心改动：遍历解压后的文件夹（替代原tar包读取） ==========
    members = {}
    for root, _, files in os.walk(extract_dir):
        for file in files:
            file_path = Path(root) / file
            # 保留相对路径（相对于解压后的根目录），和原tar包内的文件路径逻辑对齐
            rel_path = file_path.relative_to(extract_dir)
            members[str(rel_path)] = file_path

    # ====================== 1. 加载 XML 配置信息 ======================
    if load_xml:
        xml_files = [name for name in members if name.endswith('.xml')]

        if xml_files:
            # 读取解压后的xml文件
            with open(members[xml_files[0]], 'rb') as f:
                tree = ET.parse(f)
                root = tree.getroot()

                # 尝试提取采样率信息
                acq = root.find('.//acquisitionSystem')
                if acq is not None:
                    n_channels = int(acq.find('nChannels').text) if acq.find('nChannels') is not None else None
                    sampling_rate = int(acq.find('samplingRate').text) if acq.find(
                        'samplingRate') is not None else 20000
                    data['xml_info'] = {
                        'n_channels': n_channels,
                        'sampling_rate': sampling_rate
                    }
                    data['sampling_rate'] = sampling_rate
                if verbose and data['xml_info'] is not None:
                    print(f"✓ XML 配置已加载")
                    print(f"✓ channel:{data['xml_info']['n_channels']} | sampling_rate:{data['xml_info']['sampling_rate']}")
        else:
            print(f"{xml_files} don't found!")

    # ====================== 2. 加载位置信息 (.whl) ======================
    if load_position:
        whl_files = [name for name in members if name.endswith('.whl')]
        if whl_files:
            # 读取解压后的whl文件
            with open(members[whl_files[0]], 'r') as f:
                pos = pd.read_csv(f, sep=r'\s+', header=None, engine='python')
                cols = ['x1', 'y1', 'x2', 'y2']
                pos.columns = cols
                data['position'] = pos
            if verbose:
                print(f"="*50)
                print(f"✓ 位置数据: {len(pos)} 个时间点")
        else:
            print(f"{whl_files} don't found!")

    # ====================== 3. 加载 Spike 数据 ======================
    if load_spikes:
        clu_files = [name for name in members if '.clu.' in name]

        for clu_name in sorted(clu_files):
            ele = int(clu_name.split('.')[-1])


            # 读取解压后的cluster文件
            with open(members[clu_name], 'r') as f:
                clu = np.loadtxt(f, dtype=int, skiprows=1)

            # 读取时间文件
            res_name = clu_name.replace('.clu.', '.res.')
            with open(members[res_name], 'r') as f:
                times = np.loadtxt(f, dtype=np.int64)

            # 按 cluster 分组
            for clu_id in np.unique(clu):
                if clu_id <= 1:  # 0=噪声, 1=不可分类
                    continue
                mask = clu == clu_id
                n_spikes = int(mask.sum())
                if n_spikes < 50: #
                    continue

                unit_key = f"ele{ele}_clu{clu_id}"
                data['spikes'][unit_key] = {
                    'times': times[mask],  # 采样点
                    'times_sec': times[mask] / data['sampling_rate'],  # 转换为秒
                    'cluster': int(clu_id),
                    'electrode': ele,
                    'n_spikes': int(mask.sum())
                }

        if verbose:
            print(f"=" * 50)
            print(f"✓ spike数据: 提取了 {len(data['spikes'])} 个神经元单元")
            if data['spikes']:  # 避免空字典导致的索引错误
                first_unit = next(iter(data['spikes'].values()))
                print(f"✓ spike time points of the first cell:{len(first_unit['times'])} 个")

    # ====================== 4. 加载 LFP (.eeg) ======================
    if load_lfp:
        eeg_files = [name for name in members if name.endswith('.eeg')]
        if eeg_files:
            eeg_name = eeg_files[0]
            # 读取解压后的eeg二进制文件
            with open(members[eeg_name], 'rb') as f:
                # .eeg 是 16-bit int 二进制文件
                lfp_raw = np.frombuffer(f.read(), dtype=np.int16)

                if data is None:
                    print(f"data is None!")
                else:
                    # 重塑形状 (假设多通道)
                    n_samples = len(lfp_raw) // (data.get('xml_info', {}).get('n_channels', 1) or 1)
                    n_ch = data.get('xml_info', {}).get('n_channels', 1) or 1

                    lfp = lfp_raw.reshape(n_samples, n_ch).T  # shape: (n_channels, n_samples)
                    data['lfp'] = {
                        'data': lfp,
                        'sampling_rate': data['lfp_sampling_rate'],
                        'times_sec': np.arange(lfp.shape[1]) / data['lfp_sampling_rate']
                    }

            if verbose:
                print(f"=" * 50)
                print(f"✓ LFP 数据已加载: {lfp.shape[0]} 通道, {lfp.shape[1]} 个采样点，采样率 {data['lfp_sampling_rate']}")
                print(f"✓ LFP {data['lfp']['times_sec'][-1]:.2f}秒")

    if verbose:
        total_spikes = sum(u['n_spikes'] for u in data['spikes'].values())
        print(f"加载完成！总 Spike 数量: {total_spikes:}")

    if save_lfp:
        save_lfp_npy(save_dir=base_path/topdir/session, save_name=session, data=data['lfp']['data'])

    return data


def save_lfp_npy(save_dir, save_name, data: np.ndarray):
    save_path = os.path.join(save_dir, save_name + "_LFP" + ".npy")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(save_path, data)
    print(f"Saved lfp at {save_path}")


if __name__ == '__main__':
    BASE_PATH = "E:/crcns_hc3"   # ← 修改为你的数据集路径
    session = "ec013.406"
    data = load_hc3_session_full(
        base_path=BASE_PATH,
        topdir="ec013.28",
        session=session,
        load_spikes=True,
        load_lfp=True,
        load_position=True,
        save_lfp=False,  # 加载lfp数据保存为.npy
        verbose=True
    )

    # 查看结果
    print(f"\n{session}中神经元数量: {len(data['spikes'])}")
    print(f"位置数据形状: {data['position'].shape if data['position'] is not None else None}")
    if data['lfp']:
        print(f"LFP 形状: {data['lfp']['data'].shape}")