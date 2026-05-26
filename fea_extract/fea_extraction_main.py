import os
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Optional, List

from fea_extract.feature_utils_cellType import build_cell_type_mapping, compute_firing_rate_by_cell_type
# 导入工具函数
from feature_utils_LFP import get_band_power_amp
from feature_utils_spike import compute_channel_population_rate

from preprocess_LFP.LFP_prep import lfp_process


def load_hc3_session_full(
        base_path: str,
        topdir: str,
        session: str,
        CSV_FILE: Path | str,
        electrodes: Optional[List[int]] = None,
        load_spikes: bool = True,
        load_lfp: bool = True,
        load_position: bool = True,
        load_xml: bool = True,
        save_lfp: bool = True,
        verbose: bool = True
) -> Dict:
    """
    完整版 hc3 session 加载函数，整合特征提取功能：
    1. LFP：提取theta(4-12Hz)、gamma(30-80Hz)频段瞬时功率
    2. Spike：提取每个通道群体发放率、发放率STD
    输出包含特征矩阵的完整数据字典
    """
    # 修改：直接指向解压后的raw目录
    raw_dir = Path(base_path) / topdir / session / "raw"
    if not raw_dir.exists():
        raise FileNotFoundError(f"解压后的目录不存在: {raw_dir}")

    if verbose:
        print(f"正在加载: {topdir}/{session} (从解压目录读取)")

    # 初始化数据字典
    data = {
        'topdir': topdir,
        'session': session,
        'spikes': {},
        'lfp': None,
        'position': None,
        'xml_info': None,
        'sampling_rate': 20000,  # spike 采样率 20kHz
        'lfp_sampling_rate': 1250,  # LFP采样率 1.25kHz
        # 新增特征存储字段
        'lfp_features': {  # LFP频段功率特征
            'theta_power': None,  # 4-12Hz 瞬时功率
            'gamma_power': None,  # 30-80Hz 瞬时功率
            # 'theta_gamma_ratio': None  # theta/gamma功率比（可选）
        },
        'spike_features': {  # Spike群体发放率特征
            'channel_rates': {},  # 各通道群体发放率数组
            'channel_rates_std': {},  # 各通道发放率STD
            'good_units_per_channel': {},  # 各通道高质量神经元数
            'active_units_per_channel': {}  # 各通道活跃神经元数
        },
        'special_cell_fr': {},
        'feature_matrix': None  # 最终特征矩阵
    }

    # ====================== 1. 加载 XML 配置信息 ======================
    if load_xml:
        xml_files = list(raw_dir.glob("*.xml"))
        if xml_files:
            tree = ET.parse(xml_files[0])
            root = tree.getroot()
            acq = root.find('.//acquisitionSystem')
            if acq is not None:
                n_channels = int(acq.find('nChannels').text) if acq.find('nChannels') is not None else None
                sampling_rate = int(acq.find('samplingRate').text) if acq.find('samplingRate') is not None else 20000
                data['xml_info'] = {'n_channels': n_channels, 'sampling_rate': sampling_rate}
                data['sampling_rate'] = sampling_rate
            if verbose and data['xml_info'] is not None:
                print(
                    f"✓ XML 配置已加载 | 电极数:{data['xml_info']['n_channels']} | 采样率:{data['xml_info']['sampling_rate']}")
        else:
            print(f"⚠️ 未找到XML文件")

    # # ====================== 2. 加载位置信息 (.whl) ======================
    # if load_position:
    #     whl_files = list(raw_dir.glob("*.whl"))
    #     if whl_files:
    #         pos = pd.read_csv(whl_files[0], sep=r'\s+', header=None, engine='python')
    #         pos.columns = ['x1', 'y1', 'x2', 'y2']
    #         data['position'] = pos
    #         if verbose:
    #             print(f"✓ 位置数据: {len(pos)} 个时间点")
    #     else:
    #         print(f"⚠️ 未找到位置文件(.whl)")

    # ====================== 3. 加载并处理 Spike 数据 (含特征提取) ======================
    if load_spikes:
        clu_files = list(raw_dir.glob("*.clu.*"))

        cell_type_mapping = build_cell_type_mapping(
            csv_file_path=CSV_FILE,
            key_columns=["topdir", "ele", "clu"],
            cell_type_column="cellType"
        )

        if not clu_files:
            print(f"⚠️ 未找到Spike文件(.clu)")
        else:
            for clu_path in sorted(clu_files):
                # 解析通道号
                ele = int(clu_path.name.split('.')[-1])
                if electrodes and ele not in electrodes:
                    continue

                # 找到对应的.res文件
                res_path = raw_dir / clu_path.name.replace('.clu.', '.res.')
                if not res_path.exists():
                    print(f"⚠️ 未找到{clu_path.name}对应的.res文件，跳过")
                    continue

                # 调用工具函数计算群体发放率
                pop_rate, rate_std, bins, _ = compute_channel_population_rate(
                    clu_file_path=str(clu_path),
                    fs=data['sampling_rate'],
                    bin_size=0.0256  # 25.6ms时间窗口 -> 和速度序列匹配
                )


                firing_rate_dict,_ = compute_firing_rate_by_cell_type(
                    cell_type_mapping=cell_type_mapping,  # 传入预构建的映射字典
                    clu_file_path=clu_path,
                    fs=data['sampling_rate'],
                    bin_size=0.0256,
                    fixed_bins=bins
                )

                for cell_type, rate in firing_rate_dict.items():
                    if ele not in data['special_cell_fr']:
                        data['special_cell_fr'][ele] = {}
                    data['special_cell_fr'][ele][cell_type] = rate

                # 存储通道级Spike特征
                data['spike_features']['channel_rates'][ele] = pop_rate
                data['spike_features']['channel_rates_std'][ele] = rate_std

            if verbose:
                print(f"✓ Spike特征：{len(data['spike_features']['channel_rates'])} 个通道的群体发放率")
                # 打印细胞类型fr
                cell_types = set()
                for ele in data['special_cell_fr']:
                    cell_types.update(data['special_cell_fr'][ele].keys())
                print(f"✓ 检测到细胞类型: {cell_types}")

    # ====================== 4. 加载并处理 LFP 数据 (含特征提取) ======================
    if load_lfp:
        eeg_files = list(raw_dir.glob("*.eeg"))

        if verbose:
            print(f"开始处理lfp信号")

        if eeg_files:
            eeg_path = eeg_files[0]
            # 读取LFP原始数据
            with open(eeg_path, 'rb') as f:
                lfp_raw = np.frombuffer(f.read(), dtype=np.int16)

            n_ch = data.get('xml_info', {}).get('n_channels', 1) or 1
            n_samples = len(lfp_raw) // n_ch
            lfp = lfp_raw.reshape(n_samples, n_ch).T  # (n_channels, n_samples)

            # preprocess
            lfp = lfp_process(lfp,target_sfreq=39.0625).get_data()
            # deposit into data
            data['lfp'] = {
                'data': lfp,
                'sampling_rate': data['lfp_sampling_rate'],
                'times_sec': np.arange(lfp.shape[1]) / data['lfp_sampling_rate']
            }

            # 提取LFP频段功率特征（theta:4-12Hz, gamma:30-80Hz）
            fs_lfp = data['lfp_sampling_rate']
            n_channels = lfp.shape[0]

            # 对每个通道提取频段功率
            theta_band_p = []
            theta_band_a = []
            gamma_band_p = []
            gamma_band_a = []
            for ch in range(n_channels):
                ch_lfp = lfp[ch]
                # 提取theta频段功率
                theta_p, theta_a = get_band_power_amp(ch_lfp, fs_lfp, low=4, high=12)
                # 提取gamma频段功率
                gamma_p, gamma_a = get_band_power_amp(ch_lfp, fs_lfp, low=30, high=80)
                theta_band_p.append(theta_p)
                gamma_band_p.append(gamma_p)
                theta_band_a.append(theta_a)
                gamma_band_a.append(gamma_a)

            # 2. 转为数组 + 所有通道**时间维度逐点平均**
            theta_band_p = np.array(theta_band_p)  # (n_ch, n_time)
            gamma_band_p = np.array(gamma_band_p)  # (n_ch, n_time)

            # 关键修改：按通道求均值，得到全局平均功率时序 (n_time,)
            avg_theta_p = np.mean(theta_band_p, axis=0)
            avg_gamma_p = np.mean(gamma_band_p, axis=0)
            avg_tg_ratio_p = avg_theta_p / (avg_gamma_p + 1e-8)

            theta_band_a = np.array(theta_band_a)  # (n_ch, n_time)
            gamma_band_a = np.array(gamma_band_a)  # (n_ch, n_time)

            # 关键修改：按通道求均值，得到全局平均功率时序 (n_time,)
            avg_theta_a = np.mean(theta_band_a, axis=0)
            avg_gamma_a = np.mean(gamma_band_a, axis=0)

            # 存入特征
            data['lfp_features']['avg_theta_power'] = avg_theta_p
            data['lfp_features']['avg_gamma_power'] = avg_gamma_p
            data['lfp_features']['avg_theta_amp'] = avg_theta_a
            data['lfp_features']['avg_gamma_amp'] = avg_gamma_a
            data['lfp_features']['avg_theta_gamma_p_ratio'] = avg_tg_ratio_p

            if verbose:
                print(f"✓ LFP数据: {lfp.shape[0]} 通道, {lfp.shape[1]} 采样点")
                print(f"✓ LFP全局平均特征 theta:{len(avg_theta_p)}, gamma:{len(avg_gamma_p)}")

            # 保存LFP为npy文件
            if save_lfp:
                save_lfp_npy(save_dir=Path(base_path) / topdir / session, save_name=session, data=lfp)
        else:
            print(f"⚠️ 未找到LFP文件(.eeg)")

    # ====================== 5. 构建特征矩阵 ======================
    # 特征矩阵构建逻辑：
    # - 行：时间维度（对齐LFP和Spike的时间窗口）
    # - 列：特征维度（各通道theta功率、gamma功率、群体发放率、发放率STD等）
    if load_lfp or load_spikes:
        if verbose:
            print("\n正在构建特征矩阵...")
        feature_list = []
        feature_names = []

        # 取LFP平均功率时间长度作为基准
        avg_theta = data['lfp_features']['avg_theta_power']
        min_time_len = len(avg_theta)

        # 1. 添加LFP全局平均特征
        feature_list.append(avg_theta)
        feature_names.append("avg_theta_power")

        feature_list.append(data['lfp_features']['avg_gamma_power'])
        feature_names.append("avg_gamma_power")

        feature_list.append(data['lfp_features']['avg_theta_amp'])
        feature_names.append("avg_theta_amp")

        feature_list.append(data['lfp_features']['avg_gamma_amp'])
        feature_names.append("avg_gamma_amp")

        feature_list.append(data['lfp_features']['avg_theta_gamma_p_ratio'])
        feature_names.append("avg_theta_gamma_p_ratio")

        # 2. 添加Spike特征（各通道群体发放率、发放率STD）
        # 对齐Spike发放率的时间窗口到LFP时间轴
        # spike_channels = sorted(data['spike_features']['channel_rates'].keys())
        ALL_CH_INDEX = [1, 2, 3, 4, 5, 6, 7, 8]
        spike_features = data.get('spike_features', {})

        for ch in ALL_CH_INDEX:
            pop_rate = data['spike_features']['channel_rates'][ch]
            # 插值对齐到LFP时间长度
            if len(pop_rate) > 0:
                pop_rate_aligned = np.interp(
                    np.linspace(0, 1, min_time_len),
                    np.linspace(0, 1, len(pop_rate)),
                    pop_rate
                )
            else:
                pop_rate_aligned = np.zeros(min_time_len)

            feature_list.append(pop_rate_aligned)
            feature_names.append(f"ch{ch}_pop_firing_rate")

            # 发放率STD（标量扩展为时间序列）
            channel_rates_std = spike_features.get('channel_rates_std', {})
            rate_std = channel_rates_std.get(ch, 0.0)  # 不存在默认 0

            # 确保是数值，不是 None / 空
            if not isinstance(rate_std, (int, float)) or rate_std is None:
                rate_std = 0.0

            # rate_std = data['spike_features']['channel_rates_std'][ch]
            feature_list.append(np.full(min_time_len, rate_std))
            feature_names.append(f"ch{ch}_firing_rate_std")

        # 3. 添加不同类别细胞发放率特征（核心补充部分）
        # 遍历所有通道的细胞类型发放率
        ALL_CELL_TYPES=['p','i']

        # cell_fr_channels = sorted(data['special_cell_fr'].keys())
        for ele in ALL_CH_INDEX:
            cell_type_dict = data['special_cell_fr'].get(ele, {})
            for cell_type in ALL_CELL_TYPES:
                fr_data = cell_type_dict.get(cell_type, [])

                if len(fr_data) > 0:
                    # 插值对齐到基准时间长度
                    fr_aligned = np.interp(
                        np.linspace(0, 1, min_time_len),
                        np.linspace(0, 1, len(fr_data)),
                        fr_data
                    )
                else:
                    fr_aligned = np.zeros(min_time_len)

                feature_list.append(fr_aligned)
                feature_names.append(f"ch{ele}_{cell_type}_firing_rate")
                if verbose:
                    print(f"✓ 已添加通道{ele} {cell_type} 发放率特征 (长度: {len(fr_aligned)})")


        # 构建特征矩阵 (n_time_points, n_features)
        if feature_list:
            data['feature_matrix'] = np.column_stack(feature_list)
            data['feature_names'] = feature_names
            if verbose:
                print(f"✓ 特征矩阵构建完成: {data['feature_matrix'].shape} (时间点 × 特征数)")
                print(f"✓ 特征列表: {', '.join(feature_names[:5])}...")  # 打印前5个特征名
        else:
            print(f"⚠️ 特征矩阵为空（无有效LFP/Spike数据）")

    if verbose:
        total_spikes = sum(u.get('n_spikes', 0) for u in data['spikes'].values())
        print(f"\n✅ 加载完成！总Spike数: {total_spikes}")

    return data


def save_lfp_npy(save_dir, save_name, data: np.ndarray):
    """保存LFP数据为npy文件"""
    save_path = os.path.join(save_dir, save_name + ".npy")
    os.makedirs(save_dir, exist_ok=True)
    np.save(save_path, data)
    print(f"💾 LFP已保存至: {save_path}")


def save_feature_matrix(data: Dict, save_dir: str):
    """保存特征矩阵到文件"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    # 保存特征矩阵
    if data['feature_matrix'] is not None:
        np.save(save_dir / f"{data['session']}_feature_matrix.npy", data['feature_matrix'])
        # 保存特征名
        with open(save_dir / f"{data['session']}_feature_names.txt", 'w') as f:
            f.write('\n'.join(data['feature_names']))
        print(f"💾 特征矩阵已保存至: {save_dir}/{data['session']}_feature_matrix.npy")
    else:
        print(f"⚠️ 无特征矩阵可保存")


if __name__ == '__main__':
    '''
    从raw数据直接到fea_matrix
    config：session dir
            csv dir (for sorting cells)
    '''
    # 配置参数
    BASE_PATH = "E:/crcns_hc3"  # 修改为你的数据集路径
    TOPDIR = "ec013.27"
    SESSION = "ec013.388"
    CSV_FILE = r"E:\crcns-hc3-metadata-tables\hc3-metadata-tables\hc3-cell.csv"

    # 加载数据并提取特征
    data = load_hc3_session_full(
        base_path=BASE_PATH,
        topdir=TOPDIR,
        session=SESSION,
        CSV_FILE=CSV_FILE,
        load_spikes=True,
        load_lfp=True,  # 需开启LFP加载
        load_position=False,
        save_lfp=False,
        verbose=False
    )

    # 打印关键信息
    print(f"\n=== 样本 {SESSION} 特征汇总 ===")
    print(f"1. 神经元数量: {len(data['spikes'])}")
    print(f"3. LFP theta功率形状: {data['lfp_features']['avg_theta_power'].shape}")
    print(f"4. LFP gamma功率形状: {data['lfp_features']['avg_gamma_power'].shape}")
    print(f"5. 特征矩阵形状: {data['feature_matrix'].shape if data['feature_matrix'] is not None else 'None'}")

    # 保存特征矩阵
    save_feature_matrix(data, save_dir=Path(BASE_PATH) / TOPDIR / SESSION)