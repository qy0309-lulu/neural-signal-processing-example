import os
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d  # 可选：平滑


# def calculate_speed(whl_df: pd.DataFrame,
#                     fs: float = 39.0625,
#                     smoothing_sigma: float = 8.0,
#                     use_head_center: bool = False) -> np.ndarray:
#     """
#     从 .whl 文件计算动物移动速度
#
#     参数:
#         whl_df: 使用 load_whl() 加载后的 DataFrame
#         fs: .whl 采样率，默认 39.0625 Hz
#         smoothing_sigma: 高斯平滑参数（单位：样本），用于减少噪声
#         use_head_center: 是否使用前后LED中心位置（推荐）
#
#     返回:
#         speed: 一维数组，单位为 **cm/s**
#     """
#     if use_head_center:
#         # 使用头部中心位置（更稳定）
#         x = (whl_df['x2'].values + whl_df['x1'].values) * 0.5
#         y = (whl_df['y2'].values + whl_df['y1'].values) * 0.5
#     else:
#         # 只使用前部 LED（更接近鼻子）
#         x = whl_df['x1'].values
#         y = whl_df['y1'].values
#
#     # 处理 NaN（跟踪丢失）
#     valid = ~(np.isnan(x) | np.isnan(y))
#     x = np.interp(np.arange(len(x)), np.where(valid)[0], x[valid]) if np.any(valid) else x
#     y = np.interp(np.arange(len(y)), np.where(valid)[0], y[valid]) if np.any(valid) else y
#
#     # 计算位移
#     dx = np.diff(x)
#     dy = np.diff(y)
#     distance = np.sqrt(dx ** 2 + dy ** 2)  # 单位：cm
#
#     # 计算瞬时速度 (cm/s)
#     dt = 1.0 / fs
#     speed = distance / dt
#
#     # 在开头补一个值，使长度与原始数据一致
#     speed = np.concatenate(([speed[0]], speed))
#
#     # 可选：高斯平滑减少噪声
#     if smoothing_sigma > 0:
#         speed = gaussian_filter1d(speed, sigma=smoothing_sigma)
#
#     return speed

def calculate_speed(whl_df: pd.DataFrame,
                    fs: float = 39.0625,
                    smoothing_sigma: float = 8.0,
                    use_head_center: bool = False) -> np.ndarray:
    """
    从 .whl 文件计算动物移动速度 + 运动方向角度
    返回：二维数组 [n_samples, 2]，每一行是 (速度大小cm/s, 运动方向弧度)
    """
    if use_head_center:
        x = (whl_df['x2'].values + whl_df['x1'].values) * 0.5
        y = (whl_df['y2'].values + whl_df['y1'].values) * 0.5
    else:
        x = whl_df['x1'].values
        y = whl_df['y1'].values

    # 处理 NaN（跟踪丢失）
    valid = ~(np.isnan(x) | np.isnan(y))
    x = np.interp(np.arange(len(x)), np.where(valid)[0], x[valid]) if np.any(valid) else x
    y = np.interp(np.arange(len(y)), np.where(valid)[0], y[valid]) if np.any(valid) else y

    # 计算位移
    dx = np.diff(x)
    dy = np.diff(y)

    # ===================== 速度大小 =====================
    distance = np.sqrt(dx ** 2 + dy ** 2)
    dt = 1.0 / fs
    speed_magnitude = distance / dt
    speed_magnitude = np.concatenate(([speed_magnitude[0]], speed_magnitude))

    # ===================== 运动方向（角度）=====================
    # 计算方向（弧度），dx/dy=0 时保持上一方向
    direction = np.arctan2(dy, dx)  # 范围 [-π, π]
    direction = np.concatenate(([direction[0]], direction))

    # ===================== 平滑（同时平滑速度和方向）=====================
    if smoothing_sigma > 0:
        speed_magnitude = gaussian_filter1d(speed_magnitude, sigma=smoothing_sigma)
        direction = gaussian_filter1d(direction, sigma=smoothing_sigma)

    # 返回二维向量：[速度大小, 方向角度]
    # return np.column_stack([speed_magnitude, direction])
    return np.column_stack(speed_magnitude)

def calculate_head_pos(whl_df: pd.DataFrame):
    """
    从 .whl 文件计算动物移动速度 + 运动方向角度
    返回：中心坐标
    """

    x = (whl_df['x2'] + whl_df['x1']) * 0.5
    y = (whl_df['y2'] + whl_df['y1']) * 0.5

    return x.values, y.values


if __name__ == '__main__':

    # 修改数据路径 -> save——path与之对应
    base_path = "E:/crcns_hc3"
    base_path = Path(base_path)
    topdir = "ec013.28"
    session = "ec013.410"
    tar_path = base_path / topdir / f"{session}.tar.gz"

    with tarfile.open(tar_path, 'r:gz') as tar:
        members = {m.name: m for m in tar.getmembers()}

        whl_files = [name for name in members if name.endswith('.whl')]
        # if whl_files:
        #     with tar.extractfile(whl_files[0]) as f:
        #         pos = pd.read_csv(f, sep=r'\s+', header=None, engine='python')
        #
        #         cols = ['x1', 'y1', 'x2', 'y2']
        #         pos.columns = cols
        #         print(f"✓ 位置数据: {len(pos)} 个时间点")
        #
        #         speed_and_angle = calculate_speed(pos)
        #
        #         speed = speed_and_angle[:, 0]  # 速度大小
        #         angle = speed_and_angle[:, 1]  # 运动方向（弧度）
        #         pos['speed'] = speed
        #         pos['angle'] = angle
        #         print(f"pos数据形状：{pos.shape}")
        # else:
        #     print(f"{whl_files}don't found! \n")

        if whl_files:
            with tar.extractfile(whl_files[0]) as f:
                    pos = pd.read_csv(f, sep=r'\s+', header=None, engine='python')

                    cols = ['x1', 'y1', 'x2', 'y2']
                    pos.columns = cols
                    print(f"✓ 位置数据: {len(pos)} 个时间点")

                    head_pos = calculate_speed(pos)

                    pos['speed'] = head_pos
                    print(f"head_pos数据形状：{head_pos.shape}")
        else:
            print(f"{whl_files}don't found! \n")

        save_dir = base_path / topdir / session
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        save_path = save_dir / f"{session}_speed.csv"
        np.savetxt(save_path, pos, delimiter=',', fmt='%.3f')

        print(f"csv文件保存，在 {save_path}")

