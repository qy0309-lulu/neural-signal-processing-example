import os
import tarfile
import sqlite3
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import numpy as np


class HC3Loader:
    """
    CRCNS hc3 数据集加载器
    """

    def __init__(self, base_path: str):
        """
        base_path: hc3 数据集根目录（包含 crcns-hc3-metadata-tables.zip 解压后的文件）
        """
        self.base_path = Path(base_path)
        self.metadata_dir = self.base_path / "crcns-hc3-metadata-tables"

        # 加载元数据
        self._load_metadata()

    def _load_metadata(self):
        """加载 SQLite 或 CSV 元数据"""
        db_path = self.metadata_dir / "hc3-tables.db"

        if db_path.exists():
            self.conn = sqlite3.connect(db_path)
            print("✅ 已连接 SQLite 元数据数据库")
        else:
            self.conn = None
            print("⚠️ 未找到 SQLite 数据库，将使用 CSV")

    def get_sessions(self,
                     behavior: Optional[str] = None,
                     animal: Optional[str] = None,
                     regions: Optional[List[str]] = None,
                     min_duration: float = 0) -> pd.DataFrame:
        """查询符合条件的 session"""
        if self.conn is not None:
            query = "SELECT * FROM session"
            conditions = []
            params = []

            if behavior:
                conditions.append("behavior = ?")
                params.append(behavior)
            if animal:
                conditions.append("topdir LIKE ?")
                params.append(f"{animal}%")
            if min_duration > 0:
                conditions.append("duration >= ?")
                params.append(min_duration)

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            df = pd.read_sql_query(query, self.conn, params=params)
        else:
            #  fallback 到 CSV
            df = pd.read_csv(self.metadata_dir / "hc3-session.csv")
            if behavior:
                df = df[df['behavior'] == behavior]
            # ... 其他过滤类似

        # 可进一步按 region 过滤（需 join cell 表）
        return df

    def get_cells(self,
                  region: Optional[str] = None,
                  cell_type: Optional[str] = None,  # 'p' 或 'i'
                  animal: Optional[str] = None) -> pd.DataFrame:
        """获取细胞信息"""
        if self.conn is not None:
            query = "SELECT * FROM cell"
            conditions = []
            params = []

            if region:
                conditions.append("region = ?")
                params.append(region)
            if cell_type:
                conditions.append("cellType = ?")
                params.append(cell_type)
            if animal:
                conditions.append("animal = ?")
                params.append(animal)

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            return pd.read_sql_query(query, self.conn, params=params)
        else:
            return pd.read_csv(self.metadata_dir / "hc3-cell.csv")

    def load_session_data(self,
                          topdir: str,
                          session: str,
                          electrodes: Optional[List[int]] = None,
                          load_spikes: bool = True,
                          load_lfp: bool = False,
                          load_position: bool = True) -> Dict:
        """
        加载单个 session 的数据
        """
        session_dir = self.base_path / topdir
        tar_path = session_dir / f"{session}.tar.gz"

        if not tar_path.exists():
            raise FileNotFoundError(f"未找到 {tar_path}")

        data = {"topdir": topdir, "session": session, "files": {}}

        with tarfile.open(tar_path, "r:gz") as tar:
            members = tar.getmembers()

            # 加载位置信息
            if load_position:
                whl_files = [m for m in members if m.name.endswith('.whl')]
                if whl_files:
                    f = tar.extractfile(whl_files[0])
                    pos = pd.read_csv(f, sep=' ', header=None,
                                      names=['x', 'y', 'hd', 'speed'] if len(f.readline().split()) == 4 else None)
                    data['position'] = pos

            # 加载 spike 数据
            if load_spikes:
                data['spikes'] = self._load_spikes_from_tar(tar, electrodes)

            # 加载 LFP (eeg)
            if load_lfp:
                eeg_files = [m for m in members if m.name.endswith('.eeg')]
                if eeg_files:
                    # .eeg 是 1.25kHz 二进制数据，需根据 .xml 配置解析
                    data['eeg_info'] = "LFP 数据需使用 NeuroScope 或自定义读取器"

        return data

    def _load_spikes_from_tar(self,
                              tar: tarfile.TarFile,
                              electrodes: Optional[List[int]] = None) -> Dict:
        """从 tar 中读取 spike 数据（.res, .clu, .spk, .fet）"""
        spikes = {}

        clu_files = [m for m in tar.getmembers() if m.name.endswith('.clu')]

        for member in clu_files:
            ele = int(member.name.split('.')[-1])  # .clu.N 中的 N
            if electrodes and ele not in electrodes:
                continue

            # 读取 cluster
            f_clu = tar.extractfile(member)
            clu_data = np.loadtxt(f_clu, dtype=int, skiprows=1)

            # 读取时间
            res_name = member.name.replace('.clu.', '.res.')
            f_res = tar.extractfile(res_name)
            times = np.loadtxt(f_res, dtype=int)

            # 按 cluster 分组
            unique_clu = np.unique(clu_data)
            for c in unique_clu:
                if c <= 1:  # 0=噪声, 1=unsortable
                    continue
                mask = clu_data == c
                spikes[f"ele{ele}_clu{c}"] = {
                    'times': times[mask],
                    'cluster': c,
                    'electrode': ele
                }

        return spikes

    def load_cell_spikes(self, cell_id: int) -> Dict:
        """根据 cell 表中的 id 加载该细胞在所有 session 中的 spike"""
        if self.conn is None:
            raise NotImplementedError("需 SQLite 支持")

        # 查询该细胞属于哪个 topdir / ele / clu
        cell_info = pd.read_sql_query(
            "SELECT topdir, animal, ele, clu FROM cell WHERE id = ?",
            self.conn, params=(cell_id,)
        ).iloc[0]

        # 查询该细胞出现过的所有 session
        spike_counts = pd.read_sql_query(
            "SELECT s.session, k.nSpikes FROM spike_count k "
            "JOIN session s ON k.sessId = s.id "
            "WHERE k.cellId = ?", self.conn, params=(cell_id,)
        )

        return {
            'cell_id': cell_id,
            'info': cell_info.to_dict(),
            'sessions': spike_counts.to_dict('records')
        }


# ====================== 使用示例 ======================

if __name__ == "__main__":
    # 请替换为实际路径
    loader = HC3Loader(r"E:\ec012ec\ec012ec.189\ec012ec.11\ec012ec.189")

    # 1. 查询 session
    sessions = loader.get_sessions(behavior="bigSquare", min_duration=600)
    print(sessions.head())

    # 2. 查询细胞
    ca1_cells = loader.get_cells(region="CA1", cell_type="p")
    print(f"CA1 pyramidal cells: {len(ca1_cells)}")

    # 3. 加载单个 session 数据
    data = loader.load_session_data(
        topdir="ec013.51",
        session="ec013.911",
        load_spikes=True,
        load_position=True
    )

    print(f"Session {data['session']} 包含 {len(data.get('spikes', {}))} 个 unit")