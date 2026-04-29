from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .schema import AgentIssue


CANONICAL_CANDIDATES = {
    "time": ["time", "timestamp", "t", "tow", "rx_time", "gpst", "seconds", "sec"],
    "sat": ["sat", "sv", "svid", "prn", "satellite", "sv_id", "svId"],
    "cn0": ["cn0", "c/n0", "cno", "snr", "carrier_to_noise", "cn0_dbhz", "C/N0"],
    "pseudorange": ["pseudorange", "pr", "prmes", "prMes", "rho", "range", "pseudorange_m"],
    "doppler": ["doppler", "doMes", "domes", "carrier_doppler", "doppler_hz", "fd"],
    "early": ["early", "e", "corr_e", "i_e", "E", "IE", "early_corr"],
    "prompt": ["prompt", "p", "corr_p", "i_p", "P", "IP", "prompt_corr"],
    "late": ["late", "l", "corr_l", "i_l", "L", "IL", "late_corr"],
    "label": ["label", "spoofed", "is_spoof", "y", "truth", "attack"],
}


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


class DataParserAgent:
    name = "DataParserAgent"

    def read_table(self, path: str | Path) -> pd.DataFrame:
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix in [".xlsx", ".xls"]:
            return pd.read_excel(path)
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix in [".txt", ".log"]:
            # Try comma, tab, whitespace in sequence.
            for sep in [",", "\t", r"\s+"]:
                try:
                    df = pd.read_csv(path, sep=sep, engine="python")
                    if df.shape[1] >= 2:
                        return df
                except Exception:
                    pass
        raise ValueError(f"不支持的文件类型：{path.suffix}，请使用 csv/xlsx/xls/txt/log。")

    def infer_columns(self, df: pd.DataFrame, user_map: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        user_map = user_map or {}
        existing = {str(c): c for c in df.columns}
        norm_to_col = {_norm(c): c for c in df.columns}
        col_map: Dict[str, str] = {}

        for canonical, manual_col in user_map.items():
            if manual_col and manual_col in existing:
                col_map[canonical] = manual_col

        for canonical, candidates in CANONICAL_CANDIDATES.items():
            if canonical in col_map:
                continue
            for cand in candidates:
                if _norm(cand) in norm_to_col:
                    col_map[canonical] = norm_to_col[_norm(cand)]
                    break
        return col_map

    def normalize(self, df: pd.DataFrame, col_map: Dict[str, str]) -> tuple[pd.DataFrame, list[AgentIssue]]:
        issues: list[AgentIssue] = []
        required = ["time", "sat"]
        for col in required:
            if col not in col_map:
                issues.append(
                    AgentIssue(
                        self.name,
                        "error",
                        "列映射",
                        f"缺少必要列：{col}",
                        "请在原始数据中提供时间列和卫星编号列，或在界面中手动选择列名。",
                    )
                )
        if any(i.level == "error" for i in issues):
            return pd.DataFrame(), issues

        out = pd.DataFrame()
        for canonical, raw_col in col_map.items():
            out[canonical] = df[raw_col]

        out["time"] = pd.to_numeric(out["time"], errors="coerce")
        out["sat"] = out["sat"].astype(str).str.strip()

        numeric_cols = ["cn0", "pseudorange", "doppler", "early", "prompt", "late", "label"]
        for c in numeric_cols:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce")

        before = len(out)
        out = out.dropna(subset=["time", "sat"]).sort_values(["sat", "time"]).reset_index(drop=True)
        after = len(out)
        if after < before:
            issues.append(
                AgentIssue(
                    self.name,
                    "warning",
                    "数据清洗",
                    f"删除了 {before - after} 行缺少 time/sat 的记录。",
                    "建议检查原始文件中是否存在空行、表头重复或非数值时间。",
                )
            )

        # Ensure at least one observable exists.
        observables = [c for c in ["cn0", "pseudorange", "doppler", "early", "prompt", "late"] if c in out.columns]
        if not observables:
            issues.append(
                AgentIssue(
                    self.name,
                    "error",
                    "列映射",
                    "未识别到可用于欺骗检测的观测列。",
                    "至少提供 cn0、pseudorange、doppler，或 early/prompt/late 相关器输出之一。",
                )
            )

        if "label" in out.columns:
            # normalize labels to 0/1 when possible
            out["label"] = out["label"].replace({False: 0, True: 1}).astype(float)
            out.loc[~out["label"].isin([0, 1]), "label"] = np.nan

        return out, issues
