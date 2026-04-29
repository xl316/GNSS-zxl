from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve

from .schema import AgentIssue, GNSSAnalysisResult


EPS = 1e-9


def _mad_scale(x: pd.Series) -> float:
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if not np.isfinite(mad) or mad < EPS:
        std = np.nanstd(x)
        return float(std if std > EPS else 1.0)
    return float(1.4826 * mad)


def _robust_z(series: pd.Series, baseline_mask: pd.Series) -> pd.Series:
    base = series[baseline_mask & series.notna()]
    if len(base) < 5:
        base = series[series.notna()]
    if len(base) == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    med = float(np.nanmedian(base))
    scale = _mad_scale(base)
    return (series - med) / (scale + EPS)


class SpoofingDetectionAgent:
    name = "SpoofingDetectionAgent"

    def analyze(
        self,
        df: pd.DataFrame,
        original_rows: int,
        column_map: Dict[str, str],
        baseline_fraction: float = 0.2,
        threshold: float = 3.0,
        window_seconds: float = 5.0,
    ) -> GNSSAnalysisResult:
        issues: List[AgentIssue] = []
        if df.empty:
            return GNSSAnalysisResult(
                original_rows=original_rows,
                cleaned_rows=0,
                column_map=column_map,
                metric_frame=df,
                satellite_summary=pd.DataFrame(),
                event_summary={},
                issues=[AgentIssue(self.name, "error", "输入数据", "清洗后的数据为空。")],
            )

        metric = df.copy()
        t_min, t_max = float(metric["time"].min()), float(metric["time"].max())
        baseline_end = t_min + max((t_max - t_min) * baseline_fraction, EPS)
        metric["baseline"] = metric["time"] <= baseline_end

        # 1. Correlator SQM metrics.
        if set(["early", "prompt", "late"]).issubset(metric.columns):
            e = metric["early"].abs()
            p = metric["prompt"].abs().replace(0, np.nan)
            l = metric["late"].abs()
            metric["ratio"] = (e + l) / (2.0 * p + EPS)
            metric["delta"] = (e - l) / (p + EPS)
            metric["corr_power"] = p
        else:
            issues.append(
                AgentIssue(
                    self.name,
                    "warning",
                    "相关器指标",
                    "未同时检测到 early/prompt/late 三列，PR/Ratio/Delta 类 SQM 指标会被跳过。",
                    "若数据来自 GNSS-SDR 跟踪环路，建议导出 Early、Prompt、Late 相关器输出。",
                )
            )

        # 2. Pseudorange derivative and Doppler residual proxy.
        if "pseudorange" in metric.columns:
            metric["pr_rate"] = metric.groupby("sat")["pseudorange"].diff() / metric.groupby("sat")["time"].diff().replace(0, np.nan)
            metric["pr_jump"] = metric.groupby("sat")["pseudorange"].diff().abs()
        if "doppler" in metric.columns:
            metric["doppler_rate"] = metric.groupby("sat")["doppler"].diff() / metric.groupby("sat")["time"].diff().replace(0, np.nan)

        # 3. Build normalized anomaly score by robust baseline statistics.
        score_components = []
        component_names = []
        for c in ["cn0", "ratio", "delta", "corr_power", "pr_rate", "pr_jump", "doppler", "doppler_rate"]:
            if c in metric.columns:
                metric[f"z_{c}"] = np.nan
                for _, g in metric.groupby("sat"):
                    metric.loc[g.index, f"z_{c}"] = _robust_z(g[c], g["baseline"]).values
                metric[f"abs_z_{c}"] = metric[f"z_{c}"].abs().clip(upper=20)
                score_components.append(metric[f"abs_z_{c}"])
                component_names.append(c)

        if not score_components:
            issues.append(AgentIssue(self.name, "error", "检测统计量", "没有可计算的检测统计量。"))
            metric["pcs_score"] = 0.0
        else:
            score_matrix = pd.concat(score_components, axis=1)
            # PCS: power/quality/range/doppler combined score, higher means more abnormal.
            metric["pcs_score"] = score_matrix.mean(axis=1, skipna=True).fillna(0.0)

        metric["is_alert"] = metric["pcs_score"] >= threshold

        # 4. Window-level detection probability.
        if window_seconds <= 0:
            window_seconds = max((t_max - t_min) / 50.0, 1.0)
        metric["window"] = np.floor((metric["time"] - t_min) / window_seconds).astype(int)
        window_summary = (
            metric.groupby("window")
            .agg(
                time_start=("time", "min"),
                time_end=("time", "max"),
                rows=("sat", "size"),
                alert_rows=("is_alert", "sum"),
                mean_score=("pcs_score", "mean"),
                active_sats=("sat", "nunique"),
            )
            .reset_index()
        )
        window_summary["detection_probability"] = window_summary["alert_rows"] / window_summary["rows"].replace(0, np.nan)

        likely_event = window_summary.sort_values(["mean_score", "detection_probability"], ascending=False).head(1)
        if not likely_event.empty:
            likely_time = float(likely_event.iloc[0]["time_start"])
            likely_prob = float(likely_event.iloc[0]["detection_probability"])
            likely_score = float(likely_event.iloc[0]["mean_score"])
        else:
            likely_time, likely_prob, likely_score = np.nan, np.nan, np.nan

        satellite_summary = (
            metric.groupby("sat")
            .agg(
                rows=("time", "size"),
                alert_rate=("is_alert", "mean"),
                max_score=("pcs_score", "max"),
                mean_score=("pcs_score", "mean"),
            )
            .sort_values("max_score", ascending=False)
            .reset_index()
        )

        roc_summary: Optional[Dict[str, float]] = None
        if "label" in metric.columns and metric["label"].notna().sum() > 5 and metric["label"].nunique(dropna=True) == 2:
            valid = metric[["label", "pcs_score"]].dropna()
            fpr, tpr, thr = roc_curve(valid["label"].astype(int), valid["pcs_score"])
            roc_auc = auc(fpr, tpr)
            # Youden index for suggested threshold
            idx = int(np.argmax(tpr - fpr))
            roc_summary = {
                "auc": float(roc_auc),
                "best_threshold": float(thr[idx]),
                "best_tpr": float(tpr[idx]),
                "best_fpr": float(fpr[idx]),
            }

        event_summary = {
            "time_min": t_min,
            "time_max": t_max,
            "baseline_end": baseline_end,
            "threshold": threshold,
            "window_seconds": window_seconds,
            "components": component_names,
            "likely_event_time": likely_time,
            "likely_event_probability": likely_prob,
            "likely_event_score": likely_score,
            "total_alert_rate": float(metric["is_alert"].mean()),
            "satellite_count": int(metric["sat"].nunique()),
            "window_summary": window_summary,
        }

        return GNSSAnalysisResult(
            original_rows=original_rows,
            cleaned_rows=len(metric),
            column_map=column_map,
            metric_frame=metric,
            satellite_summary=satellite_summary,
            event_summary=event_summary,
            roc_summary=roc_summary,
            figures=[],
            issues=issues,
        )
