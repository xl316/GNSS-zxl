from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve

from .schema import AgentIssue, GNSSAnalysisResult


class FigureAgent:
    name = "FigureAgent"

    def make_figures(self, result: GNSSAnalysisResult, out_dir: str | Path) -> GNSSAnalysisResult:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        metric = result.metric_frame
        figures: List[Path] = []

        if metric.empty:
            result.issues.append(AgentIssue(self.name, "error", "绘图", "没有可绘制的数据。"))
            return result

        # 1. PCS score over time.
        fig_path = out_dir / "pcs_score_over_time.png"
        plt.figure(figsize=(11, 5))
        for sat, g in metric.groupby("sat"):
            plt.plot(g["time"], g["pcs_score"], linewidth=0.9, alpha=0.75, label=str(sat))
        plt.axhline(result.event_summary.get("threshold", 3.0), linestyle="--", linewidth=1.0, label="threshold")
        plt.xlabel("Time / s")
        plt.ylabel("PCS anomaly score")
        plt.title("GNSS Spoofing Detection Score by Satellite")
        if metric["sat"].nunique() <= 12:
            plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=220)
        plt.close()
        figures.append(fig_path)

        # 2. Window detection probability.
        window_summary = result.event_summary.get("window_summary")
        if isinstance(window_summary, pd.DataFrame) and not window_summary.empty:
            fig_path = out_dir / "window_detection_probability.png"
            plt.figure(figsize=(11, 4.8))
            plt.plot(window_summary["time_start"], window_summary["detection_probability"], marker="o", linewidth=1.2)
            plt.xlabel("Window start time / s")
            plt.ylabel("Detection probability")
            plt.ylim(-0.02, 1.02)
            plt.title("Windowed Detection Probability")
            plt.tight_layout()
            plt.savefig(fig_path, dpi=220)
            plt.close()
            figures.append(fig_path)

        # 3. CN0 / ratio / delta panels as separate figures for thesis usage.
        for col, ylabel, title in [
            ("cn0", "C/N0 / dB-Hz", "C/N0 Trend"),
            ("ratio", "Ratio", "Correlator Ratio SQM"),
            ("delta", "Delta", "Correlator Delta SQM"),
        ]:
            if col in metric.columns:
                fig_path = out_dir / f"{col}_trend.png"
                plt.figure(figsize=(11, 4.8))
                for sat, g in metric.groupby("sat"):
                    plt.plot(g["time"], g[col], linewidth=0.9, alpha=0.75, label=str(sat))
                plt.xlabel("Time / s")
                plt.ylabel(ylabel)
                plt.title(title)
                if metric["sat"].nunique() <= 12:
                    plt.legend(ncol=2, fontsize=8)
                plt.tight_layout()
                plt.savefig(fig_path, dpi=220)
                plt.close()
                figures.append(fig_path)

        # 4. ROC curve if labels exist.
        if result.roc_summary and "label" in metric.columns:
            valid = metric[["label", "pcs_score"]].dropna()
            if valid["label"].nunique() == 2:
                fpr, tpr, _ = roc_curve(valid["label"].astype(int), valid["pcs_score"])
                fig_path = out_dir / "roc_curve.png"
                plt.figure(figsize=(6.5, 5.5))
                plt.plot(fpr, tpr, linewidth=1.4, label=f"AUC={result.roc_summary['auc']:.3f}")
                plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0)
                plt.xlabel("False alarm probability")
                plt.ylabel("Detection probability")
                plt.title("ROC Curve")
                plt.legend()
                plt.tight_layout()
                plt.savefig(fig_path, dpi=220)
                plt.close()
                figures.append(fig_path)

        result.figures = figures
        return result
