from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from .llm import LLMClient
from .schema import AgentIssue, GNSSAnalysisResult, ThesisQAResult, WorkflowResult


class ReportAgent:
    name = "ReportAgent"

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or LLMClient()

    def build_report(
        self,
        gnss_result: GNSSAnalysisResult | None,
        thesis_result: ThesisQAResult | None,
        out_dir: str | Path,
        polish_with_llm: bool = True,
    ) -> WorkflowResult:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        md = []
        md.append("# GNSS 欺骗检测实验分析与论文质检多 Agent 报告\n")

        if gnss_result is not None:
            md.extend(self._gnss_section(gnss_result))
        if thesis_result is not None:
            md.extend(self._thesis_section(thesis_result))

        report = "\n".join(md)
        if polish_with_llm and self.llm.enabled:
            polished = self.llm.complete(
                "你是中文科研报告润色助手。在不改变数据含义的前提下，提高报告表达的正式性。保留 Markdown 结构。",
                report,
                max_chars=12000,
            )
            if polished and not polished.startswith("[LLM 调用失败"):
                report = polished

        path = out_dir / "multi_agent_report.md"
        path.write_text(report, encoding="utf-8")
        return WorkflowResult(gnss_result=gnss_result, thesis_result=thesis_result, report_markdown=report, report_path=path)

    def _gnss_section(self, r: GNSSAnalysisResult) -> List[str]:
        ev = r.event_summary or {}
        lines = ["\n## 一、GNSS 欺骗检测实验分析", ""]
        lines.append(f"- 原始记录数：{r.original_rows}")
        lines.append(f"- 清洗后记录数：{r.cleaned_rows}")
        lines.append(f"- 卫星数量：{ev.get('satellite_count', 'NA')}")
        lines.append(f"- 使用检测分量：{', '.join(ev.get('components', [])) or '未计算'}")
        lines.append(f"- 基线区间截止时刻：{ev.get('baseline_end', float('nan')):.3f} s")
        lines.append(f"- 总体告警比例：{ev.get('total_alert_rate', float('nan')):.3f}")
        lines.append(f"- 疑似异常最强窗口起始时刻：{ev.get('likely_event_time', float('nan')):.3f} s")
        lines.append(f"- 该窗口平均检测概率：{ev.get('likely_event_probability', float('nan')):.3f}")
        lines.append(f"- 该窗口平均异常分数：{ev.get('likely_event_score', float('nan')):.3f}")
        if r.roc_summary:
            lines.append(
                f"- ROC AUC：{r.roc_summary['auc']:.3f}；建议阈值：{r.roc_summary['best_threshold']:.3f}；"
                f"对应检测率：{r.roc_summary['best_tpr']:.3f}；虚警率：{r.roc_summary['best_fpr']:.3f}"
            )
        lines.append("")
        lines.append("### 1.1 可疑卫星排序")
        if not r.satellite_summary.empty:
            show = r.satellite_summary.head(10).copy()
            lines.append(show.to_markdown(index=False))
        else:
            lines.append("未生成卫星级统计。")
        lines.append("")
        lines.append("### 1.2 自动生成图表")
        if r.figures:
            for fig in r.figures:
                lines.append(f"- {fig.name}")
        else:
            lines.append("未生成图表。")
        lines.append("")
        lines.append("### 1.3 数据与检测流程问题")
        lines.extend(self._issue_lines(r.issues))
        return lines

    def _thesis_section(self, r: ThesisQAResult) -> List[str]:
        lines = ["\n## 二、论文质检结果", ""]
        lines.append(f"- 检测段落数：{r.paragraphs}")
        lines.append(f"- 规则检查问题数：{len(r.issues)}")
        lines.append("")
        lines.append("### 2.1 规则检查清单")
        lines.extend(self._issue_lines(r.issues, limit=80))
        if r.llm_review:
            lines.append("")
            lines.append("### 2.2 LLM 审阅建议")
            lines.append(r.llm_review)
        return lines

    def _issue_lines(self, issues: List[AgentIssue], limit: int = 50) -> List[str]:
        if not issues:
            return ["- 未发现明显问题。"]
        lines = []
        for issue in issues[:limit]:
            sug = f" 建议：{issue.suggestion}" if issue.suggestion else ""
            lines.append(f"- **{issue.level}**｜{issue.agent}｜{issue.location}：{issue.message}{sug}")
        if len(issues) > limit:
            lines.append(f"- 其余 {len(issues) - limit} 条问题已省略，建议在导出的结构化结果中查看。")
        return lines
