from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from .data_parser import DataParserAgent
from .detection_agent import SpoofingDetectionAgent
from .figure_agent import FigureAgent
from .llm import LLMClient
from .report_agent import ReportAgent
from .schema import WorkflowResult
from .thesis_qa_agent import ThesisQAAgent


class MultiAgentOrchestrator:
    """Coordinates deterministic agents + optional LLM agent.

    Logic chain:
    DataParserAgent -> SpoofingDetectionAgent -> FigureAgent -> ThesisQAAgent -> ReportAgent
    """

    def __init__(self, out_dir: str | Path = "outputs", llm_model: str | None = None):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.llm = LLMClient(model=llm_model)
        self.parser = DataParserAgent()
        self.detector = SpoofingDetectionAgent()
        self.figure_agent = FigureAgent()
        self.thesis_agent = ThesisQAAgent(self.llm)
        self.report_agent = ReportAgent(self.llm)

    def run(
        self,
        gnss_path: str | Path | None = None,
        thesis_path: str | Path | None = None,
        column_map: Optional[Dict[str, str]] = None,
        baseline_fraction: float = 0.2,
        threshold: float = 3.0,
        window_seconds: float = 5.0,
        enable_llm: bool = True,
    ) -> WorkflowResult:
        gnss_result = None
        thesis_result = None

        if gnss_path:
            raw = self.parser.read_table(gnss_path)
            inferred_map = self.parser.infer_columns(raw, column_map)
            normalized, parse_issues = self.parser.normalize(raw, inferred_map)
            gnss_result = self.detector.analyze(
                normalized,
                original_rows=len(raw),
                column_map=inferred_map,
                baseline_fraction=baseline_fraction,
                threshold=threshold,
                window_seconds=window_seconds,
            )
            gnss_result.issues = parse_issues + gnss_result.issues
            gnss_result = self.figure_agent.make_figures(gnss_result, self.out_dir / "figures")
            gnss_result.metric_frame.to_csv(self.out_dir / "gnss_metrics.csv", index=False, encoding="utf-8-sig")
            gnss_result.satellite_summary.to_csv(self.out_dir / "satellite_summary.csv", index=False, encoding="utf-8-sig")
            ws = gnss_result.event_summary.get("window_summary")
            if ws is not None:
                ws.to_csv(self.out_dir / "window_summary.csv", index=False, encoding="utf-8-sig")

        if thesis_path:
            thesis_result = self.thesis_agent.check(thesis_path, enable_llm=enable_llm)

        return self.report_agent.build_report(gnss_result, thesis_result, self.out_dir, polish_with_llm=enable_llm)
