from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class AgentIssue:
    agent: str
    level: str  # info | warning | error
    location: str
    message: str
    suggestion: str = ""


@dataclass
class GNSSAnalysisResult:
    original_rows: int
    cleaned_rows: int
    column_map: Dict[str, str]
    metric_frame: pd.DataFrame
    satellite_summary: pd.DataFrame
    event_summary: Dict[str, Any]
    roc_summary: Optional[Dict[str, float]] = None
    figures: List[Path] = field(default_factory=list)
    issues: List[AgentIssue] = field(default_factory=list)


@dataclass
class ThesisQAResult:
    paragraphs: int
    issues: List[AgentIssue]
    llm_review: str = ""


@dataclass
class WorkflowResult:
    gnss_result: Optional[GNSSAnalysisResult] = None
    thesis_result: Optional[ThesisQAResult] = None
    report_markdown: str = ""
    report_path: Optional[Path] = None
