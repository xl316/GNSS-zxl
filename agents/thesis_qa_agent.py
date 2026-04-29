from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Tuple

from docx import Document

from .llm import LLMClient
from .schema import AgentIssue, ThesisQAResult


class ThesisQAAgent:
    name = "ThesisQAAgent"

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or LLMClient()

    def read_text(self, path: str | Path) -> Tuple[str, List[str]]:
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".docx":
            doc = Document(path)
            paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            return "\n".join(paras), paras
        if suffix in [".txt", ".md"]:
            text = path.read_text(encoding="utf-8", errors="ignore")
            paras = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
            return text, paras
        raise ValueError("论文质检支持 .docx/.txt/.md 文件。")

    def check(self, path: str | Path, enable_llm: bool = True) -> ThesisQAResult:
        text, paras = self.read_text(path)
        issues: List[AgentIssue] = []
        issues.extend(self._check_structure(paras))
        issues.extend(self._check_numbering(paras))
        issues.extend(self._check_common_writing(paras))
        issues.extend(self._check_references(text))

        llm_review = ""
        if enable_llm:
            sample = "\n".join(paras[:80])
            system = (
                "你是硕士论文审稿助手。请只输出问题清单，关注：逻辑连贯性、术语统一、"
                "GNSS欺骗检测表述是否严谨、图表说明是否充分、是否存在口语化表达。"
            )
            llm_review = self.llm.complete(system, sample, max_chars=9000)

        return ThesisQAResult(paragraphs=len(paras), issues=issues, llm_review=llm_review)

    def _check_structure(self, paras: List[str]) -> Iterable[AgentIssue]:
        joined = "\n".join(paras)
        required = ["摘要", "关键词", "绪论", "结论", "参考文献"]
        for word in required:
            if word not in joined:
                yield AgentIssue(
                    self.name,
                    "warning",
                    "全文结构",
                    f"未检测到“{word}”相关标题或内容。",
                    "请确认论文结构是否符合学校模板要求。",
                )

        for i, p in enumerate(paras, start=1):
            if re.match(r"^第[一二三四五六七八九十]+章", p) and len(p) > 30:
                yield AgentIssue(
                    self.name,
                    "warning",
                    f"第{i}段",
                    "章标题过长，可能混入正文。",
                    "建议章标题单独成段，并与正文分开。",
                )

    def _check_numbering(self, paras: List[str]) -> Iterable[AgentIssue]:
        fig_nums = []
        tab_nums = []
        for i, p in enumerate(paras, start=1):
            for m in re.finditer(r"图\s*(\d+)[-.－—](\d+)", p):
                fig_nums.append((i, int(m.group(1)), int(m.group(2)), m.group(0)))
            for m in re.finditer(r"表\s*(\d+)[-.－—](\d+)", p):
                tab_nums.append((i, int(m.group(1)), int(m.group(2)), m.group(0)))

            if "如图" in p and not re.search(r"如图\s*\d+[-.－—]\d+", p):
                yield AgentIssue(
                    self.name,
                    "info",
                    f"第{i}段",
                    "出现“如图”，但没有检测到规范的图号。",
                    "建议使用“如图2-1所示”这类格式。",
                )
            if "如表" in p and not re.search(r"如表\s*\d+[-.－—]\d+", p):
                yield AgentIssue(
                    self.name,
                    "info",
                    f"第{i}段",
                    "出现“如表”，但没有检测到规范的表号。",
                    "建议使用“如表3-2所示”这类格式。",
                )

        for label, nums in [("图", fig_nums), ("表", tab_nums)]:
            by_chapter = {}
            for item in nums:
                _, chapter, idx, raw = item
                by_chapter.setdefault(chapter, []).append((idx, item))
            for chapter, values in by_chapter.items():
                sorted_values = sorted(values)
                expected = 1
                for idx, item in sorted_values:
                    if idx != expected:
                        yield AgentIssue(
                            self.name,
                            "warning",
                            f"第{item[0]}段",
                            f"{label}{chapter}-{idx} 编号可能不连续，预期约为 {label}{chapter}-{expected}。",
                            "请核对图表是否删改后未更新编号或交叉引用。",
                        )
                        expected = idx + 1
                    else:
                        expected += 1

    def _check_common_writing(self, paras: List[str]) -> Iterable[AgentIssue]:
        patterns = [
            (r"\bGNSS\s*欺骗\b", "术语建议统一为“GNSS欺骗”或“GNSS 欺骗”，全文保持一致。"),
            (r"\btoken\b", "中文申报材料中建议统一写为“Token”。"),
            (r"\bAI\b", "建议首次出现时写明“人工智能（AI）”。"),
            (r"本文主要是", "表达偏口语，可改为“本文主要”。"),
            (r"很大程度", "表达略口语，可改为“显著”“较大程度”。"),
            (r"可以看出", "论文中可改为“由图可知”“结果表明”。"),
            (r"大量的", "可视上下文简化为“大量”。"),
            (r"进行(了)?分析", "可考虑压缩表达，如“分析了”。"),
        ]
        for i, p in enumerate(paras, start=1):
            if len(p) > 500:
                yield AgentIssue(
                    self.name,
                    "info",
                    f"第{i}段",
                    "段落较长，可能影响可读性。",
                    "建议拆分为“方法—结果—解释”几段。",
                )
            for pat, sug in patterns:
                if re.search(pat, p, flags=re.IGNORECASE):
                    yield AgentIssue(self.name, "info", f"第{i}段", f"检测到表达：{pat}", sug)
            if re.search(r"[A-Za-z][\u4e00-\u9fa5]|[\u4e00-\u9fa5][A-Za-z]", p):
                yield AgentIssue(
                    self.name,
                    "info",
                    f"第{i}段",
                    "中英文之间可能缺少空格。",
                    "按模板要求决定是否统一为“GNSS 欺骗”这类中英文空格格式。",
                )

    def _check_references(self, text: str) -> Iterable[AgentIssue]:
        citations = re.findall(r"\[(\d+)\]", text)
        if not citations:
            yield AgentIssue(
                self.name,
                "warning",
                "参考文献",
                "全文未检测到形如 [1] 的引用标注。",
                "请确认参考文献引用格式是否符合模板。",
            )
            return
        nums = sorted(set(int(x) for x in citations))
        for expected, actual in enumerate(nums, start=1):
            if actual != expected:
                yield AgentIssue(
                    self.name,
                    "warning",
                    "参考文献",
                    f"引用序号可能不连续：检测到 {actual}，预期 {expected}。",
                    "请检查是否删除文献后未更新引用编号。",
                )
                break
