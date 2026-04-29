from __future__ import annotations

import argparse
from pathlib import Path

from agents.orchestrator import MultiAgentOrchestrator
from sample_data import generate_sample


def main() -> None:
    parser = argparse.ArgumentParser(description="GNSS 欺骗检测实验分析与论文质检多 Agent 系统")
    parser.add_argument("--gnss", type=str, default="", help="GNSS 实验数据文件：csv/xlsx/xls/txt/log")
    parser.add_argument("--thesis", type=str, default="", help="论文文件：docx/txt/md")
    parser.add_argument("--out", type=str, default="outputs", help="输出目录")
    parser.add_argument("--baseline-fraction", type=float, default=0.2, help="前多少比例作为正常基线")
    parser.add_argument("--threshold", type=float, default=3.0, help="PCS 告警阈值")
    parser.add_argument("--window-seconds", type=float, default=5.0, help="窗口化检测概率统计秒数")
    parser.add_argument("--demo", action="store_true", help="生成并使用一份模拟 GNSS 数据")
    parser.add_argument("--no-llm", action="store_true", help="关闭 LLM 审阅/润色")
    args = parser.parse_args()

    gnss_path = args.gnss
    if args.demo:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        gnss_path = str(generate_sample(out_dir / "sample_gnss_data.csv"))
        print(f"已生成模拟数据：{gnss_path}")

    if not gnss_path and not args.thesis:
        raise SystemExit("请提供 --gnss 或 --thesis，或者使用 --demo。")

    orchestrator = MultiAgentOrchestrator(out_dir=args.out)
    result = orchestrator.run(
        gnss_path=gnss_path or None,
        thesis_path=args.thesis or None,
        baseline_fraction=args.baseline_fraction,
        threshold=args.threshold,
        window_seconds=args.window_seconds,
        enable_llm=not args.no_llm,
    )
    print("\n========== 报告预览 ==========")
    print(result.report_markdown[:3000])
    print("\n========== 输出文件 ==========")
    print(result.report_path)


if __name__ == "__main__":
    main()
