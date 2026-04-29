from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from agents.orchestrator import MultiAgentOrchestrator
from sample_data import generate_sample


st.set_page_config(page_title="GNSS 多 Agent 实验分析与论文质检", layout="wide")
st.title("GNSS 欺骗检测实验分析与论文质检多 Agent 系统")
st.caption("数据解析 Agent → 欺骗检测 Agent → 图表 Agent → 论文质检 Agent → 报告 Agent")

with st.sidebar:
    st.header("参数设置")
    baseline_fraction = st.slider("正常基线比例", 0.05, 0.5, 0.2, 0.05)
    threshold = st.slider("PCS 告警阈值", 1.0, 8.0, 3.0, 0.1)
    window_seconds = st.number_input("窗口长度 / s", min_value=1.0, max_value=120.0, value=5.0, step=1.0)
    enable_llm = st.toggle("启用 LLM 审阅/润色（需要 OPENAI_API_KEY）", value=bool(os.getenv("OPENAI_API_KEY")))
    out_dir = st.text_input("输出目录", value="outputs")

col1, col2 = st.columns(2)
with col1:
    st.subheader("1. 上传 GNSS 实验数据")
    gnss_file = st.file_uploader("支持 csv/xlsx/xls/txt/log", type=["csv", "xlsx", "xls", "txt", "log"])
    use_demo = st.button("没有数据？生成模拟数据演示")

with col2:
    st.subheader("2. 上传论文文件")
    thesis_file = st.file_uploader("支持 docx/txt/md", type=["docx", "txt", "md"])

st.divider()
run = st.button("开始运行多 Agent 工作流", type="primary")


def save_upload(uploaded_file, tmp_dir: Path) -> Path:
    path = tmp_dir / uploaded_file.name
    path.write_bytes(uploaded_file.getbuffer())
    return path


if run:
    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        gnss_path = None
        thesis_path = None
        if use_demo or gnss_file is None and thesis_file is None:
            gnss_path = generate_sample(tmp_dir / "sample_gnss_data.csv")
            st.info("已自动生成模拟 GNSS 欺骗数据用于演示。")
        elif gnss_file is not None:
            gnss_path = save_upload(gnss_file, tmp_dir)
        if thesis_file is not None:
            thesis_path = save_upload(thesis_file, tmp_dir)

        orchestrator = MultiAgentOrchestrator(out_dir=out_dir)
        with st.spinner("Agent 正在分析，请稍候..."):
            result = orchestrator.run(
                gnss_path=gnss_path,
                thesis_path=thesis_path,
                baseline_fraction=baseline_fraction,
                threshold=threshold,
                window_seconds=window_seconds,
                enable_llm=enable_llm,
            )

    st.success("分析完成")

    tabs = st.tabs(["报告", "GNSS 指标", "卫星排序", "图表", "论文问题"])
    with tabs[0]:
        st.markdown(result.report_markdown)
        if result.report_path and Path(result.report_path).exists():
            st.download_button(
                "下载 Markdown 报告",
                data=Path(result.report_path).read_text(encoding="utf-8"),
                file_name="multi_agent_report.md",
                mime="text/markdown",
            )
    with tabs[1]:
        if result.gnss_result is not None:
            st.dataframe(result.gnss_result.metric_frame.head(1000), use_container_width=True)
            csv = result.gnss_result.metric_frame.to_csv(index=False, encoding="utf-8-sig")
            st.download_button("下载 GNSS 指标 CSV", csv, "gnss_metrics.csv", "text/csv")
        else:
            st.info("未上传 GNSS 数据。")
    with tabs[2]:
        if result.gnss_result is not None:
            st.dataframe(result.gnss_result.satellite_summary, use_container_width=True)
        else:
            st.info("未上传 GNSS 数据。")
    with tabs[3]:
        if result.gnss_result is not None and result.gnss_result.figures:
            for fig in result.gnss_result.figures:
                st.image(str(fig), caption=fig.name, use_container_width=True)
        else:
            st.info("没有图表。")
    with tabs[4]:
        if result.thesis_result is not None:
            issues = [i.__dict__ for i in result.thesis_result.issues]
            st.dataframe(pd.DataFrame(issues), use_container_width=True)
            if result.thesis_result.llm_review:
                st.markdown("### LLM 审阅建议")
                st.markdown(result.thesis_result.llm_review)
        else:
            st.info("未上传论文文件。")
else:
    st.info("上传 GNSS 数据或论文文件后点击运行。你也可以直接点击演示按钮并运行。")
