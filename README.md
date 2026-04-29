# GNSS 欺骗检测实验分析与论文质检多 Agent 系统

这个项目是一个可运行的 Python 多 Agent 原型，适合写到“Agent 或 AI 驱动构建的具体成果”里，也可以继续扩展成你的论文实验工具。

## 1. 功能

- **DataParserAgent**：自动读取 csv/xlsx/xls/txt/log，推断 GNSS 实验数据列名。
- **SpoofingDetectionAgent**：计算 C/N0、伪距变化率、多普勒变化率、Early/Prompt/Late 相关器 Ratio/Delta 指标，并融合为 PCS 异常分数。
- **FigureAgent**：自动生成 PCS 曲线、窗口检测概率曲线、C/N0 曲线、Ratio/Delta 曲线、ROC 曲线。
- **ThesisQAAgent**：读取 docx/txt/md 论文文件，检查结构、图表编号、引用编号、口语化表达、中英文空格等问题。
- **ReportAgent**：自动生成 Markdown 报告。
- **可选 LLM**：设置 `OPENAI_API_KEY` 后，自动启用论文审阅和报告润色；不设置也可以完整运行规则分析流程。

## 2. 安装

建议 Python 3.10+。

```bash
cd gnss_multi_agent
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## 3. 运行演示数据

```bash
python run_cli.py --demo --no-llm
```

输出目录默认为：

```text
outputs/
  multi_agent_report.md
  gnss_metrics.csv
  satellite_summary.csv
  window_summary.csv
  figures/*.png
```

## 4. 运行自己的 GNSS 数据

```bash
python run_cli.py --gnss your_data.csv --threshold 3.0 --window-seconds 5 --no-llm
```

推荐数据列名如下，程序也会自动识别常见别名：

| 标准列 | 含义 | 是否必需 |
|---|---|---|
| time | 时间，单位秒 | 必需 |
| sat | 卫星编号，如 G17 | 必需 |
| cn0 | 载噪比 C/N0 | 可选 |
| pseudorange | 伪距，单位米 | 可选 |
| doppler | 多普勒，单位 Hz | 可选 |
| early | Early 相关器输出 | 可选 |
| prompt | Prompt 相关器输出 | 可选 |
| late | Late 相关器输出 | 可选 |
| label | 真实标签，0=正常，1=欺骗 | 可选，有则自动画 ROC |

## 5. 运行论文质检

```bash
python run_cli.py --thesis thesis.docx --no-llm
```

同时分析 GNSS 数据和论文：

```bash
python run_cli.py --gnss your_data.csv --thesis thesis.docx --no-llm
```

## 6. 启动网页界面

```bash
streamlit run app.py
```

浏览器打开后上传实验数据和论文即可。

## 7. 启用 OpenAI LLM 审阅

Windows PowerShell：

```powershell
$env:OPENAI_API_KEY="你的 API Key"
$env:OPENAI_MODEL="gpt-5.5"
streamlit run app.py
```

Linux/macOS：

```bash
export OPENAI_API_KEY="你的 API Key"
export OPENAI_MODEL="gpt-5.5"
streamlit run app.py
```

没有 API Key 时，系统仍会运行确定性 Agent：数据解析、欺骗检测、图表生成和规则质检。

## 8. 可写进申报表的描述

我构建了一个面向 GNSS 欺骗检测研究的科研辅助多 Agent 系统，主要解决实验数据分析复杂、论文图表整理耗时、格式审查容易遗漏等痛点。系统由数据解析 Agent、欺骗检测 Agent、图表生成 Agent、论文质检 Agent 和报告生成 Agent 协同工作。数据解析 Agent 自动读取 GNSS-SDR 或接收机导出的实验数据，并识别时间、卫星编号、载噪比、伪距、多普勒及 Early/Prompt/Late 相关器输出等字段；欺骗检测 Agent 基于基线统计计算 C/N0 异常、伪距跳变、多普勒变化、Ratio 和 Delta 等 SQM 指标，并融合形成 PCS 异常分数；图表 Agent 自动输出检测概率曲线、ROC 曲线和论文可用的对比图；论文质检 Agent 检查错别字风险、图表编号、引用编号、术语统一和中英文排版问题；最终报告 Agent 汇总实验结果、可疑卫星排序和论文修改建议。该系统可将原本需要人工反复整理的实验分析与论文检查流程压缩到数小时内，提高实验复现、图表生成和论文修改效率。
