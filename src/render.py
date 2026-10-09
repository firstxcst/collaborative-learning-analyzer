"""把协作报告渲染成**完全自包含**的 HTML 页面。

设计约束：
* **不引用任何外部资源**（无 CDN、无外部字体、无 JS 依赖）——
  研究/教学场景经常在内网或离线环境使用，任何外链都会让它变成一张白纸。
* 只做展示，不做判定：不可用的指标显示为「不可用」，绝不画成 0。
* 页面顶部固定声明数据完整性状态，避免读者把缺失模态当作"表现差"。

用法::

    cla demo --html report.html
    python -c "from collaborative_learning_analyzer.render import render_report; ..."
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any, List, Optional, Union

try:  # 作为包导入
    from .data_models import CollaborationLevel, GroupCollaborationReport
except ImportError:  # 独立运行
    from data_models import CollaborationLevel, GroupCollaborationReport

__all__ = ["render_report", "render_html"]

_LEVEL_TEXT = {
    CollaborationLevel.EXCELLENT: "协作优秀",
    CollaborationLevel.GOOD: "协作良好",
    CollaborationLevel.FAIR: "协作一般",
    CollaborationLevel.POOR: "协作较差",
    CollaborationLevel.CRITICAL: "协作严重不足",
}

#: 分数刻度色带（与等级区间一致），使用固定色值以便离线渲染
_LEVEL_COLORS = {
    CollaborationLevel.CRITICAL: "#A32D2D",
    CollaborationLevel.POOR: "#D85A30",
    CollaborationLevel.FAIR: "#BA7517",
    CollaborationLevel.GOOD: "#378ADD",
    CollaborationLevel.EXCELLENT: "#1D9E75",
}


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _bar(label: str, value: Optional[float], hint: str = "") -> str:
    """渲染一个 0-1 指标条；``None`` 显示为不可用，不画成 0。"""
    if value is None:
        return (
            f'<div class="metric"><div class="metric-head">'
            f"<span>{_esc(label)}</span><span class=\"muted\">不可用</span></div>"
            f'<div class="track"><div class="fill unavailable"></div></div>'
            f'<div class="hint">{_esc(hint or "该模态未产出有效数据")}</div></div>'
        )
    pct = max(0.0, min(1.0, float(value))) * 100
    return (
        f'<div class="metric"><div class="metric-head">'
        f"<span>{_esc(label)}</span><span class=\"value\">{pct:.0f}%</span></div>"
        f'<div class="track"><div class="fill" style="width:{pct:.1f}%"></div></div>'
        f'<div class="hint">{_esc(hint)}</div></div>'
    )


def _contributions_table(report: GroupCollaborationReport) -> str:
    if not report.individual_contributions:
        return '<p class="muted">没有可用的逐人贡献度数据。</p>'

    rows: List[str] = []
    for c in report.individual_contributions:
        scored = c.diagnosis.get("details", {}).get("scored_dimensions", [])
        notes: List[str] = []
        for weakness in c.diagnosis.get("weaknesses", []):
            notes.append(f'<span class="tag warn">{_esc(weakness)}</span>')
        for strength in c.diagnosis.get("strengths", []):
            notes.append(f'<span class="tag ok">{_esc(strength)}</span>')
        if "nonverbal" not in scored:
            notes.append('<span class="tag muted">非语言维度未参与计分</span>')
        if "semantic" not in scored:
            notes.append('<span class="tag muted">语义维度未参与计分</span>')
        rows.append(
            "<tr>"
            f'<td class="who">{_esc(c.person_id)}</td>'
            f"<td>{c.speaking_seconds:.1f}s</td>"
            f"<td>{c.speaking_share:.0%}</td>"
            f"<td>{c.turns}</td>"
            f'<td class="num">{c.total_score:.3f}</td>'
            f'<td class="notes">{" ".join(notes) or "—"}</td>'
            "</tr>"
        )

    return (
        "<table>"
        "<thead><tr>"
        "<th>成员</th><th>发言时长</th><th>发言份额</th><th>发言轮次</th>"
        "<th>贡献度</th><th>诊断</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


def _list_block(title: str, items: List[str], css: str = "") -> str:
    if not items:
        return ""
    body = "".join(f"<li>{_esc(item)}</li>" for item in items)
    return f'<div class="block {css}"><h3>{_esc(title)}</h3><ul>{body}</ul></div>'


def render_html(report: GroupCollaborationReport) -> str:
    """生成自包含 HTML 字符串。"""
    level = report.health_level
    color = _LEVEL_COLORS.get(level, "#5F5E5A")
    completeness = report.data_completeness or {}
    missing = [k for k, v in completeness.items() if not v]

    completeness_chips = "".join(
        f'<span class="chip {"on" if v else "off"}">{_esc(k)} '
        f'{"已分析" if v else "未使用"}</span>'
        for k, v in completeness.items()
    )

    metrics = "".join(
        [
            _bar(
                "参与活跃度",
                report.participation_activity,
                "组内实际发言量相对目标发言量的完成度",
            ),
            _bar(
                "发言均衡度",
                report.evenness,
                "发言份额的均匀程度：完全均等 100%，一人独占 0%",
            ),
            _bar("主题相关度", report.topic_relevance, "需要语义分析结果"),
            _bar(
                "交互深度",
                report.interaction_depth,
                "观点碰撞频率 + 论证深度 + 共识质量",
            ),
        ]
    )

    missing_note = ""
    if missing:
        missing_note = (
            '<p class="note warn"><strong>注意：</strong>以下模态没有产出有效数据，'
            f'相关维度<strong>未参与计分</strong>，不要把它们读作“表现差”：'
            f'{_esc("、".join(missing))}。这些维度在下方指标条中显示为“不可用”。</p>'
        )

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>协作分析报告 · {_esc(report.group_id)}</title>
<style>
  :root {{ --ink:#1F1F1F; --muted:#6B6B6B; --line:#E3E3E3; --bg:#FFFFFF; --soft:#F7F7F5; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; padding:32px 20px 56px; background:var(--bg); color:var(--ink);
         font:14px/1.6 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif; }}
  .wrap {{ max-width:920px; margin:0 auto; }}
  header {{ display:flex; align-items:flex-end; justify-content:space-between;
            gap:16px; padding-bottom:16px; border-bottom:1px solid var(--line); }}
  h1 {{ font-size:20px; font-weight:600; margin:0; }}
  .sub {{ color:var(--muted); font-size:13px; margin-top:4px; }}
  .score {{ text-align:right; }}
  .score .n {{ font-size:44px; font-weight:600; line-height:1; color:{color}; }}
  .score .l {{ font-size:13px; color:var(--muted); margin-top:4px; }}
  .chips {{ margin:16px 0 4px; display:flex; flex-wrap:wrap; gap:8px; }}
  .chip {{ font-size:12px; padding:3px 10px; border-radius:999px; border:1px solid var(--line); }}
  .chip.on {{ background:#EAF3DE; border-color:#C0DD97; }}
  .chip.off {{ background:#FCEBEB; border-color:#F7C1C1; color:#791F1F; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr));
           gap:18px; margin:22px 0; }}
  .metric-head {{ display:flex; justify-content:space-between; font-size:13px; margin-bottom:6px; }}
  .metric-head .value {{ font-weight:600; }}
  .track {{ height:8px; border-radius:4px; background:var(--soft); overflow:hidden; }}
  .fill {{ height:100%; background:#378ADD; }}
  .fill.unavailable {{ background:repeating-linear-gradient(45deg,#E3E3E3,#E3E3E3 6px,#F7F7F5 6px,#F7F7F5 12px); }}
  .hint {{ font-size:12px; color:var(--muted); margin-top:6px; }}
  h3 {{ font-size:14px; font-weight:600; margin:0 0 8px; }}
  table {{ width:100%; border-collapse:collapse; margin-top:8px; font-size:13px; }}
  th,td {{ text-align:left; padding:8px 10px; border-bottom:1px solid var(--line); }}
  th {{ color:var(--muted); font-weight:500; }}
  td.num {{ font-variant-numeric:tabular-nums; font-weight:600; }}
  td.who {{ font-weight:600; }}
  .notes {{ line-height:2; }}
  .tag {{ font-size:12px; padding:2px 8px; border-radius:4px; display:inline-block; margin-right:4px; }}
  .tag.warn {{ background:#FAEEDA; }}
  .tag.ok {{ background:#EAF3DE; }}
  .tag.muted {{ background:var(--soft); color:var(--muted); }}
  .block {{ margin-top:22px; }}
  .block ul {{ margin:0; padding-left:20px; }}
  .block.warn-soft {{ background:#FAEEDA; border-radius:8px; padding:14px 18px; }}
  .block.warn-soft ul {{ padding-left:18px; }}
  .note {{ font-size:13px; padding:12px 16px; border-radius:8px; background:#FAEEDA; }}
  .note.warn strong {{ color:#854F0B; }}
  .muted {{ color:var(--muted); }}
  footer {{ margin-top:32px; padding-top:14px; border-top:1px solid var(--line);
            font-size:12px; color:var(--muted); }}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      <h1>小组协作过程性分析报告</h1>
      <div class="sub">小组 {_esc(report.group_id)} · 时长 {report.total_duration:.1f} 秒 ·
        成员 {_esc("、".join(report.member_ids) or "未识别")}</div>
    </div>
    <div class="score">
      <div class="n">{report.overall_health_score:.1f}</div>
      <div class="l">{_esc(_LEVEL_TEXT.get(level, level.value))}</div>
    </div>
  </header>

  <div class="chips">{completeness_chips}</div>
  {missing_note}

  <div class="grid">{metrics}</div>

  <h3>逐人贡献度</h3>
  {_contributions_table(report)}

  {_list_block("诊断", report.diagnoses)}
  {_list_block("建议", report.suggestions)}
  {_list_block("过程告警（数据完整性问题，请务必阅读）", report.warnings, "warn-soft")}

  <footer>
    生成时间 {_esc(report.created_at)} · collaborative-learning-analyzer<br>
    本报告仅为教学参考线索，<strong>不得作为评价学生的依据</strong>，
    也不得在论文中作为测量结果使用（指标权重尚未经过实证标定）。
  </footer>
</div>
</body>
</html>
"""


def render_report(
    report: Union[GroupCollaborationReport, str, Path], output_path: Union[str, Path]
) -> Path:
    """把报告渲染为 HTML 文件，返回输出路径。

    Args:
        report: 报告对象，或已保存的报告 JSON 路径
        output_path: HTML 输出路径
    """
    if isinstance(report, (str, Path)):
        report = GroupCollaborationReport.load(str(report))

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(report), encoding="utf-8")
    return path
