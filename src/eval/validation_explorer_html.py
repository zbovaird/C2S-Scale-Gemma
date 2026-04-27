"""Render a self-contained HTML validation explorer."""

from __future__ import annotations

import json
from html import escape
from typing import Any, Dict


def _serialize_payload(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, indent=2).replace("</", "<\\/")


def render_validation_explorer_html(payload: Dict[str, Any]) -> str:
    """Render a lightweight interactive HTML view from an explorer payload."""
    serialized_payload = _serialize_payload(payload)
    title = escape(str(payload.get("track_name") or "validation_explorer"), quote=True)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 24px;
      color: #1f2937;
      background: #f8fafc;
    }}
    h1, h2 {{
      margin-bottom: 12px;
    }}
    .cards, .charts {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
      margin-bottom: 24px;
    }}
    .card, .panel {{
      background: #ffffff;
      border: 1px solid #dbe4ee;
      border-radius: 10px;
      padding: 14px;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: #ffffff;
    }}
    th, td {{
      text-align: left;
      padding: 8px;
      border-bottom: 1px solid #e5e7eb;
      font-size: 14px;
    }}
    .chart-title {{
      font-weight: 600;
      margin-bottom: 8px;
    }}
    .legend {{
      font-size: 12px;
      color: #475569;
      margin-top: 8px;
    }}
    svg {{
      width: 100%;
      height: 220px;
      background: #ffffff;
    }}
    .evidence-list {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 12px;
    }}
    .small {{
      color: #64748b;
      font-size: 12px;
    }}
  </style>
</head>
<body>
  <h1>Validation Explorer</h1>
  <p class="small" id="subtitle"></p>
  <section>
    <h2>Overview</h2>
    <div class="cards" id="overview-cards"></div>
  </section>
  <section>
    <h2>Run Overview</h2>
    <div class="panel">
      <table id="run-table"></table>
    </div>
  </section>
  <section>
    <h2>Trajectory Charts</h2>
    <div class="charts" id="chart-grid"></div>
  </section>
  <section>
    <h2>Recommendation Evidence</h2>
    <div class="evidence-list">
      <div class="panel">
        <div class="chart-title">Supporting timepoints</div>
        <div id="supporting-evidence"></div>
      </div>
      <div class="panel">
        <div class="chart-title">Concerning timepoints</div>
        <div id="concerning-evidence"></div>
      </div>
    </div>
  </section>
  <script>
    const payload = {serialized_payload};

    const palette = ["#2563eb", "#059669", "#d97706", "#dc2626", "#7c3aed"];

    function escapeHtml(value) {{
      return String(value ?? "n/a")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
    }}

    function setSubtitle() {{
      const subtitle = payload.dataset_profile
        ? `${{payload.track_name}} | ${{payload.dataset_profile}}`
        : String(payload.track_name || "");
      document.getElementById("subtitle").textContent = subtitle;
    }}

    function renderCards() {{
      const container = document.getElementById("overview-cards");
      (payload.overview_cards || []).forEach((card) => {{
        const el = document.createElement("div");
        el.className = "card";
        el.innerHTML = `<div class="small">${{escapeHtml(card.label)}}</div><div>${{escapeHtml(card.value)}}</div>`;
        container.appendChild(el);
      }});
    }}

    function renderRunTable() {{
      const table = document.getElementById("run-table");
      const rows = payload.run_table || [];
      if (!rows.length) {{
        table.innerHTML = "<tr><td>No runs available.</td></tr>";
        return;
      }}
      const columns = Object.keys(rows[0]);
      const thead = `<thead><tr>${{columns.map((col) => `<th>${{escapeHtml(col)}}</th>`).join("")}}</tr></thead>`;
      const tbody = `<tbody>${{rows.map((row) => `<tr>${{columns.map((col) => `<td>${{escapeHtml(row[col])}}</td>`).join("")}}</tr>`).join("")}}</tbody>`;
      table.innerHTML = thead + tbody;
    }}

    function renderLineChart(chart, parent) {{
      const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      const width = 360;
      const height = 220;
      const padding = 28;
      svg.setAttribute("viewBox", `0 0 ${{width}} ${{height}}`);
      const labels = [];
      const values = [];
      Object.values(chart.series || {{}}).forEach((rows) => {{
        rows.forEach((row) => {{
          labels.push(String(row.timepoint));
          values.push(Number(row.value || 0));
        }});
      }});
      const uniqueLabels = [...new Set(labels)];
      const minValue = Math.min(...values, 0);
      const maxValue = Math.max(...values, 1);
      Object.entries(chart.series || {{}}).forEach(([label, rows], index) => {{
        if (!rows.length) return;
        const points = rows.map((row) => {{
          const xIndex = uniqueLabels.indexOf(String(row.timepoint));
          const x = padding + (uniqueLabels.length <= 1 ? 0 : (xIndex * (width - padding * 2)) / (uniqueLabels.length - 1));
          const value = Number(row.value || 0);
          const y = height - padding - ((value - minValue) * (height - padding * 2)) / Math.max(maxValue - minValue, 1e-6);
          return `${{x}},${{y}}`;
        }});
        const path = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
        path.setAttribute("fill", "none");
        path.setAttribute("stroke", palette[index % palette.length]);
        path.setAttribute("stroke-width", "2");
        path.setAttribute("points", points.join(" "));
        svg.appendChild(path);
      }});
      parent.appendChild(svg);
      const legend = document.createElement("div");
      legend.className = "legend";
      legend.textContent = Object.keys(chart.series || {{}}).join(" | ");
      parent.appendChild(legend);
    }}

    function renderBarChart(chart, parent) {{
      const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      const width = 360;
      const height = 220;
      const padding = 28;
      svg.setAttribute("viewBox", `0 0 ${{width}} ${{height}}`);
      const rows = chart.series || [];
      const maxAbs = Math.max(...rows.map((row) => Math.abs(Number(row.value || 0))), 1);
      rows.forEach((row, index) => {{
        const x = padding + index * ((width - padding * 2) / Math.max(rows.length, 1));
        const barWidth = Math.max(((width - padding * 2) / Math.max(rows.length, 1)) - 10, 12);
        const zeroY = height / 2;
        const value = Number(row.value || 0);
        const barHeight = (Math.abs(value) * (height / 2 - padding)) / maxAbs;
        const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        rect.setAttribute("x", String(x));
        rect.setAttribute("width", String(barWidth));
        rect.setAttribute("y", String(value >= 0 ? zeroY - barHeight : zeroY));
        rect.setAttribute("height", String(barHeight));
        rect.setAttribute("fill", value >= 0 ? "#2563eb" : "#dc2626");
        svg.appendChild(rect);
      }});
      parent.appendChild(svg);
    }}

    function renderCharts() {{
      const grid = document.getElementById("chart-grid");
      (payload.charts || []).forEach((chart) => {{
        const panel = document.createElement("div");
        panel.className = "panel";
        panel.innerHTML = `<div class="chart-title">${{escapeHtml(chart.title)}}</div>`;
        if (chart.kind === "line") {{
          renderLineChart(chart, panel);
        }} else {{
          renderBarChart(chart, panel);
        }}
        grid.appendChild(panel);
      }});
    }}

    function renderEvidence(targetId, rows) {{
      const target = document.getElementById(targetId);
      if (!rows.length) {{
        target.textContent = "No evidence rows available.";
        return;
      }}
      target.innerHTML = rows.map((row) => `
        <div class="small">
          <strong>${{escapeHtml(row.timepoint)}}</strong> |
          safe ${{Number(row.delta_safe_fraction || 0).toFixed(3)}} |
          productive ${{Number(row.delta_productive_fraction || 0).toFixed(3)}} |
          risk ${{Number(row.delta_risk_fraction || 0).toFixed(3)}}
        </div>
      `).join("");
    }}

    setSubtitle();
    renderCards();
    renderRunTable();
    renderCharts();
    const evidence = payload.recommendation?.evidence || {{}};
    renderEvidence("supporting-evidence", evidence.top_supporting_timepoints || []);
    renderEvidence("concerning-evidence", evidence.top_concerning_timepoints || []);
  </script>
</body>
</html>
"""
