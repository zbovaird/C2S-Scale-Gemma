"""Render a self-contained HTML trajectory projection viewer."""

from __future__ import annotations

import json
from typing import Any, Dict


def _serialize_payload(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, indent=2).replace("</", "<\\/")


def render_validation_trajectory_projection_html(payload: Dict[str, Any]) -> str:
    """Render a lightweight HTML viewer for trajectory projection artifacts."""
    serialized = _serialize_payload(payload)
    title = payload.get("track_name") or "validation_trajectory_projection"
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
    .toolbar {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-bottom: 16px;
    }}
    .panel {{
      background: #ffffff;
      border: 1px solid #dbe4ee;
      border-radius: 10px;
      padding: 14px;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
    }}
    #plot {{
      width: 100%;
      max-width: 960px;
      height: 640px;
      border: 1px solid #dbe4ee;
      background: #ffffff;
    }}
    .small {{
      color: #64748b;
      font-size: 12px;
    }}
    #legend {{
      margin-top: 12px;
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }}
    .legend-item {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 12px;
    }}
    .swatch {{
      width: 12px;
      height: 12px;
      border-radius: 50%;
      display: inline-block;
    }}
  </style>
</head>
<body>
  <h1>Trajectory Projection Viewer</h1>
  <p class="small" id="subtitle"></p>
  <div class="toolbar panel">
    <label>
      Run
      <select id="run-select"></select>
    </label>
    <label>
      Color by
      <select id="color-mode">
        <option value="branch">Branch</option>
        <option value="safe_zone">Safe zone</option>
        <option value="timepoint">Timepoint</option>
      </select>
    </label>
  </div>
  <div class="panel">
    <canvas id="plot" width="960" height="640"></canvas>
    <div id="legend"></div>
  </div>
  <script>
    const payload = {serialized};
    const canvas = document.getElementById("plot");
    const ctx = canvas.getContext("2d");
    const runSelect = document.getElementById("run-select");
    const colorModeSelect = document.getElementById("color-mode");
    const legend = document.getElementById("legend");
    const palette = ["#2563eb", "#059669", "#d97706", "#dc2626", "#7c3aed", "#0891b2", "#9333ea"];

    function setSubtitle() {{
      const subtitle = payload.dataset_profile
        ? `${{payload.track_name}} | ${{payload.dataset_profile}} | ${{payload.projection_method}}`
        : String(payload.track_name || "");
      document.getElementById("subtitle").textContent = subtitle;
    }}

    function populateRunSelect() {{
      (payload.runs || []).forEach((run, idx) => {{
        const option = document.createElement("option");
        option.value = String(idx);
        option.textContent = `${{run.label}} (${{run.alignment_mode}})`;
        runSelect.appendChild(option);
      }});
    }}

    function getColorKey(row, mode) {{
      if (mode === "safe_zone") {{
        return row.longevity_safe_zone ? "safe_zone" : "not_safe";
      }}
      if (mode === "timepoint") {{
        return String(row.timepoint || "unknown");
      }}
      return String(row.branch_label || "unknown");
    }}

    function buildColorMap(rows, mode) {{
      const keys = [...new Set(rows.map((row) => getColorKey(row, mode)))];
      const colorMap = {{}};
      keys.forEach((key, idx) => {{
        if (mode === "safe_zone") {{
          colorMap[key] = key === "safe_zone" ? "#059669" : "#94a3b8";
        }} else {{
          colorMap[key] = palette[idx % palette.length];
        }}
      }});
      return colorMap;
    }}

    function projectPoint(x, y, bounds) {{
      const padding = 40;
      const xSpan = Math.max(bounds.maxX - bounds.minX, 1e-6);
      const ySpan = Math.max(bounds.maxY - bounds.minY, 1e-6);
      return {{
        x: padding + ((x - bounds.minX) / xSpan) * (canvas.width - padding * 2),
        y: canvas.height - padding - ((y - bounds.minY) / ySpan) * (canvas.height - padding * 2),
      }};
    }}

    function renderLegend(colorMap) {{
      legend.innerHTML = "";
      Object.entries(colorMap).forEach(([label, color]) => {{
        const item = document.createElement("div");
        item.className = "legend-item";
        item.innerHTML = `<span class="swatch" style="background:${{color}}"></span><span>${{label}}</span>`;
        legend.appendChild(item);
      }});
    }}

    function render() {{
      const run = (payload.runs || [])[Number(runSelect.value) || 0];
      if (!run) {{
        return;
      }}
      const mode = colorModeSelect.value;
      const rows = run.rows || [];
      const colorMap = buildColorMap(rows, mode);
      const values = rows.flatMap((row) => [
        [Number(row.baseline_x || 0), Number(row.baseline_y || 0)],
        [Number(row.perturbed_x || 0), Number(row.perturbed_y || 0)],
      ]);
      const xs = values.map((pair) => pair[0]);
      const ys = values.map((pair) => pair[1]);
      const bounds = {{
        minX: Math.min(...xs, -1),
        maxX: Math.max(...xs, 1),
        minY: Math.min(...ys, -1),
        maxY: Math.max(...ys, 1),
      }};

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      rows.forEach((row) => {{
        const start = projectPoint(Number(row.baseline_x || 0), Number(row.baseline_y || 0), bounds);
        const end = projectPoint(Number(row.perturbed_x || 0), Number(row.perturbed_y || 0), bounds);
        const color = colorMap[getColorKey(row, mode)];

        ctx.strokeStyle = "rgba(148, 163, 184, 0.45)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(start.x, start.y);
        ctx.lineTo(end.x, end.y);
        ctx.stroke();

        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(end.x, end.y, 4, 0, Math.PI * 2);
        ctx.fill();
      }});

      renderLegend(colorMap);
    }}

    populateRunSelect();
    setSubtitle();
    runSelect.addEventListener("change", render);
    colorModeSelect.addEventListener("change", render);
    render();
  </script>
</body>
</html>
"""
