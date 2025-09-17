import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ART_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ART_DIR, exist_ok=True)

CAT_LIMIT = 30
DIV_ID = "umap_plot"

# Keep dropdown sane: only useful fields
ALLOWED_NUMERIC = ["cluster", "popularity", "tempo", "energy", "danceability", "valence", "loudness", "year"]
ALLOWED_CATEGORICAL = ["cluster", "genre", "artist_name", "artists"]

def is_numeric(s):
    return pd.api.types.is_numeric_dtype(s)

def cat_to_colors(series, palette=None, limit=CAT_LIMIT):
    s = series.astype("string").fillna("Unknown")
    top = s.value_counts().head(limit).index.tolist()
    masked = s.where(s.isin(top), other="Other")
    cats = pd.Index(masked.unique())
    if palette is None:
        palette = (
            px.colors.qualitative.Alphabet
            + px.colors.qualitative.Set3
            + px.colors.qualitative.Dark24
        )
    colors = {cat: palette[i % len(palette)] for i, cat in enumerate(cats)}
    return masked.map(colors).tolist(), cats.tolist(), colors

def build_hover(df):
    # Rich hover content (quick glance; pinned panel uses same customdata)
    cols = []
    for c in ["__display", "genre", "year", "popularity", "tempo", "energy",
              "danceability", "valence", "loudness", "cluster"]:
        if c in df.columns:
            cols.append(c)
    for j in range(1, 6):
        c = f"nn{j}"
        if c in df.columns:
            cols.append(c)
    if "__spotify_url" in df.columns:
        cols.append("__spotify_url")

    custom = df[cols].fillna("").to_numpy()

    # Escape braces for Plotly hovertemplate: %{{customdata[idx]}}
    parts = []
    if "__display" in cols:
        idx = cols.index("__display")
        parts.append(f"<b>%{{customdata[{idx}]}}</b>")
    if "genre" in cols:
        idx = cols.index("genre")
        parts.append(f"genre: %{{customdata[{idx}]}}")
    for tag in ["year", "popularity", "tempo", "energy", "danceability", "valence", "loudness", "cluster"]:
        if tag in cols:
            idx = cols.index(tag)
            parts.append(f"{tag}: %{{customdata[{idx}]}}")

    nn_lines = []
    for j in range(1, 6):
        name = f"nn{j}"
        if name in cols:
            idx = cols.index(name)
            nn_lines.append(f"%{{customdata[{idx}]}}")
    if nn_lines:
        parts.append("<b>Similar:</b><br>" + "<br>".join(nn_lines))

    if "__spotify_url" in cols:
        idx = cols.index("__spotify_url")
        parts.append(f"<a href='%{{customdata[{idx}]}}' target='_blank'>Open in Spotify search</a>")

    hovertemplate = "<br>".join(parts) + "<extra></extra>"
    return custom, hovertemplate, cols

def main():
    emb_csv = os.path.join(ART_DIR, "embeddings_3d.csv")
    df = pd.read_csv(emb_csv)

    exclude = {"x", "y", "z", "__display", "__spotify_url", "track_name", "name", "nn1", "nn2", "nn3", "nn4", "nn5"}
    all_cols = [c for c in df.columns if c not in exclude]
    numeric_cols = [c for c in all_cols if is_numeric(df[c]) and c in ALLOWED_NUMERIC]
    cat_cols = [c for c in all_cols if (c not in numeric_cols) and (c in ALLOWED_CATEGORICAL)]

    # Guarantee cluster shows up if present
    if "cluster" in df.columns and "cluster" not in (numeric_cols + cat_cols):
        cat_cols.append("cluster")

    # Build hover/customdata
    custom, hovertemplate, custom_cols = build_hover(df)

    # Default color choice
    pref = ["cluster", "genre", "popularity", "energy", "danceability", "valence", "loudness", "year"]
    default_color = next((c for c in pref if c in (numeric_cols + cat_cols)), None)

    # Initial marker color
    if default_color in numeric_cols:
        marker_color = df[default_color].to_numpy()
        showscale, colorscale = True, "Viridis"
    elif default_color in cat_cols:
        marker_color, _, _ = cat_to_colors(df[default_color])
        showscale, colorscale = False, None
    else:
        marker_color, showscale, colorscale = "rgba(90,90,90,0.7)", False, None

    trace = go.Scatter3d(
        x=df["x"],
        y=df["y"],
        z=df["z"],
        mode="markers",
        marker=dict(size=2, color=marker_color, colorscale=colorscale, showscale=showscale, opacity=0.9),
        customdata=custom,
        hovertemplate=hovertemplate,
    )
    fig = go.Figure([trace])
    fig.update_layout(
        title="Spotify 3D Embedding Explorer (UMAP)",
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, t=50, b=0),
    )

    # Dropdown buttons
    buttons = []
    # Uniform
    buttons.append(
        dict(
            label="Uniform",
            method="restyle",
            args=[{"marker.color": [["rgba(90,90,90,0.7)"]], "marker.colorscale": [None], "marker.showscale": [False]}, [0]],
        )
    )
    # Numeric
    for col in numeric_cols:
        buttons.append(
            dict(
                label=f"{col} (num)",
                method="restyle",
                args=[{"marker.color": [df[col].to_numpy()], "marker.colorscale": ["Viridis"], "marker.showscale": [True]}, [0]],
            )
        )
    # Categorical
    for col in cat_cols:
        color_list, _, _ = cat_to_colors(df[col])
        buttons.append(
            dict(
                label=f"{col} (cat)",
                method="restyle",
                args=[{"marker.color": [color_list], "marker.colorscale": [None], "marker.showscale": [False]}, [0]],
            )
        )

    fig.update_layout(
        updatemenus=[
            dict(
                type="dropdown",
                x=0.02,
                y=0.98,
                xanchor="left",
                yanchor="top",
                buttons=buttons,
                direction="down",
                showactive=True,
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="rgba(0,0,0,0.2)",
            )
        ]
    )

    # ----------- JS: robust boot + click-to-pin panel -----------
    post_js = f"""
(function(){{
  function ready(fn) {{
    if (document.readyState !== 'loading') fn();
    else document.addEventListener('DOMContentLoaded', fn);
  }}
  ready(function(){{
    function getGd() {{
      return document.getElementById('{DIV_ID}') || document.querySelector('.plotly-graph-div');
    }}
    function boot() {{
      var gd = getGd();
      if (!gd || typeof Plotly === 'undefined') {{
        setTimeout(boot, 50);
        return;
      }}

      // Floating info panel
      const panel = document.createElement('div');
      Object.assign(panel.style, {{
        position:'absolute', top:'12px', right:'12px', maxWidth:'360px', maxHeight:'60vh',
        overflow:'auto', padding:'10px 12px', border:'1px solid rgba(0,0,0,0.15)',
        borderRadius:'12px', background:'rgba(255,255,255,0.95)', boxShadow:'0 4px 18px rgba(0,0,0,0.15)',
        fontFamily:'system-ui,-apple-system,Segoe UI,Roboto,Arial', fontSize:'12px', lineHeight:'1.35',
        display:'none', pointerEvents:'auto', zIndex:50
      }});
      const content = document.createElement('div');
      const closeBtn = document.createElement('button');
      closeBtn.textContent = 'Close';
      Object.assign(closeBtn.style, {{
        marginTop:'8px', padding:'6px 10px', border:'1px solid rgba(0,0,0,0.15)',
        borderRadius:'6px', cursor:'pointer', background:'#fff'
      }});
      closeBtn.onclick = () => panel.style.display = 'none';
      panel.appendChild(content); panel.appendChild(closeBtn);
      gd.parentNode.style.position = 'relative';
      gd.parentNode.appendChild(panel);

      // Drag vs click discrimination
      let downPos = null, dragging = false;
      const CLICK_DIST = 5;

      gd.addEventListener('pointerdown', ev => {{
        downPos = {{x: ev.clientX, y: ev.clientY}};
        dragging = false;
      }});
      gd.addEventListener('pointermove', ev => {{
        if (!downPos) return;
        const dx = Math.abs(ev.clientX - downPos.x);
        const dy = Math.abs(ev.clientY - downPos.y);
        if (dx > CLICK_DIST || dy > CLICK_DIST) dragging = true;
      }});
      gd.addEventListener('pointerup', () => {{ downPos = null; }});

      // Hide panel while rotating/zooming
      gd.on('plotly_relayouting', () => {{ panel.style.display = 'none'; }});
      document.addEventListener('keydown', (e) => {{ if (e.key === 'Escape') panel.style.display = 'none'; }});
      gd.on('plotly_doubleclick', () => panel.style.display = 'none');

      // Orbit drag
      Plotly.relayout(gd, {{'scene.dragmode': 'orbit'}});

      // Click to pin (only if not dragging)
      gd.on('plotly_click', function(ev){{
        if (dragging) return;
        if (!ev || !ev.points || !ev.points.length) return;
        const cd = ev.points[0].customdata || [];
        window.__fillPanel && window.__fillPanel(content, cd, panel);
        panel.style.display = 'block';
      }});
    }}
    boot();
  }});
}})();
"""

    # Build the fillPanel function (indexes into customdata)
    def idx(name): 
        return custom_cols.index(name) if name in custom_cols else -1

    def js_field(label, name):
        i = idx(name)
        # Use template literals in JS; double braces to escape f-string braces
        return f"(i=> i>=0 && cd[i] ? `<div><b>{label}</b>: ${{cd[i]}}</div>` : '')({i})"

    nn_items = " + '<br>' + ".join([f"(i=> i>=0 && cd[i] ? cd[i] : '')({idx(f'nn{j}')})" for j in range(1, 6)])
    nn_block = ("`<div><b>Similar:</b><br>` + " + nn_items + " + `</div>`") if any(idx(f"nn{j}") >= 0 for j in range(1, 6)) else "''"
    spotify_block = f"(i=> i>=0 && cd[i] ? `<div style='margin-top:6px;'><a href='${{cd[i]}}' target='_blank'>Open in Spotify search</a></div>` : '')({idx('__spotify_url')})"

    panel_js = f"""
(function(){{
  const fill = function(content, cd, panel){{
    let html = '';
    {"html += " + js_field("track", "__display") + ";" if idx("__display") >= 0 else ""}
    {"html += " + js_field("genre", "genre") + ";" if idx("genre") >= 0 else ""}
    {"html += " + js_field("year", "year") + ";" if idx("year") >= 0 else ""}
    {"html += " + js_field("popularity", "popularity") + ";" if idx("popularity") >= 0 else ""}
    {"html += " + js_field("tempo", "tempo") + ";" if idx("tempo") >= 0 else ""}
    {"html += " + js_field("energy", "energy") + ";" if idx("energy") >= 0 else ""}
    {"html += " + js_field("danceability", "danceability") + ";" if idx("danceability") >= 0 else ""}
    {"html += " + js_field("valence", "valence") + ";" if idx("valence") >= 0 else ""}
    {"html += " + js_field("loudness", "loudness") + ";" if idx("loudness") >= 0 else ""}
    {"html += " + js_field("cluster", "cluster") + ";" if idx("cluster") >= 0 else ""}
    html += {nn_block};
    html += {spotify_block};
    content.innerHTML = html;
  }};
  window.__fillPanel = fill;
}})();
"""

    full_js = post_js + "\n" + panel_js

    out_html = os.path.join(ART_DIR, "spotify_3d_plot.html")
    pio.write_html(
        fig,
        file=out_html,
        include_plotlyjs=True,  # inline Plotly to avoid CDN timing issues
        full_html=True,
        div_id=DIV_ID,
        post_script=full_js,
    )
    print("Numeric color options:", numeric_cols)
    print("Categorical color options:", cat_cols)
    print(f"Saved interactive plot with dropdown + click-to-pin panel: {out_html}")

if __name__ == "__main__":
    main()
