# 🎶 Spotify 3D Embedding Explorer

An ML project that downloads a Spotify tracks dataset, builds 3D embeddings with **UMAP**, and renders an **interactive Plotly 3D explorer** you can open right in your browser.  

---

## 🚀 Quickstart

```bash
# 1) Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run the pipeline
python -m src.data.download_spotify
python -m src.features.make_embeddings
python -m src.viz.plot_3d

# 4) Open the saved HTML visualization
open artifacts/spotify_3d_plot.html   # Windows: start artifacts\spotify_3d_plot.html

spotify-3d-embedding-explorer/
├─ src/
│  ├─ data/
│  │  └─ download_spotify.py
│  ├─ features/
│  │  └─ make_embeddings.py
│  └─ viz/
│     └─ plot_3d.py
├─ artifacts/      # generated plots + embeddings
├─ data/           # raw datasets (after download)
├─ notebooks/      # (optional) experiments
├─ scripts/        # helper shell scripts
├─ app/            # (optional) Streamlit/Gradio app in the future
├─ requirements.txt
├─ .gitignore
└─ README.md
