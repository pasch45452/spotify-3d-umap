
#!/usr/bin/env bash
set -euo pipefail

python -m src.data.download_spotify
python -m src.features.make_embeddings
python -m src.viz.plot_3d

echo "Done. Open artifacts/spotify_3d_plot.html in your browser."
