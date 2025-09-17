import os, shutil, zipfile
import kagglehub

# Swap slug here if you change datasets later
DATASET = "zaheenhamidani/ultimate-spotify-tracks-db"

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
ART_DIR  = os.path.join(BASE_DIR, "artifacts")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ART_DIR, exist_ok=True)

# Common filenames seen across Spotify datasets on Kaggle
CANDIDATE_FILES = [
    "tracks.csv",
    "SpotifyFeatures.csv",
    "data.csv",
    "spotify.csv",
]

OUT_CSV = os.path.join(DATA_DIR, "tracks.csv")  # normalized path the rest of the pipeline expects

def main():
    # Download (kagglehub will cache to ~/.cache)
    path = kagglehub.dataset_download(DATASET)
    print(f"Dataset downloaded to: {path}")

    # Try to find a plausible CSV
    src_csv = None
    for root, _, files in os.walk(path):
        for name in files:
            if name in CANDIDATE_FILES or name.lower().endswith(".csv"):
                # Prefer known names first; else take the first CSV we see
                src_csv = os.path.join(root, name)
                if name in CANDIDATE_FILES:
                    break
        if src_csv:
            break

    if not src_csv:
        raise FileNotFoundError(f"No CSV found under {path}")

    # Normalize to data/tracks.csv for the rest of the pipeline
    shutil.copy2(src_csv, OUT_CSV)
    print(f"Materialized: {OUT_CSV} (from {os.path.basename(src_csv)})")

if __name__ == "__main__":
    main()
