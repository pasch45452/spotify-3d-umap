import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import umap

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
ART_DIR  = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ART_DIR, exist_ok=True)

FEATURES = [
    "acousticness","danceability","energy","instrumentalness",
    "liveness","loudness","speechiness","valence","tempo",
    "duration_ms","mode","key","time_signature"
]

LABELS = ["track_name","name","artists","artist_name","genre","popularity","year","release_date","album","album_name"]

KEY_MAP = {"C":0,"C#":1,"Db":1,"D":2,"D#":3,"Eb":3,"E":4,"F":5,
           "F#":6,"Gb":6,"G":7,"G#":8,"Ab":8,"A":9,"A#":10,"Bb":10,"B":11}
MODE_MAP = {"Major":1,"Minor":0,"major":1,"minor":0,"MAJOR":1,"MINOR":0}

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "mode" in df.columns:
        s = df["mode"]
        df["mode"] = s.map(MODE_MAP).fillna(pd.to_numeric(s, errors="coerce"))
    if "key" in df.columns:
        s = df["key"]
        mapped = s.map(KEY_MAP)
        df["key"] = mapped.where(~mapped.isna(), pd.to_numeric(s, errors="coerce"))
    # coerce likely numeric
    for col in ["time_signature","duration_ms","tempo","loudness",
                "acousticness","danceability","energy","instrumentalness",
                "liveness","speechiness","valence","popularity","year"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _s(x):
    """safe string: NaN/None -> '' else str(x)"""
    return "" if pd.isna(x) else str(x)

def display_name(row):
    title  = _s(row.get("track_name") or row.get("name"))
    artist = _s(row.get("artist_name") or row.get("artists"))
    return f"{title} — {artist}" if title and artist else (title or artist or "(unknown)")

def spotify_search_url(row):
    title  = _s(row.get("track_name") or row.get("name"))
    artist = _s(row.get("artist_name") or row.get("artists"))
    q = "+".join([v for v in [title, artist] if v]).replace(" ", "+")
    return f"https://open.spotify.com/search/{q}" if q else ""


def main():
    tracks_csv = os.path.join(DATA_DIR, "tracks.csv")
    df_raw = pd.read_csv(tracks_csv, low_memory=False)
    df = preprocess(df_raw)

    # features available
    avail = [c for c in FEATURES if c in df.columns]
    if not avail:
        raise ValueError(f"No expected feature columns found in {tracks_csv}.")

    X = df[avail].replace([np.inf, -np.inf], np.nan)

    # drop empty cols, impute means
    all_nan_cols = [c for c in X.columns if X[c].isna().all()]
    if all_nan_cols:
        X = X.drop(columns=all_nan_cols)
        avail = [c for c in avail if c not in all_nan_cols]
    X = X.dropna(how="all")
    X = X.fillna(X.mean(numeric_only=True))
    if X.empty:
        raise ValueError("No usable rows after cleaning.")

    # scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # keep aligned index for metadata
    kept_index = X.index

    # --- (A) k-means clustering (quick, informative) ---
    n_clusters = 20
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster = km.fit_predict(X_scaled)

    # --- (B) kNN for “similar tracks” on hover ---
    # use the high-dim scaled space (not UMAP) for better neighbors
    knn = NearestNeighbors(n_neighbors=6, metric="euclidean")  # 1 self + 5 neighbors
    knn.fit(X_scaled)
    dists, idxs = knn.kneighbors(X_scaled)
    # build neighbor label strings (skip self at position 0)
    # prepare label columns for hover text
    meta_cols = [c for c in LABELS if c in df.columns]
    meta = df.loc[kept_index, meta_cols].copy()
    meta["__display"] = meta.apply(display_name, axis=1)
    meta["__spotify_url"] = df.loc[kept_index].apply(spotify_search_url, axis=1)

    neighbors_cols = []
    for j in range(1, 6):
        col = f"nn{j}"
        neighbors_cols.append(col)
        # idxs[:, j] are row positions into kept_index
        meta[col] = [meta["__display"].iloc[i] for i in idxs[:, j]]

    # --- (C) UMAP 3D for visualization ---
    reducer = umap.UMAP(n_components=3, random_state=42, n_neighbors=30, min_dist=0.1)
    emb3d = reducer.fit_transform(X_scaled)

    # assemble output
    out = meta.copy()
    out["cluster"] = cluster
    out["x"], out["y"], out["z"] = emb3d[:,0], emb3d[:,1], emb3d[:,2]
    # keep optional helpful numerics for dropdown coloring
    for c in ["popularity","year","tempo","energy","danceability","valence","loudness"]:
        if c in df.columns:
            out[c] = df.loc[kept_index, c].values

    out_csv = os.path.join(ART_DIR, "embeddings_3d.csv")
    out.to_csv(out_csv, index=False)
    print(f"✅ Saved embeddings: {out_csv}  rows={len(out)}  features={list(X.columns)}")

if __name__ == "__main__":
    main()
