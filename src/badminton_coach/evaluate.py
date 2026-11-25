"""Shared evaluation for every model (baselines, GRU, ShuttleNet, LLM).

Task   : given strokes 1-5 of a test rally, predict stroke 6 (see data.py).
Metrics: accuracy, macro-F1, top-3 accuracy (shot type);
         mean Euclidean landing error in court-coordinate units.

Coordinates: ShuttleSet landings are in one fixed court frame that covers BOTH
halves (net near standardized y = 0, centre line near x = 0). Stroke t usually
lands on the same half as stroke t-2, so models may "canonicalise" coordinates
by rotating the court 180 degrees: (x, y) -> (-x, -y).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

PROCESSED = Path("data/processed")
RESULTS_CSV = Path("results/metrics.csv")


def load_processed() -> tuple[pd.DataFrame, dict]:
    df = pd.read_parquet(PROCESSED / "strokes.parquet")
    meta = json.loads((PROCESSED / "meta.json").read_text(encoding="utf-8"))
    return df, meta


def next_shot_table(df: pd.DataFrame, split: str, encode_length: int = 5) -> pd.DataFrame:
    """One row per rally with more than `encode_length` strokes.

    Context columns hold lists of length `encode_length`; target is stroke encode_length+1.
    """
    rows = []
    for rally_id, r in df[df["split"] == split].groupby("rally_id", sort=False):
        if len(r) <= encode_length:
            continue
        ctx, tgt = r.iloc[:encode_length], r.iloc[encode_length]
        rows.append({
            "rally_id": rally_id,
            "match_id": r["match_id"].iloc[0],
            "ctx_type": ctx["type_id"].tolist(),
            "ctx_player": ctx["player_id"].tolist(),
            "ctx_x": ctx["landing_x"].tolist(),
            "ctx_y": ctx["landing_y"].tolist(),
            "target_player": int(tgt["player_id"]),
            "target_type": int(tgt["type_id"]),
            "target_x": float(tgt["landing_x"]),
            "target_y": float(tgt["landing_y"]),
        })
    return pd.DataFrame(rows)


def side_sign(y_two_strokes_before: np.ndarray) -> np.ndarray:
    """+1 / -1: which half the next stroke is expected to land on."""
    s = np.sign(np.asarray(y_two_strokes_before, dtype=float))
    s[s == 0] = 1
    return s


def to_court_units(dx: np.ndarray, dy: np.ndarray, meta: dict) -> np.ndarray:
    """Undo the standardisation so x and y use the same unit."""
    return np.hypot(dx * meta["coord_std"]["x"], dy * meta["coord_std"]["y"])


def score(model: str, probs: np.ndarray, xy_pred: np.ndarray, table: pd.DataFrame,
          meta: dict, seed: int | None = None, note: str = "") -> dict:
    """probs: (n, n_types + 1) over type ids (column 0 = padding, ignored)."""
    probs = np.asarray(probs, dtype=float).copy()
    probs[:, 0] = -np.inf
    y_true = table["target_type"].to_numpy()
    y_pred = probs.argmax(1)
    top3 = np.argsort(-probs, axis=1)[:, :3]
    labels = sorted(set(y_true))                       # classes present in the test targets

    dist = to_court_units(xy_pred[:, 0] - table["target_x"].to_numpy(),
                          xy_pred[:, 1] - table["target_y"].to_numpy(), meta)
    return {
        "model": model,
        "seed": seed,
        "n": len(table),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "top3_accuracy": float(np.mean([t in row for t, row in zip(y_true, top3)])),
        "landing_error": float(dist.mean()),
        "note": note,
    }


def save_result(result: dict) -> None:
    """Append one row to results/metrics.csv (replacing an old row with same model + seed)."""
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([result])
    if RESULTS_CSV.exists():
        old = pd.read_csv(RESULTS_CSV)
        same = (old["model"] == result["model"]) & (old["seed"].fillna(-1) == (result["seed"] if result["seed"] is not None else -1))
        new = pd.concat([old[~same], new], ignore_index=True)
    new.to_csv(RESULTS_CSV, index=False)


def print_results() -> None:
    df = pd.read_csv(RESULTS_CSV)
    cols = ["accuracy", "macro_f1", "top3_accuracy", "landing_error"]
    summary = df.groupby("model", sort=False)[cols].agg(["mean", "std"]).round(3)
    print(summary.to_string(na_rep="-"))


if __name__ == "__main__":
    print_results()
