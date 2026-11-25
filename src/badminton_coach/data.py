"""Load the cleaned ShuttleSet strokes, split train/test, and build next-shot samples.

Input  : dataset.csv produced by the official ShuttleNet cleaner
         (third_party/CoachAI-Projects/Stroke Forecasting/data/dataset.csv).
         Using the same file for every model keeps cleaning and coordinates identical.
Output : data/processed/strokes.parquet  - all strokes with split + encoded ids
         data/processed/meta.json         - vocabularies and settings

Split (same as the official code): within each match, the first 80% of rallies
-> train, the remaining 20% -> test.
Task: given the first 5 strokes of a rally, predict stroke 6.
(= official ShuttleNet setting encode_length=4: strokes 1-4 encoded, stroke 5 is the
first decoder input, stroke 6 is the first prediction.)
"""
import argparse
import json
from pathlib import Path

import pandas as pd

DEFAULT_INPUT = Path("third_party/CoachAI-Projects/Stroke Forecasting/data/dataset.csv")
OUT_DIR = Path("data/processed")
ENCODE_LENGTH = 5          # observed strokes before the predicted one
TRAIN_RATIO = 0.8

# 10 merged shot types used by the official ShuttleNet preprocessing
SHOT_TYPES = {
    "發短球": "short service",
    "發長球": "long service",
    "網前球": "net shot",
    "推撲球": "push/rush",
    "挑球": "lob",
    "長球": "clear",
    "殺球": "smash",
    "切球": "drop",
    "平球": "drive",
    "接殺防守": "defensive return",
}

# Inverse of the official standardisation, for plotting / distance in court units
COORD_MEAN = {"x": 175.0, "y": 467.0}
COORD_STD = {"x": 82.0, "y": 192.0}


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    unknown = set(df["type"].unique()) - set(SHOT_TYPES)
    if unknown:
        raise ValueError(f"Unexpected shot types: {unknown}")
    return df   # keep file order: the official split depends on it


def add_split(df: pd.DataFrame) -> pd.DataFrame:
    """First 80% of rallies in each match -> train, rest -> test."""
    split = {}
    for _, match in df.groupby("match_id", sort=False):
        rallies = match["rally_id"].unique()          # file order, exactly as the official code
        cut = int(len(rallies) * TRAIN_RATIO)
        split.update({r: "train" for r in rallies[:cut]})
        split.update({r: "test" for r in rallies[cut:]})
    df = df.assign(split=df["rally_id"].map(split))
    return df.sort_values(["match_id", "rally_id", "ball_round"], kind="stable").reset_index(drop=True)


def encode(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Integer ids, 0 reserved for padding."""
    type_vocab = {t: i + 1 for i, t in enumerate(SHOT_TYPES)}
    players = sorted(df["player"].unique())
    player_vocab = {p: i + 1 for i, p in enumerate(players)}
    df = df.assign(type_id=df["type"].map(type_vocab),
                   player_id=df["player"].map(player_vocab),
                   type_en=df["type"].map(SHOT_TYPES))
    meta = {
        "encode_length": ENCODE_LENGTH,
        "train_ratio": TRAIN_RATIO,
        "type_vocab": type_vocab,
        "type_en": SHOT_TYPES,
        "player_vocab": player_vocab,
        "coord_mean": COORD_MEAN,
        "coord_std": COORD_STD,
    }
    return df, meta


def summarize(df: pd.DataFrame) -> None:
    lengths = df.groupby("rally_id").agg(n=("ball_round", "size"), split=("split", "first"))
    print(f"Matches: {df['match_id'].nunique()}  Players: {df['player'].nunique()}")
    print(f"Rallies: {len(lengths)}  Strokes: {len(df)}")
    for s in ["train", "test"]:
        part = lengths[lengths["split"] == s]
        usable = (part["n"] > ENCODE_LENGTH).sum()
        print(f"  {s:5s}: {len(part)} rallies, {usable} with > {ENCODE_LENGTH} strokes (next-shot samples)")

    # Players seen only in test would make the player embedding meaningless
    train_players = set(df.loc[df["split"] == "train", "player"])
    test_only = set(df.loc[df["split"] == "test", "player"]) - train_players
    print(f"Players only in test: {sorted(test_only) or 'none'}")

    # Class distribution of the prediction target in test
    target = df[(df["split"] == "test") & (df["ball_round"] == ENCODE_LENGTH + 1)]
    print(f"\nTest targets (stroke {ENCODE_LENGTH + 1}) by shot type:")
    print(target["type_en"].value_counts().to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    args = ap.parse_args()

    df = add_split(load(args.input))
    df, meta = encode(df)
    summarize(df)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_DIR / "strokes.parquet", index=False)
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved {OUT_DIR / 'strokes.parquet'} and {OUT_DIR / 'meta.json'}")


if __name__ == "__main__":
    main()
