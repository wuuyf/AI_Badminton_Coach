"""Zero-shot LLM baseline: ask Gemini to forecast stroke 6 from strokes 1-5.

Same test rallies and same metrics as every other model (evaluate.py).
Gemini sees only strokes 1-5 of each test rally (no labels, no training data).
Rallies are sent in batches to save requests; answers are cached
(results/advice_cache.json), so re-running costs no extra requests.

Usage:
  python -m badminton_coach.gemini_forecast --dry-run      # print one prompt, no API call
  python -m badminton_coach.gemini_forecast                # run (about 12 requests)
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

from badminton_coach import llm_client
from badminton_coach.evaluate import load_processed, next_shot_table, print_results, save_result, score

BATCH_SIZE = 30
SECONDS_BETWEEN_CALLS = 13          # free tier: 5 requests per minute

INSTRUCTIONS = """You are a badminton analyst. For each men's or women's singles rally below,
you see the first {k} strokes. Predict stroke {k1} (the next shot).

Shot types (use exactly these names):
{types}

Coordinates: one fixed frame for the whole court, in court units.
x runs across the court (about 0 to 355), y runs along the court (about 85 to 850).
The net is at about y = 467; y < 467 is one half, y > 467 is the other half,
and the distance from 467 is the depth from the net.
A "landing" is where that stroke's shuttle landed.

For each rally return:
- "top3": the three most likely shot types for stroke {k1}, most likely first
- "probs": their probabilities (three numbers, sum <= 1)
- "landing": predicted [x, y] landing point of stroke {k1}

Answer with JSON only: a list of objects {{"id": ..., "top3": [...], "probs": [...], "landing": [x, y]}},
one per rally, same ids as given.

Rallies:
{rallies}"""


def to_court(v: float, axis: str, meta: dict) -> float:
    return v * meta["coord_std"][axis] + meta["coord_mean"][axis]


def to_std(v: float, axis: str, meta: dict) -> float:
    return (v - meta["coord_mean"][axis]) / meta["coord_std"][axis]


def describe_rally(row: pd.Series, meta: dict) -> dict:
    id2type = {v: meta["type_en"][k] for k, v in meta["type_vocab"].items()}
    id2player = {v: k for k, v in meta["player_vocab"].items()}
    strokes = [{
        "stroke": i + 1,
        "hitter": id2player[p],
        "type": id2type[t],
        "landing": [round(to_court(x, "x", meta)), round(to_court(y, "y", meta))],
    } for i, (t, p, x, y) in enumerate(zip(row["ctx_type"], row["ctx_player"], row["ctx_x"], row["ctx_y"]))]
    return {"id": int(row["rally_id"]), "strokes": strokes,
            "next_hitter": id2player[row["target_player"]]}


def build_prompt(batch: pd.DataFrame, meta: dict) -> str:
    k = meta["encode_length"]
    rallies = [describe_rally(r, meta) for _, r in batch.iterrows()]
    return INSTRUCTIONS.format(k=k, k1=k + 1, types=", ".join(meta["type_en"].values()),
                               rallies=json.dumps(rallies, ensure_ascii=False))


def parse(text: str, batch: pd.DataFrame, meta: dict, probs: np.ndarray, xy: np.ndarray,
          index: dict) -> int:
    """Fill probs / xy rows for this batch; return the number of rallies Gemini failed to answer."""
    en2id = {meta["type_en"][k]: v for k, v in meta["type_vocab"].items()}
    answered = 0
    for item in json.loads(text):
        i = index.get(int(item.get("id", -1)))
        if i is None:
            continue
        top3, p = item.get("top3", []), item.get("probs", [])
        ids = [en2id.get(str(t).strip().lower()) for t in top3]
        if not ids or ids[0] is None:
            continue
        rest = max(0.0, 1 - sum(p[:3])) if len(p) >= 3 else 0.1
        probs[i, 1:] = rest / (probs.shape[1] - 1) * 1e-3   # tiny mass elsewhere
        for rank, t in enumerate(ids):
            if t is not None:
                probs[i, t] = (p[rank] if rank < len(p) else 0.01) + (3 - rank) * 1e-6
        lx, ly = item.get("landing", [None, None])
        if lx is not None and ly is not None:
            xy[i] = [to_std(float(lx), "x", meta), to_std(float(ly), "y", meta)]
        answered += 1
    return len(batch) - answered


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--model", default=llm_client.DEFAULT_MODEL)
    args = ap.parse_args()

    df, meta = load_processed()
    test = next_shot_table(df, "test", meta["encode_length"]).reset_index(drop=True)
    batches = [test.iloc[i:i + BATCH_SIZE] for i in range(0, len(test), BATCH_SIZE)]
    if args.dry_run:
        print(build_prompt(batches[0].head(2), meta))
        print(f"\n{len(test)} rallies -> {len(batches)} requests")
        return

    n_cls = len(meta["type_vocab"]) + 1
    probs = np.zeros((len(test), n_cls))
    xy = np.zeros((len(test), 2))                         # unanswered -> centre of court
    index = {int(r): i for i, r in enumerate(test["rally_id"])}
    missing = 0
    for b, batch in enumerate(batches, 1):
        prompt = build_prompt(batch, meta)
        before = len(llm_client._load_cache())
        text = llm_client.generate(prompt, model=args.model, json_mode=True)
        missing += parse(text, batch, meta, probs, xy, index)
        print(f"batch {b}/{len(batches)} done")
        if len(llm_client._load_cache()) > before and b < len(batches):
            time.sleep(SECONDS_BETWEEN_CALLS)             # only wait after a real API call

    print(f"Unanswered rallies (scored as wrong): {missing}")
    save_result(score(f"Gemini ({args.model}, zero-shot)", probs, xy, test, meta,
                      note=f"unanswered={missing}"))
    print_results()


if __name__ == "__main__":
    main()
