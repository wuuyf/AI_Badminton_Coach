"""Own GRU model for next-stroke forecasting, with an input ablation.

Variants (--features):
  type             : shot type only
  type+loc         : + landing coordinates
  type+loc+player  : + player identity (who hits each stroke, and who hits next)

Training : every position of every training rally (predict stroke t+1 from strokes 1..t).
Validation: last 10% of training rallies in each match (early stopping; test set untouched).
Testing  : strokes 1-5 -> stroke 6, scored with the shared evaluate.py.

Landing: the model predicts the point in a "canonical" court half and the prediction is
rotated back with the half of stroke t-2 (same rule as the baselines), so every variant
is scored on the same footing.

Usage:
  python -m badminton_coach.gru_model --features type+loc+player --seeds 42 43 44
"""
import argparse
import copy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from badminton_coach.evaluate import (load_processed, next_shot_table, print_results,
                                      save_result, score, side_sign)

VAL_RATIO = 0.1
FIG_DIR = Path("results/figures")
MODEL_DIR = Path("model")                    # ignored by git
NAMES = {"type": "GRU (type)", "type+loc": "GRU (+loc)", "type+loc+player": "GRU (+loc+player)"}


# ---------------------------------------------------------------- data
def split_validation(df: pd.DataFrame) -> pd.DataFrame:
    """Move the last 10% of training rallies in each match to 'val'."""
    df = df.copy()
    for _, m in df[df["split"] == "train"].groupby("match_id", sort=False):
        rallies = m["rally_id"].unique()
        n_val = max(1, int(round(len(rallies) * VAL_RATIO)))
        df.loc[df["rally_id"].isin(rallies[-n_val:]), "split"] = "val"
    return df


def rally_tensors(df: pd.DataFrame, split: str) -> dict:
    """Padded arrays for all rallies of a split (inputs = strokes 1..n-1, targets = 2..n)."""
    rallies = [r for _, r in df[df["split"] == split].groupby("rally_id", sort=False) if len(r) >= 2]
    L = max(len(r) for r in rallies) - 1
    N = len(rallies)
    a = {k: np.zeros((N, L), dtype=np.int64) for k in ["type", "player", "next_player", "tgt_type"]}
    a.update({k: np.zeros((N, L), dtype=np.float32) for k in ["x", "y", "tgt_cx", "tgt_cy", "xy_mask"]})
    for i, r in enumerate(rallies):
        t, p = r["type_id"].to_numpy(), r["player_id"].to_numpy()
        x, y = r["landing_x"].to_numpy(), r["landing_y"].to_numpy()
        n = len(r) - 1
        a["type"][i, :n], a["player"][i, :n], a["x"][i, :n], a["y"][i, :n] = t[:-1], p[:-1], x[:-1], y[:-1]
        a["next_player"][i, :n], a["tgt_type"][i, :n] = p[1:], t[1:]
        # canonical landing target for stroke k (k >= 3): rotate by the half of stroke k-2
        for j in range(1, n):                      # target index j -> stroke j+2, two before = stroke j
            s = side_sign([y[j - 1]])[0]
            a["tgt_cx"][i, j], a["tgt_cy"][i, j] = x[j + 1] * s, y[j + 1] * s
            a["xy_mask"][i, j] = 1.0
    return {k: torch.from_numpy(v) for k, v in a.items()}


def test_tensors(test: pd.DataFrame) -> dict:
    t = {
        "type": np.stack(test["ctx_type"]), "player": np.stack(test["ctx_player"]),
        "x": np.stack(test["ctx_x"]).astype(np.float32), "y": np.stack(test["ctx_y"]).astype(np.float32),
    }
    nxt = np.zeros_like(t["player"])
    nxt[:, :-1] = t["player"][:, 1:]
    nxt[:, -1] = test["target_player"].to_numpy()
    t["next_player"] = nxt
    return {k: torch.from_numpy(v) for k, v in t.items()}


# ---------------------------------------------------------------- model
class StrokeGRU(nn.Module):
    def __init__(self, n_types: int, n_players: int, use_loc: bool, use_player: bool,
                 emb: int = 32, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.use_loc, self.use_player = use_loc, use_player
        self.type_emb = nn.Embedding(n_types, emb, padding_idx=0)
        in_dim = emb
        if use_loc:
            self.loc = nn.Linear(2, 16)
            in_dim += 16
        if use_player:
            self.player_emb = nn.Embedding(n_players, 16, padding_idx=0)
            in_dim += 16
        self.gru = nn.GRU(in_dim, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        head_in = hidden + (16 if use_player else 0)        # + who hits next
        self.type_head = nn.Linear(head_in, n_types)
        self.xy_head = nn.Linear(head_in, 2)

    def forward(self, b: dict):
        parts = [self.type_emb(b["type"])]
        if self.use_loc:
            parts.append(torch.relu(self.loc(torch.stack([b["x"], b["y"]], -1))))
        if self.use_player:
            parts.append(self.player_emb(b["player"]))
        h, _ = self.gru(self.drop(torch.cat(parts, -1)))
        h = self.drop(h)
        if self.use_player:
            h = torch.cat([h, self.player_emb(b["next_player"])], -1)
        return self.type_head(h), self.xy_head(h)


def loss_fn(model, b):
    logits, xy = model(b)
    mask = b["tgt_type"] > 0
    ce = nn.functional.cross_entropy(logits[mask], b["tgt_type"][mask])
    m = b["xy_mask"]
    se = ((xy[..., 0] - b["tgt_cx"]) ** 2 + (xy[..., 1] - b["tgt_cy"]) ** 2) * m
    return ce + se.sum() / m.sum().clamp(min=1), ce


def batches(data: dict, size: int, shuffle: bool, gen: torch.Generator):
    n = len(data["type"])
    idx = torch.randperm(n, generator=gen) if shuffle else torch.arange(n)
    for i in range(0, n, size):
        j = idx[i:i + size]
        yield {k: v[j] for k, v in data.items()}


def train_one(features: str, seed: int, train: dict, val: dict, meta: dict,
              epochs: int = 200, patience: int = 15, lr: float = 1e-3, bs: int = 64):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = StrokeGRU(len(meta["type_vocab"]) + 1, len(meta["player_vocab"]) + 1,
                      use_loc="loc" in features, use_player="player" in features)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = {"train": [], "val": []}
    best, best_state, wait = float("inf"), None, 0

    for ep in range(epochs):
        model.train()
        tot, nb = 0.0, 0
        for b in batches(train, bs, True, gen):
            opt.zero_grad()
            loss, _ = loss_fn(model, b)
            loss.backward()
            opt.step()
            tot, nb = tot + loss.item(), nb + 1
        model.eval()
        with torch.no_grad():
            vloss, _ = loss_fn(model, val)
        hist["train"].append(tot / nb)
        hist["val"].append(vloss.item())
        if vloss.item() < best - 1e-4:
            best, best_state, wait = vloss.item(), copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    return model, hist


def plot_history(hist: dict, title: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    best = int(np.argmin(hist["val"]))
    plt.figure(figsize=(7, 4))
    plt.plot(hist["train"], label="train")
    plt.plot(hist["val"], label="validation")
    plt.axvline(best, ls="--", c="grey", label=f"best epoch {best + 1}")
    plt.xlabel("epoch"), plt.ylabel("loss (type CE + landing MSE)"), plt.title(title)
    plt.legend(), plt.tight_layout(), plt.savefig(path, dpi=150), plt.close()


def predict(model, test: pd.DataFrame):
    model.eval()
    with torch.no_grad():
        logits, xy = model(test_tensors(test))
    probs = torch.softmax(logits[:, -1, :], -1).numpy()
    s = side_sign([y[-2] for y in test["ctx_y"]])
    return probs, xy[:, -1, :].numpy() * s[:, None]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", choices=list(NAMES), required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = ap.parse_args()

    df, meta = load_processed()
    df = split_validation(df)
    train, val = rally_tensors(df, "train"), rally_tensors(df, "val")
    test = next_shot_table(df, "test", meta["encode_length"])
    print(f"Rallies  train: {len(train['type'])}  val: {len(val['type'])}  test samples: {len(test)}")

    name = NAMES[args.features]
    for seed in args.seeds:
        model, hist = train_one(args.features, seed, train, val, meta)
        tag = f"gru_{args.features.replace('+', '_')}_seed{seed}"
        plot_history(hist, f"{name}, seed {seed}", FIG_DIR / f"{tag}.png")
        MODEL_DIR.mkdir(exist_ok=True)
        torch.save(model.state_dict(), MODEL_DIR / f"{tag}.pt")
        probs, xy = predict(model, test)
        save_result(score(name, probs, xy, test, meta, seed=seed,
                          note=f"best epoch {int(np.argmin(hist['val'])) + 1}"))
        print(f"  seed {seed}: stopped after {len(hist['val'])} epochs, best {int(np.argmin(hist['val'])) + 1}")

    print()
    print_results()


if __name__ == "__main__":
    main()
