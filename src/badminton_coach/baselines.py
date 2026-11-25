"""Two simple baselines. A learned model should clearly beat both.

1. Most-frequent : always predicts the shot-type distribution of the training set;
                   landing = average (canonicalised) landing point.
2. Markov        : P(next type | previous type), estimated from all consecutive
                   stroke pairs in training rallies (add-one smoothing);
                   landing = average (canonicalised) landing point given the previous type.

Training uses every stroke t >= 3 of every training rally (t-2 is needed to know
which half of the court the shot lands on). Testing uses stroke 5 only.
"""
import numpy as np
import pandas as pd

from badminton_coach.evaluate import (load_processed, next_shot_table, print_results,
                                      save_result, score, side_sign)


def training_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """(prev_type, next_type, canonical next landing) for every stroke t >= 3 in train."""
    tr = df[df["split"] == "train"].copy()
    g = tr.groupby("rally_id", sort=False)
    tr["prev_type"] = g["type_id"].shift(1)
    tr["y_two_before"] = g["landing_y"].shift(2)
    tr = tr.dropna(subset=["prev_type", "y_two_before"])
    s = side_sign(tr["y_two_before"].to_numpy())
    return pd.DataFrame({
        "prev_type": tr["prev_type"].astype(int).to_numpy(),
        "next_type": tr["type_id"].to_numpy(),
        "cx": tr["landing_x"].to_numpy() * s,
        "cy": tr["landing_y"].to_numpy() * s,
    })


def most_frequent(pairs: pd.DataFrame, test: pd.DataFrame, n_cls: int):
    counts = np.bincount(pairs["next_type"], minlength=n_cls).astype(float)
    probs = np.tile(counts / counts.sum(), (len(test), 1))
    mean_xy = pairs[["cx", "cy"]].mean().to_numpy()
    s = side_sign([y[2] for y in test["ctx_y"]])          # stroke 3 -> half of stroke 5
    return probs, np.outer(s, mean_xy)


def markov(pairs: pd.DataFrame, test: pd.DataFrame, n_cls: int):
    trans = np.ones((n_cls, n_cls))                         # add-one smoothing
    np.add.at(trans, (pairs["prev_type"], pairs["next_type"]), 1)
    trans[:, 0] = 0                                         # never predict padding
    trans /= trans.sum(1, keepdims=True)

    mean_xy = pairs.groupby("prev_type")[["cx", "cy"]].mean()
    overall = pairs[["cx", "cy"]].mean().to_numpy()

    prev = np.array([t[-1] for t in test["ctx_type"]])
    probs = trans[prev]
    xy = np.array([mean_xy.loc[p].to_numpy() if p in mean_xy.index else overall for p in prev])
    s = side_sign([y[2] for y in test["ctx_y"]])
    return probs, xy * s[:, None]


def main() -> None:
    df, meta = load_processed()
    n_cls = len(meta["type_vocab"]) + 1                     # + padding id 0
    pairs = training_pairs(df)
    test = next_shot_table(df, "test", meta["encode_length"])
    print(f"Training pairs: {len(pairs)}  Test samples: {len(test)}\n")

    for name, fn in [("Most-frequent", most_frequent), ("Markov", markov)]:
        probs, xy = fn(pairs, test, n_cls)
        save_result(score(name, probs, xy, test, meta))

    print_results()


if __name__ == "__main__":
    main()
