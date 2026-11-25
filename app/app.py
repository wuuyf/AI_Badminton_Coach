"""AI Badminton Coach - demo app.

Run from the project root, with the .venv activated:
  python -m streamlit run app/app.py
(Use "python -m" so Streamlit runs with this project's .venv, not another Python.)

Needs: data/processed/ (badminton_coach.data), results/player_stats.json
(badminton_coach.player_stats) and a trained GRU in model/ (scripts/train_gru.sh).
Only TEST rallies are shown, so every prediction is on data the models never trained on.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

from badminton_coach import baselines, court
from badminton_coach.evaluate import load_processed, next_shot_table
from badminton_coach.gru_model import StrokeGRU, predict as gru_predict
from badminton_coach.player_stats import profile_text

MODEL_PATH = Path("model/gru_type_loc_player_seed42.pt")
STATS_PATH = Path("results/player_stats.json")
MATCH_CSV = Path("data/raw/match.csv")
GRU_LANDING_ERROR = 114.1          # mean test landing error of the GRU (court units), from results/metrics.csv

st.set_page_config(page_title="AI Badminton Coach", page_icon="🏸", layout="wide")


# ---------------------------------------------------------------- loading
@st.cache_data
def load_data():
    df, meta = load_processed()
    matches = pd.read_csv(MATCH_CSV).set_index("id")
    stats = json.loads(STATS_PATH.read_text(encoding="utf-8"))
    return df, meta, matches, stats


@st.cache_resource
def load_models(_df, _meta):
    gru = StrokeGRU(len(_meta["type_vocab"]) + 1, len(_meta["player_vocab"]) + 1,
                    use_loc=True, use_player=True)
    gru.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    gru.eval()
    pairs = baselines.training_pairs(_df)
    return gru, pairs


df, meta, matches, stats = load_data()
gru, pairs = load_models(df, meta)
id2type = {v: meta["type_en"][k] for k, v in meta["type_vocab"].items()}
id2player = {v: k for k, v in meta["player_vocab"].items()}
n_cls = len(meta["type_vocab"]) + 1


# ---------------------------------------------------------------- sidebar: pick a rally
st.sidebar.title("🏸 AI Badminton Coach")
test = df[df["split"] == "test"]
lengths = test.groupby("rally_id").size()
test = test[test["rally_id"].isin(lengths[lengths >= 3].index)]

match_ids = test["match_id"].unique()
def match_label(mid):
    r = matches.loc[mid]
    return f"{r['winner']} vs {r['loser']} · {r['tournament']} {r['year']} {r['round']}"
mid = st.sidebar.selectbox("Match", match_ids, format_func=match_label)

rallies = test[test["match_id"] == mid].groupby("rally_id", sort=False).agg(
    set=("set", "first"), rally=("rally", "first"), n=("ball_round", "size"))
rid = st.sidebar.selectbox(
    "Rally (test set)", rallies.index,
    format_func=lambda r: f"Set {rallies.loc[r, 'set']} · rally {rallies.loc[r, 'rally']} · {rallies.loc[r, 'n']} strokes")
n_strokes = int(rallies.loc[rid, "n"])
k = st.sidebar.slider("Strokes observed", 2, n_strokes - 1, min(5, n_strokes - 1))
reveal = st.sidebar.checkbox("Reveal the actual next stroke", value=False)
st.sidebar.caption("Models: GRU (+location +player), Markov baseline. "
                   "Only test rallies are shown.")


# ---------------------------------------------------------------- predictions
row = next_shot_table(df[df["rally_id"] == rid], "test", encode_length=k)
gru_probs, gru_xy = gru_predict(gru, row)
mk_probs, _ = baselines.markov(pairs, row, n_cls)

r = row.iloc[0]
hitters = [id2player[p] for p in r["ctx_player"]]
types = [id2type[t] for t in r["ctx_type"]]
xs = list(np.array(r["ctx_x"]) * meta["coord_std"]["x"] + meta["coord_mean"]["x"])
ys = list(np.array(r["ctx_y"]) * meta["coord_std"]["y"] + meta["coord_mean"]["y"])
px = gru_xy[0, 0] * meta["coord_std"]["x"] + meta["coord_mean"]["x"]
py = gru_xy[0, 1] * meta["coord_std"]["y"] + meta["coord_mean"]["y"]
tx = r["target_x"] * meta["coord_std"]["x"] + meta["coord_mean"]["x"]
ty = r["target_y"] * meta["coord_std"]["y"] + meta["coord_mean"]["y"]
next_hitter = id2player[r["target_player"]]
actual = id2type[r["target_type"]]


def top3(probs):
    p = probs[0].copy()
    p[0] = -1
    idx = np.argsort(-p)[:3]
    return pd.DataFrame({"shot": [id2type[i] for i in idx], "probability": p[idx]})


# ---------------------------------------------------------------- layout
st.subheader(f"Stroke {k + 1}: what will {next_hitter} play next?")
left, right = st.columns([1, 1.3])

with left:
    fig, ax = plt.subplots(figsize=(4.2, 8))
    court.draw_court(ax)
    court.draw_rally(ax, xs, ys, hitters, types, hitters[0])
    court.draw_prediction(ax, px, py, GRU_LANDING_ERROR, true_xy=(tx, ty) if reveal else None)
    st.pyplot(fig)
    st.caption("Each dot = where a shot **landed**; number = stroke order; colour = who **hit** it "
               f"(🔵 {hitters[0]}, 🟡 {hitters[1]}). ⭐ = predicted landing of the next shot, "
               "circle = the model's average error (about 2 m)." + (" ❌ = actual landing." if reveal else ""))

with right:
    st.subheader("Next-shot prediction")
    c1, c2 = st.columns(2)
    for col, name, probs in [(c1, "GRU (deep learning)", gru_probs), (c2, "Markov (last shot only)", mk_probs)]:
        with col:
            st.markdown(f"**{name}**")
            t = top3(probs)
            for _, x in t.iterrows():
                mark = " ✅" if reveal and x["shot"] == actual else ""
                st.progress(float(x["probability"]), text=f"{x['shot']} · {x['probability']:.0%}{mark}")
    if reveal:
        st.success(f"Actual stroke {k + 1}: **{actual}** by {next_hitter}")

    st.subheader(f"{next_hitter}: habits from training data")
    profile = profile_text(stats, next_hitter, incoming=types[-1]).split("\n")
    st.markdown(f"*{profile[0]}*\n\n" + "\n".join(profile[1:]))

    st.subheader("Coach's advice")
    st.info("LLM-generated tactical advice will appear here (in progress).")

with st.expander("Strokes so far"):
    st.table(pd.DataFrame({"stroke": range(1, k + 1), "hitter": hitters, "shot": types,
                           "landing x": np.round(xs), "landing y": np.round(ys)}).set_index("stroke"))
