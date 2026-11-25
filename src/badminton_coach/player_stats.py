"""Per-player habit statistics from the TRAINING rallies only (no test leakage).

What a coach would look at, computed from the data:
  shot_mix        : which shot types the player hits most (services excluded)
  replies         : given the opponent's previous shot, how the player answers
  attack_share    : share of attacking shots (ATTACK_TYPES) among non-service strokes
  winners         : the player's rally-ending winning shots, by type
  errors          : the player's own rally-ending errors, by type and reason
  conceded        : opponent winning shots that ended rallies against the player, by type

Used by: the grounded Gemini forecaster, the tactical-advice prompt and the demo app.
Output : results/player_stats.json  (aggregate counts only, safe to publish)

Usage:
  python -m badminton_coach.player_stats                     # build the JSON
  python -m badminton_coach.player_stats --player "Viktor AXELSEN"   # print one profile
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from badminton_coach.evaluate import load_processed

RAW_MATCH_CSV = Path("data/raw/match.csv")
OUT = Path("results/player_stats.json")
SERVICES = {"short service", "long service"}
ATTACK_TYPES = {"smash", "push/rush"}          # adjust with a coach's judgement
MIN_N = 10                                      # minimum count before a reply pattern is reported

# lose_reason labels in ShuttleSet (Chinese) -> English
REASONS = {"對手落地致勝": "opponent winner", "出界": "out", "掛網": "net", "未過網": "not over net",
           "落點判斷失誤": "misjudged landing", "對手落地判斷失誤": "opponent misjudged", "犯規": "fault"}


def add_point_winner(df: pd.DataFrame) -> pd.DataFrame:
    """getpoint_player is 'A'/'B' (A = match winner); map it to the player's name."""
    m = pd.read_csv(RAW_MATCH_CSV)[["id", "winner", "loser"]].rename(columns={"id": "match_id"})
    df = df.merge(m, on="match_id", how="left")
    df["point_winner"] = df["getpoint_player"].map({"A": 0, "B": 1})
    df["point_winner"] = df.apply(
        lambda r: (r["winner"], r["loser"])[int(r["point_winner"])] if pd.notna(r["point_winner"]) else None, axis=1)
    return df.drop(columns=["winner", "loser"])


def shares(counter: Counter, top: int | None = None) -> dict:
    n = sum(counter.values())
    items = counter.most_common(top)
    return {k: round(v / n, 3) for k, v in items} if n else {}


def compute(df: pd.DataFrame) -> dict:
    tr = df[df["split"] == "train"].sort_values(["rally_id", "ball_round"]).copy()
    tr["prev_type"] = tr.groupby("rally_id")["type_en"].shift(1)
    last = tr.groupby("rally_id").tail(1)

    stats = {}
    for player, g in tr.groupby("player"):
        rally_play = g[~g["type_en"].isin(SERVICES)]
        replies = {}
        for incoming, h in g.dropna(subset=["prev_type"]).groupby("prev_type"):
            if len(h) >= MIN_N:
                replies[incoming] = {"n": len(h), "top": shares(Counter(h["type_en"]), 3)}

        ended = last[last["rally_id"].isin(g["rally_id"])]
        own_last = ended[ended["player"] == player]
        opp_last = ended[ended["player"] != player]
        winners = own_last[own_last["point_winner"] == player]
        errors = own_last[own_last["point_winner"] != player]
        conceded = opp_last[opp_last["point_winner"] != player]

        stats[player] = {
            "n_rallies": int(g["rally_id"].nunique()),
            "n_strokes": int(len(g)),
            "shot_mix": shares(Counter(rally_play["type_en"]), 5),
            "attack_share": round(rally_play["type_en"].isin(ATTACK_TYPES).mean(), 3),
            "replies": replies,
            "winners": dict(Counter(winners["type_en"]).most_common(5)),
            "errors": {t: {"n": int(len(e)), "reasons": dict(Counter(e["lose_reason"].map(REASONS).fillna("unknown")))}
                       for t, e in errors.groupby("type_en")},
            "conceded": dict(Counter(conceded["type_en"]).most_common(5)),
        }
    return stats


def profile_text(stats: dict, player: str, incoming: str | None = None) -> str:
    """Compact English text for an LLM prompt."""
    s = stats[player]
    lines = [f"{player} (training data: {s['n_rallies']} rallies, {s['n_strokes']} strokes)",
             "- most frequent shots: " + ", ".join(f"{k} {v:.0%}" for k, v in s["shot_mix"].items()),
             f"- attacking share (smash + push/rush): {s['attack_share']:.0%}"]
    if incoming and incoming in s["replies"]:
        r = s["replies"][incoming]
        lines.append(f"- after the opponent's {incoming} (n={r['n']}): "
                     + ", ".join(f"{k} {v:.0%}" for k, v in r["top"].items()))
    if s["winners"]:
        lines.append("- wins rallies most with: " + ", ".join(f"{k} ({v})" for k, v in s["winners"].items()))
    if s["errors"]:
        top_err = sorted(s["errors"].items(), key=lambda kv: -kv[1]["n"])[:3]
        lines.append("- own errors most on: " + ", ".join(f"{k} ({v['n']})" for k, v in top_err))
    if s["conceded"]:
        lines.append("- loses rallies most to opponent's: " + ", ".join(f"{k} ({v})" for k, v in s["conceded"].items()))
    if s["n_rallies"] < 50:
        lines.append("- note: limited data for this player")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--player")
    args = ap.parse_args()

    df, _ = load_processed()
    stats = compute(add_point_winner(df))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {OUT} ({len(stats)} players)\n")

    if args.player:
        print(profile_text(stats, args.player))
    else:
        summary = pd.DataFrame({p: {"rallies": s["n_rallies"], "attack": s["attack_share"],
                                    "top shot": next(iter(s["shot_mix"]), "")} for p, s in stats.items()}).T
        print(summary.sort_values("rallies", ascending=False).to_string())


if __name__ == "__main__":
    main()
