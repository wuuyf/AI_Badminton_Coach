"""Score a ShuttleNet model (trained with the official code) on our shared test set.

The official code is NOT part of this repo; it is imported from
third_party/CoachAI-Projects/Stroke Forecasting/ (see scripts/setup_data.sh).

Official setting encode_length=4:
  encoder input  : strokes 1-4
  decoder input  : stroke 5
  first prediction: stroke 6   <- what we score

The official code numbers shot types and players in its own order
(saved in the model's config file), so ids are mapped through names.

Usage:
  python -m badminton_coach.shuttlenet_eval --model_dir "<.../model/shuttlenet_seed42/1>" --seed 42
"""
import argparse
import ast
import sys
from pathlib import Path

import numpy as np
import torch

from badminton_coach.evaluate import load_processed, next_shot_table, save_result, score, print_results

SF_DIR = Path("third_party/CoachAI-Projects/Stroke Forecasting")


def load_model(model_dir: Path):
    sys.path.insert(0, str(SF_DIR.resolve()))
    from ShuttleNet.ShuttleNet import ShotGenEncoder, ShotGenPredictor   # official code

    config = ast.literal_eval((model_dir / "config").read_text())
    encoder, decoder = ShotGenEncoder(config), ShotGenPredictor(config)
    encoder.load_state_dict(torch.load(model_dir / "encoder", map_location="cpu"))
    decoder.load_state_dict(torch.load(model_dir / "decoder", map_location="cpu"))
    encoder.eval(), decoder.eval()
    return encoder, decoder, config


def predict(encoder, decoder, config, test, meta):
    enc_len = config["encode_length"]
    assert enc_len + 1 == meta["encode_length"], "ShuttleNet encode_length must be 4 (data.py uses 5)"

    # our id -> official id
    our_type_name = {v: k for k, v in meta["type_vocab"].items()}
    our_player_name = {v: k for k, v in meta["player_vocab"].items()}
    off_type = {name: i + 1 for i, name in enumerate(config["uniques_type"])}
    off_player = {name: i + 1 for i, name in enumerate(config["uniques_player"])}
    map_type = np.vectorize(lambda t: off_type[our_type_name[t]])
    map_player = np.vectorize(lambda p: off_player[our_player_name[p]])

    shot = torch.tensor(map_type(np.stack(test["ctx_type"])), dtype=torch.long)
    player = torch.tensor(map_player(np.stack(test["ctx_player"])), dtype=torch.long)
    x = torch.tensor(np.stack(test["ctx_x"]), dtype=torch.float)
    y = torch.tensor(np.stack(test["ctx_y"]), dtype=torch.float)
    target_player = torch.tensor(map_player(test["target_player"].to_numpy()), dtype=torch.long).unsqueeze(1)

    with torch.no_grad():
        enc = encoder(shot[:, :enc_len], x[:, :enc_len], y[:, :enc_len], player[:, :enc_len])
        out_xy, logits = decoder(shot[:, enc_len:], x[:, enc_len:], y[:, enc_len:],
                                 player[:, enc_len:], *enc, target_player)
    off_probs = torch.softmax(logits[:, -1, :], dim=-1).numpy()
    xy = out_xy[:, -1, :2].numpy()                 # mean of the predicted 2-D Gaussian

    # official columns -> our columns
    probs = np.zeros((len(test), len(meta["type_vocab"]) + 1))
    for name, our_id in meta["type_vocab"].items():
        if name in off_type:
            probs[:, our_id] = off_probs[:, off_type[name]]
    return probs, xy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()

    df, meta = load_processed()
    test = next_shot_table(df, "test", meta["encode_length"])
    encoder, decoder, config = load_model(args.model_dir)
    probs, xy = predict(encoder, decoder, config, test, meta)
    save_result(score("ShuttleNet", probs, xy, test, meta, seed=args.seed))
    print_results()


if __name__ == "__main__":
    main()
