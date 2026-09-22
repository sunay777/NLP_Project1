"""Milestone 1: train a single generation and verify an induction head forms.

Run:  python -m experiments.run_single --mode base
Expect query accuracy to approach ~1.0 and a clearly identified induction head
with strength close to 1.0.
"""
import argparse
import numpy as np

from src.config import Config, resolve_device
from src import data as datamod
from src.train import train_model
from src import probes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="base", choices=["base", "extended"])
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = Config(mode=args.mode, steps=args.steps, seed=args.seed)
    device = resolve_device(cfg.device)
    print(f"device={device}  mode={cfg.mode}  vocab={cfg.vocab_size}  seq_len={cfg.max_len}")

    rng = np.random.default_rng(cfg.seed)
    real = lambda bs, r, dev: datamod.make_batch(cfg, bs, r, dev)
    probe_batch = datamod.make_batch(cfg, cfg.eval_batch, rng, device)

    model, history = train_model(cfg, real, val_sampler=real, probe_batch=probe_batch)
    print(f"\nmodel params: {model.num_params():,}")

    scores = probes.head_scores(model, probe_batch, cfg)
    picked = probes.pick_heads(scores)
    print("\nHead identification:")
    pt = picked["prev_token_head"]
    ih = picked["induction_head"]
    print(f"  previous-token head: layer {pt[0]} head {pt[1]}  score={pt[2]:.3f}")
    print(f"  induction head:      layer {ih[0]} head {ih[1]}  score={ih[2]:.3f}")


if __name__ == "__main__":
    main()
