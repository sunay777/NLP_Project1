"""Single-generation training loop.

A `sampler` is any callable (batch_size, rng, device) -> batch dict. Real data
uses data.make_batch; a collapse generation uses a fixed TensorDataset.sample.
Loss is next-token cross-entropy over all supervised positions (targets == -100
elsewhere), which gives the dense induction signal the circuit needs.
"""
import math
import numpy as np
import torch
import torch.nn.functional as F

from .model import InductionTransformer
from .config import resolve_device
from . import metrics, probes
from .data import IGNORE


def loss_fn(model, batch):
    logits = model(batch["idx"])
    V = logits.shape[-1]
    return F.cross_entropy(logits.reshape(-1, V), batch["targets"].reshape(-1),
                           ignore_index=IGNORE)


def _lr_scale(step, warmup, total, decay):
    if step < warmup:
        return step / max(1, warmup)
    if not decay:
        return 1.0
    prog = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))


def train_model(cfg, sampler, val_sampler=None, probe_batch=None, log=True):
    device = resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    # evaluation draws from its OWN stream, so changing eval_every / val settings
    # never changes the training data order (previously they shared `rng`).
    eval_rng = np.random.default_rng(cfg.seed + 7_777_777)

    model = InductionTransformer(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: _lr_scale(s, cfg.warmup, cfg.steps, cfg.lr_decay))

    history = []
    for step in range(cfg.steps):
        model.train()
        batch = sampler(cfg.batch_size, rng, device)
        loss = loss_fn(model, batch)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % cfg.eval_every == 0 or step == cfg.steps - 1:
            vb = val_sampler(cfg.eval_batch, eval_rng, device) if val_sampler else batch
            rec = {
                "step": step,
                "train_loss": float(loss.item()),
                "query_acc": metrics.query_accuracy(model, vb),
                "query_ppl": metrics.query_perplexity(model, vb),
            }
            if probe_batch is not None:
                rec["induction"] = probes.induction_strength(model, probe_batch, cfg)
            history.append(rec)
            if log:
                ind = f" | induction={rec['induction']:.2f}" if "induction" in rec else ""
                print(f"  step {step:5d} | loss {rec['train_loss']:.4f} | "
                      f"query_acc {rec['query_acc']:.3f} | ppl {rec['query_ppl']:.2f}{ind}")
    return model, history
