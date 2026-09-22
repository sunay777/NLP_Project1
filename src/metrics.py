"""Metrics: accuracy, perplexity and — the collapse signal — generation diversity.

The collapse signal lives in the DIVERSITY of what the extended model generates
in its free slot (the next symbol sq2). We track the entropy of that symbol
distribution and its effective support; both shrink as the model collapses.
"""
import math
import numpy as np
import torch


@torch.no_grad()
def query_accuracy(model, batch):
    model.eval()
    logits = model(batch["idx"])
    B = batch["idx"].shape[0]
    rows = torch.arange(B, device=logits.device)
    pred = logits[rows, batch["query_pos"]].argmax(dim=-1)
    return (pred == batch["query_tgt"]).float().mean().item()


@torch.no_grad()
def query_perplexity(model, batch):
    model.eval()
    logits = model(batch["idx"])
    B = batch["idx"].shape[0]
    rows = torch.arange(B, device=logits.device)
    logp = torch.log_softmax(logits[rows, batch["query_pos"]], dim=-1)
    nll = -logp[rows, batch["query_tgt"]].mean().item()
    return math.exp(nll)


@torch.no_grad()
def position_accuracy(model, batch, pos):
    """Accuracy at a fixed sequence position against its target (for extended)."""
    model.eval()
    logits = model(batch["idx"])
    B = batch["idx"].shape[0]
    rows = torch.arange(B, device=logits.device)
    pred = logits[rows, pos].argmax(dim=-1)
    tgt = batch["targets"][:, pos]
    valid = tgt != -100
    if valid.sum() == 0:
        return float("nan")
    return (pred[valid] == tgt[valid]).float().mean().item()


def _counts(symbol_ids, n_symbols):
    arr = np.asarray(symbol_ids)
    # a well-trained model emits symbol ids in [0, n_symbols); early/collapsed
    # models can emit label-range tokens too, so size the histogram to fit both.
    size = max(n_symbols, int(arr.max()) + 1) if arr.size else n_symbols
    return np.bincount(arr, minlength=size).astype(np.float64), size


def distribution_stats(symbol_ids, n_symbols):
    counts, size = _counts(symbol_ids, n_symbols)
    p = counts / counts.sum()
    nz = p[p > 0]
    entropy = float(-(nz * np.log(nz)).sum())
    return {
        "entropy": entropy,
        "normalised_entropy": entropy / math.log(n_symbols),
        "effective_symbols": float(math.exp(entropy)),
        "unique_symbols": int((counts > 0).sum()),
    }


def kl_to_uniform(symbol_ids, n_symbols):
    counts, size = _counts(symbol_ids, n_symbols)
    p = counts / counts.sum()
    u = np.ones(size) / size
    mask = p > 0
    return float((p[mask] * np.log(p[mask] / u[mask])).sum())
