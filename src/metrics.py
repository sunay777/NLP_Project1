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


# --------------------------------------------------------------------------- #
# Conditional (per-context) diversity — the collapse signal that matters.
#
# sq2 is always one of the K in-context symbols, and those are a uniform draw
# from n_symbols, so the MARGINAL histogram over symbol ids stays ~uniform even
# if the model always picks, e.g., the first study slot. Collapse of the free
# choice therefore has to be measured per context:
#   cond_entropy  mean_b H(p(sq2 | context_b) renormalised over the K in-context
#                 symbols) / ln K         (1 = uniform choice, 0 = deterministic)
#   off_context   probability mass the model puts OUTSIDE the K in-context
#                 symbols (hallucinated symbols or label tokens)
#   slot_entropy  entropy of WHICH study slot the sampled sq2 came from, / ln K
#                 (detects positional bias, e.g. 'always copy the first pair')
# --------------------------------------------------------------------------- #
@torch.no_grad()
def conditional_diversity(model, batch, cfg):
    K = cfg.n_pairs
    model.eval()
    logits = model(batch["idx"])
    p = torch.softmax(logits[:, 4 * K - 1], dim=-1)                     # (B,V)
    classes = batch["idx"][:, 0:2 * K:2]                                # study symbols (B,K)
    in_ctx = p.gather(1, classes)                                       # (B,K)
    mass = in_ctx.sum(1)
    q = (in_ctx / mass[:, None]).clamp_min(1e-12)
    H = -(q * q.log()).sum(1)
    return {
        "cond_entropy": (H.mean() / math.log(K)).item(),
        "cond_entropy_p10": (torch.quantile(H, 0.10) / math.log(K)).item(),
        "off_context": (1 - mass).mean().item(),
    }


def slot_stats(sampled_sym, idx, K):
    """Which study slot each sampled sq2 came from; entropy over slots / ln K."""
    sym = torch.as_tensor(sampled_sym).to(idx.device)
    study = idx[:, 0:2 * K:2]
    hit = study == sym[:, None]                                         # (B,K)
    in_ctx = hit.any(1)
    slot = hit.float().argmax(1)[in_ctx].cpu().numpy()
    counts = np.bincount(slot, minlength=K).astype(np.float64)
    if counts.sum() == 0:
        return {"slot_entropy": 0.0, "off_context_rate": 1.0, "slot_hist": counts.tolist()}
    pp = counts / counts.sum()
    nz = pp[pp > 0]
    return {
        "slot_entropy": float(-(nz * np.log(nz)).sum() / math.log(K)),
        "off_context_rate": float(1 - in_ctx.float().mean().item()),
        "slot_hist": (counts / counts.sum()).tolist(),
    }
