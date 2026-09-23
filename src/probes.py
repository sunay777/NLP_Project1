"""Interpretability probes for the induction circuit (Option A).

Scalar diagnostics per (layer, head), all in [0, 1]:

  prev_token_score  - at each study-label position t, attention mass on t-1 (the
                      symbol that precedes the label). A previous-token head
                      feeding the canonical induction circuit scores ~1 here.

  induction_score   - LEGACY probe: at the FIRST query position, attention mass on
                      match_label_pos (the label after the query symbol's earlier
                      occurrence). Kept for continuity with earlier results.

Robust circuit probe (circuit_probe), computed over ALL K query positions:

  canonical   - mass on the label after the matching study symbol (match_sym+1).
  shifted     - mass on the study symbol AFTER the matched pair (match_sym+2).
                Some seeds solve the task this way: layer 0 writes both the
                pair's symbol (t-2 head) and its label (t-1 head) into the next
                symbol's residual, and the L1 head matches on the former and
                reads out the latter. Verified by ablation (seed 3, extended).
  copy        - canonical + shifted: mass on a position that carries the answer.

  Summary scalars:
    induction_best   max over final-layer heads of `copy`      (log per generation)
    induction_mass   mean over final-layer heads of `copy`     (redundancy-aware)
    circuit_type     'canonical' | 'shifted' (which pattern the best head uses)
    ablation_acc     query-label accuracy with the induction set zero-ablated
    causal_drop      unablated accuracy - ablation_acc           (causal evidence)

Why: base-mode models spread induction over 3 heads (single-head ablation costs
<4% accuracy), and some extended seeds use the shifted pattern, so a single-head
canonical score can report 'no circuit' for a model that is solving the task.
"""
import torch


@torch.no_grad()
def head_scores(model, batch, cfg):
    model.eval()
    logits, attns = model(batch["idx"], return_attn=True)     # attns[layer]: (B,H,T,T)
    K = cfg.n_pairs
    B = batch["idx"].shape[0]
    rows = torch.arange(B, device=batch["idx"].device)
    label_positions = list(range(1, 2 * K, 2))                # 1,3,...,2K-1

    prev, induct = [], []
    for att in attns:
        H = att.shape[1]
        ps = []
        for h in range(H):
            vals = [att[:, h, t, t - 1] for t in label_positions]
            ps.append(torch.stack(vals, 0).mean().item())
        prev.append(ps)
        qp = batch["query_pos"]
        mp = batch["match_label_pos"]
        is_ = []
        for h in range(H):
            is_.append(att[rows, h, qp, mp].mean().item())
        induct.append(is_)
    return {"prev_token": prev, "induction": induct}


def pick_heads(scores):
    prev = scores["prev_token"]
    induct = scores["induction"]
    pt_head = max(range(len(prev[0])), key=lambda h: prev[0][h])
    last = len(induct) - 1
    ind_head = max(range(len(induct[last])), key=lambda h: induct[last][h])
    return {
        "prev_token_head": (0, pt_head, prev[0][pt_head]),
        "induction_head": (last, ind_head, induct[last][ind_head]),
    }


@torch.no_grad()
def induction_strength(model, batch, cfg):
    """LEGACY single-head, first-query canonical score (see circuit_probe)."""
    s = head_scores(model, batch, cfg)
    return pick_heads(s)["induction_head"][2]


# --------------------------------------------------------------------------- #
# Robust probe
# --------------------------------------------------------------------------- #
def match_positions(idx, K):
    """Study position of each query symbol's match -> (sym_pos, label_pos), (B,K)."""
    study_sym = idx[:, 0:2 * K:2]
    q_sym = idx[:, 2 * K:4 * K:2]
    slot = (q_sym[:, :, None] == study_sym[:, None, :]).float().argmax(-1)
    return 2 * slot, 2 * slot + 1


def ablate_heads(model, cfg, heads):
    """Zero-ablate (layer, head) outputs via a pre-hook on attn.proj.
    Returns hook handles; call .remove() on each to restore."""
    dh = cfg.d_model // cfg.n_heads
    hooks = []
    for l in sorted(set(l for l, _ in heads)):
        hs = [h for ll, h in heads if ll == l]

        def pre(mod, inp, hs=hs):
            x = inp[0].clone()
            for h in hs:
                x[..., h * dh:(h + 1) * dh] = 0
            return (x,)
        hooks.append(model.blocks[l].attn.proj.register_forward_pre_hook(pre))
    return hooks


@torch.no_grad()
def query_label_accuracy(model, batch, cfg):
    """Accuracy over ALL K query-label predictions against the batch's targets."""
    K = cfg.n_pairs
    ps = list(range(2 * K, 4 * K, 2))
    pred = model(batch["idx"]).argmax(-1)
    return (pred[:, ps] == batch["targets"][:, ps]).float().mean().item()


@torch.no_grad()
def circuit_probe(model, batch, cfg, set_threshold=0.35):
    """Robust, redundancy- and variant-aware induction probe (see module doc).
    `batch` should be REAL data (targets = ground truth)."""
    model.eval()
    K = cfg.n_pairs
    idx = batch["idx"]
    B = idx.shape[0]
    L = idx.shape[1]
    rows = torch.arange(B, device=idx.device)[:, None]
    _, attns = model(idx, return_attn=True)
    sym_pos, lab_pos = match_positions(idx, K)
    qpos = torch.arange(2 * K, 4 * K, 2, device=idx.device)[None, :].expand(B, K)
    shift_pos = (sym_pos + 2).clamp(max=L - 1)
    # when the match is the LAST study pair, match_sym+2 is the first query symbol
    # (2K); only count the shifted pattern where that position is strictly earlier.
    shift_valid = (shift_pos < qpos).float()

    last = len(attns) - 1
    per_head = []
    for h in range(attns[last].shape[1]):
        a = attns[last][:, h]
        can_bj = a[rows, qpos, lab_pos]                              # (B,K)
        sh_bj = a[rows, qpos, shift_pos] * shift_valid               # (B,K)
        per_head.append({"head": h, "canonical": can_bj.mean().item(),
                         "shifted": sh_bj.mean().item(),
                         "copy": (can_bj + sh_bj).mean().item()})

    best = max(per_head, key=lambda r: r["copy"])
    induction_set = [(last, r["head"]) for r in per_head if r["copy"] > set_threshold]
    acc = query_label_accuracy(model, batch, cfg)
    hk = ablate_heads(model, cfg, induction_set) if induction_set else []
    abl = query_label_accuracy(model, batch, cfg)
    for x in hk:
        x.remove()

    # layer-0 support: canonical needs t-1 at labels; shifted needs t-2 at symbols
    a0 = attns[0]
    lab_t = torch.arange(1, 2 * K, 2, device=idx.device)
    sym_t = torch.arange(2, 2 * K, 2, device=idx.device)
    prev_lab = a0[:, :, lab_t, lab_t - 1].mean(dim=(0, 2))          # (H,)
    prev2_sym = a0[:, :, sym_t, sym_t - 2].mean(dim=(0, 2))         # (H,)

    return {
        "induction_legacy": induction_strength(model, batch, cfg),
        "induction_best": best["copy"],
        "induction_mass": sum(r["copy"] for r in per_head) / len(per_head),
        "circuit_type": "canonical" if best["canonical"] >= best["shifted"] else "shifted",
        "induction_set": [h for _, h in induction_set],
        "query_acc_all": acc,
        "ablation_acc": abl,
        "causal_drop": acc - abl,
        "prev_token_best": prev_lab.max().item(),
        "prev2_token_best": prev2_sym.max().item(),
        "per_head": per_head,
    }
