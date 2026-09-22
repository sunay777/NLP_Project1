"""Interpretability probes for the induction circuit (Option A).

Two scalar diagnostics per (layer, head), both in [0, 1]:

  prev_token_score  - at each label position t, how much attention mass lands on
                      t-1 (the symbol that precedes the label). The previous-token
                      head that feeds the induction circuit should score ~1 here.

  induction_score   - at the query position, how much attention mass lands on
                      match_label_pos (the label that followed the query symbol's
                      earlier occurrence). The induction head should score ~1 here.
                      This is our 'copy fidelity': track it PER GENERATION to see
                      whether/where the circuit degrades under collapse.

pick_heads() auto-identifies the previous-token head (best in layer 0) and the
induction head (best in the final layer) so downstream logging is mode-agnostic.
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
        # previous-token score: mean over label positions of att[:, h, t, t-1]
        ps = []
        for h in range(H):
            vals = [att[:, h, t, t - 1] for t in label_positions]
            ps.append(torch.stack(vals, 0).mean().item())
        prev.append(ps)
        # induction score: att[:, h, query_pos, match_label_pos]
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
    # previous-token head: best in layer 0
    pt_head = max(range(len(prev[0])), key=lambda h: prev[0][h])
    # induction head: best in the final layer
    last = len(induct) - 1
    ind_head = max(range(len(induct[last])), key=lambda h: induct[last][h])
    return {
        "prev_token_head": (0, pt_head, prev[0][pt_head]),
        "induction_head": (last, ind_head, induct[last][ind_head]),
    }


@torch.no_grad()
def induction_strength(model, batch, cfg):
    """Single summary number: the strength of the best induction head in the
    final layer. Convenient to log every generation for Option A."""
    s = head_scores(model, batch, cfg)
    return pick_heads(s)["induction_head"][2]
