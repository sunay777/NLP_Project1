"""Data generator for the in-context-learning symbol task (Singh et al., 2024).

Study -> query formulation (the version that forms a clean induction head):
  * pick K distinct symbols, give each a fixed random label (resampled every
    sequence, so the map must be read from context, not memorised in weights);
  * STUDY block: the K (symbol, label) pairs in a random order;
  * QUERY block: the SAME K symbols in a DIFFERENT random order, each followed by
    its label. The model predicts each query label by finding that symbol in the
    study block and copying the following label -> an induction head.
Because the query order is independent of the study order, the match offset
varies from query to query, so a positional shortcut cannot solve it: only a
content-based induction head can. Every query is a clean, supervised target.

Layout (K = n_pairs):
  base:     [study 2K] [query 2K]                 len 4K
            supervise every query SYMBOL position -> the following label
  extended: [study 2K] [query 2K] sq2 lq2         len 4K+2
            additionally predict sq2 (free next symbol) and lq2 (its label);
            sq2 is the generative degree of freedom that drives collapse.

Returned batch dict:
  idx (B,L), targets (B,L, -100 where unsupervised),
  query_pos (B,)  first query-symbol position (== 2K),
  query_tgt (B,)  its label,
  match_label_pos (B,)  the study label position it must copy (unique match),
  gen_symbol_pos (int)  position predicting the generative symbol (extended) or -1
"""
import numpy as np
import torch

IGNORE = -100


def make_batch(cfg, batch_size, rng, device="cpu"):
    K, B = cfg.n_pairs, batch_size
    L = cfg.max_len
    off = cfg.label_offset
    rows = np.arange(B)

    classes = np.argsort(rng.random((B, cfg.n_symbols)), axis=1)[:, :K]     # (B,K) distinct symbols
    class_label = rng.integers(0, cfg.n_labels, size=(B, K)) + off          # (B,K)

    study_perm = np.argsort(rng.random((B, K)), axis=1)                     # class order in study
    query_perm = np.argsort(rng.random((B, K)), axis=1)                     # class order in query

    study_sym = np.take_along_axis(classes, study_perm, axis=1)
    study_lab = np.take_along_axis(class_label, study_perm, axis=1)
    query_sym = np.take_along_axis(classes, query_perm, axis=1)
    query_lab = np.take_along_axis(class_label, query_perm, axis=1)

    seq = np.zeros((B, L), dtype=np.int64)
    seq[:, 0:2 * K:2] = study_sym
    seq[:, 1:2 * K:2] = study_lab
    seq[:, 2 * K:4 * K:2] = query_sym
    seq[:, 2 * K + 1:4 * K:2] = query_lab

    targets = np.full((B, L), IGNORE, dtype=np.int64)
    # predict each query label from its query symbol (induction)
    q_sym_pos = np.arange(2 * K, 4 * K, 2)                                  # 2K,2K+2,...,4K-2
    targets[:, q_sym_pos] = query_lab

    # clean probe target: first query (j=0). Its study position gives the match.
    q0_class = query_perm[:, 0]                                             # class index queried first
    study_pos = np.argmax(study_perm == q0_class[:, None], axis=1)          # its slot in study
    match_label_pos = 2 * study_pos + 1
    query_tgt = query_lab[:, 0]

    gen_symbol_pos = -1
    if cfg.mode == "extended":
        nxt = rng.integers(0, K, size=B)                                   # free next symbol
        sq2 = classes[rows, nxt]
        lq2 = class_label[rows, nxt]
        seq[:, 4 * K] = sq2
        seq[:, 4 * K + 1] = lq2
        targets[:, 4 * K - 1] = sq2                                        # predict next symbol
        targets[:, 4 * K] = lq2                                            # predict its label
        gen_symbol_pos = 4 * K - 1

    return {
        "idx": torch.from_numpy(seq).to(device),
        "targets": torch.from_numpy(targets).to(device),
        "query_pos": torch.full((B,), 2 * K, dtype=torch.long, device=device),
        "query_tgt": torch.from_numpy(query_tgt).to(device),
        "match_label_pos": torch.from_numpy(match_label_pos).to(device),
        "gen_symbol_pos": gen_symbol_pos,
    }


class TensorDataset:
    """A fixed pool of sequences for one generation of the collapse loop."""

    def __init__(self, batch):
        self.data = {k: v for k, v in batch.items() if torch.is_tensor(v)}
        self.meta = {k: v for k, v in batch.items() if not torch.is_tensor(v)}
        self.n = self.data["idx"].shape[0]

    def sample(self, batch_size, rng):
        sel = torch.from_numpy(rng.integers(0, self.n, size=batch_size)).to(self.data["idx"].device)
        out = {k: v[sel] for k, v in self.data.items()}
        out.update(self.meta)
        return out

    @staticmethod
    def concat(a, b):
        merged = {k: torch.cat([a.data[k], b.data[k]], dim=0) for k in a.data}
        merged.update(a.meta)
        return TensorDataset(merged)
