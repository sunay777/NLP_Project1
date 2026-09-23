"""Central configuration for the induction-head / model-collapse experiments.

Token scheme (single shared vocabulary):
    symbols occupy ids [0, n_symbols)
    labels  occupy ids [n_symbols, n_symbols + n_labels)
Keeping symbols and labels in disjoint id ranges makes attention maps and
embedding plots unambiguous to read during interpretation.

Task shape (see src/data.py): each sequence draws K = n_pairs distinct symbols,
each with a random label; a STUDY block lists the K pairs in one random order and
a QUERY block repeats the same symbols in an independent random order, so each
query label must be copied from the study block by a content-based induction head.
"""
from dataclasses import dataclass


@dataclass
class Config:
    # ---- task / data ----
    # Defaults chosen so the induction circuit forms reliably and fast on CPU.
    # Difficulty rises steeply with n_pairs and n_symbols: scale these up (and give
    # more steps) once you have a GPU/MPS. See README for a scaling note.
    n_symbols: int = 16      # pool of symbol ids (diversity is measured over this)
    n_labels: int = 6        # pool of label ids
    n_pairs: int = 4         # K: distinct symbols; K study pairs + K queries
    mode: str = "base"       # "base" (generate 1 token) or "extended" (generate 3)

    # ---- model ----
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2        # L0 hosts the previous-token head, L1 the induction head
    use_mlp: bool = False    # attention-only by default -> cleaner circuits
    d_mlp: int = 256
    use_rope: bool = True    # rotary positions: lets the previous-token head form
    tie_embeddings: bool = True  # tie embed<->unembed: aids the copy operation

    # ---- training ----
    batch_size: int = 256
    steps: int = 4000        # optimiser steps per generation
    lr: float = 1e-3
    weight_decay: float = 1e-2
    warmup: int = 200
    lr_decay: bool = True    # cosine decay after warmup; a constant LR destabilises
                             # the model out of the induction solution once found
    seed: int = 0
    device: str = "auto"

    # ---- eval ----
    eval_batch: int = 1024
    eval_every: int = 250

    @property
    def label_offset(self) -> int:
        return self.n_symbols

    @property
    def vocab_size(self) -> int:
        return self.n_symbols + self.n_labels

    @property
    def max_len(self) -> int:
        # base:     [study 2K][query 2K]           -> 4K
        # extended: [study 2K][query 2K] sq2 lq2   -> 4K + 2
        return 4 * self.n_pairs + (0 if self.mode == "base" else 2)


def resolve_device(name: str):
    import torch
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
