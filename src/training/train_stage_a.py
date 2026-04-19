"""Stage A — article-level representation training (v2).

Changes vs v1 (see audit §§1.1, 1.2, 1.3):
  * Single `DSFSArticleEncoder` owns the MuRIL+LoRA backbone AND every head
    (proj / trust / cleavage / hostility). No more function-attribute
    `_proj_to_hidden` hack; the whole thing saves/loads as one state_dict.
  * `ClusterBatchSampler` guarantees positive pairs in every contrastive
    batch — the default RandomSampler produced ~0 positives at scale.
  * `ClusterCentroidBank` holds EMA cluster centroids across batches so the
    consensus-deviation loss has a stable reference (variance reduction).
  * `source_diversity_regularizer` keeps τ identifiable.
  * bf16 autocast (or fp32 fallback) instead of fp16 — avoids silent NaNs
    from logsumexp overflow in the contrastive loss.
  * Cluster-level train/val split (no positive-pair leakage) + early stopping
    on val loss.
  * Checkpointing every `save_every_steps` (default 500) with top-K retention.

Data expected (single parquet):
  article_id, source_domain, text, event_cluster_id, avg_tone
Plus a dict `atoms[article_id] -> FactualAtoms` for Jaccard.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import pickle
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from ..data.factual_signals import FactualAtoms, jaccard
from ..models.article_encoder import DSFSArticleEncoder, EncoderConfig
from ..models.cleavage_classifier import CLEAVAGES, bce_with_smoothing, weak_label
from ..models.hostility_encoder import hostility_loss, weak_hostility_targets
from ..models.trust_learner import (
    ClusterCentroidBank,
    agreement_contrastive_loss,
    consensus_deviation_loss,
    source_diversity_regularizer,
    source_trust_from_logits,
)
from .samplers import ClusterBatchSampler, positives_per_batch

log = logging.getLogger(__name__)


# ------------------- Dataset / collate ------------------- #

class ArticleDataset(Dataset):
    def __init__(self, df: pd.DataFrame, atoms: dict[str, FactualAtoms], tokenizer,
                 max_length: int = 256):
        self.df = df.reset_index(drop=True)
        self.atoms = atoms
        self.tok = tokenizer
        self.max_length = max_length
        self.cleavage_targets = np.stack([weak_label(t or "") for t in self.df["text"].tolist()])
        self.hostility_targets = weak_hostility_targets(
            self.df["avg_tone"].astype(float).to_numpy()
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> dict:
        row = self.df.iloc[i]
        enc = self.tok(row["text"] or "", truncation=True, padding="max_length",
                       max_length=self.max_length, return_tensors="pt")
        cv = self.cleavage_targets[i]
        k_idx = int(np.argmax(cv)) if cv.sum() > 0 else 0
        cluster = row["event_cluster_id"]
        cluster = int(cluster) if pd.notna(cluster) else -1
        return {
            "idx": i,
            "article_id": row["article_id"],
            "source": row["source_domain"],
            "cluster_id": cluster,
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "cleavage_y": torch.from_numpy(cv).float(),
            "cleavage_idx": torch.tensor(k_idx, dtype=torch.long),
            "hostility_y": torch.tensor(self.hostility_targets[i], dtype=torch.float),
        }


def collate(batch: list[dict], atoms: dict[str, FactualAtoms]) -> dict:
    ids = [b["article_id"] for b in batch]
    srcs = [b["source"] for b in batch]
    clus = torch.tensor([b["cluster_id"] for b in batch], dtype=torch.long)
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attn = torch.stack([b["attention_mask"] for b in batch])
    cl_y = torch.stack([b["cleavage_y"] for b in batch])
    cl_i = torch.stack([b["cleavage_idx"] for b in batch])
    h_y = torch.stack([b["hostility_y"] for b in batch])
    B = len(batch)
    J = torch.zeros(B, B, dtype=torch.float)
    for i in range(B):
        ai = atoms.get(ids[i])
        for j in range(i + 1, B):
            aj = atoms.get(ids[j])
            v = jaccard(ai, aj) if (ai and aj) else 0.0
            J[i, j] = J[j, i] = v
    return {
        "article_ids": ids, "sources": srcs, "cluster_id": clus,
        "input_ids": input_ids, "attention_mask": attn,
        "cleavage_y": cl_y, "cleavage_idx": cl_i, "hostility_y": h_y,
        "jaccard": J,
    }


# ------------------- Split helper ------------------- #

def _cluster_val_split(cluster_ids: list[int], val_frac: float, seed: int
                       ) -> tuple[set[int], set[int]]:
    """Return (train_clusters, val_clusters) — disjoint sets of cluster_ids.
    Singleton rows (cluster_id=-1) go to train by default."""
    rng = np.random.default_rng(seed)
    unique = sorted({int(c) for c in cluster_ids if int(c) >= 0})
    rng.shuffle(unique)
    n_val = max(1, int(round(len(unique) * val_frac)))
    val = set(unique[:n_val])
    train = set(unique[n_val:])
    return train, val


# ------------------- Autocast helper ------------------- #

def _autocast_dtype(device: str) -> torch.dtype | None:
    """bf16 if available on CUDA (A100+, H100, most recent consumer GPUs),
    else fp32. Never fp16 for this loop — audit §1.2."""
    if device != "cuda":
        return None
    try:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
    except Exception:
        pass
    return None  # fall back to fp32


# ------------------- Checkpointing ------------------- #

def _save_ckpt(path: Path, enc: DSFSArticleEncoder, optim: torch.optim.Optimizer,
               step: int, best_val: float, centroids: dict[int, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "encoder": enc.state_dict(),
        "optimizer": optim.state_dict(),
        "step": step,
        "best_val": best_val,
        "centroids": {int(k): v.detach().cpu() for k, v in centroids.items()},
    }, path)


def _load_ckpt(path: Path, enc: DSFSArticleEncoder, optim: torch.optim.Optimizer,
               bank: ClusterCentroidBank, map_location: str) -> tuple[int, float]:
    ck = torch.load(path, map_location=map_location)
    enc.load_state_dict(ck["encoder"])
    optim.load_state_dict(ck["optimizer"])
    for k, v in ck.get("centroids", {}).items():
        bank.centroids[int(k)] = v.to(map_location)
    return int(ck.get("step", 0)), float(ck.get("best_val", math.inf))


def _prune_top_k(ckpt_dir: Path, top_k: int) -> None:
    files = sorted(ckpt_dir.glob("step*.pt"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    for f in files[top_k:]:
        try:
            f.unlink()
        except OSError:
            pass


# ------------------- Main training loop ------------------- #

def train_stage_a(articles_parquet: str | Path,
                  atoms_pickle: str | Path,
                  cfg_path: str | Path,
                  out_dir: str | Path,
                  epochs: int | None = None,
                  resume: str | Path | None = None,
                  save_every_steps: int = 500,
                  top_k_ckpts: int = 3,
                  val_frac: float = 0.1,
                  patience_epochs: int = 2,
                  seed: int = 7) -> None:
    cfg = yaml.safe_load(open(cfg_path))
    tcfg = cfg["trust_learning"]
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"

    df = pd.read_parquet(articles_parquet)
    required = {"article_id", "source_domain", "text", "event_cluster_id", "avg_tone"}
    missing = required - set(df.columns)
    assert not missing, f"missing columns: {missing}"

    with open(atoms_pickle, "rb") as f:
        atoms: dict[str, FactualAtoms] = pickle.load(f)

    # Model — single unified encoder.
    enc = DSFSArticleEncoder(EncoderConfig(
        model_name=cfg["model"]["backbone"],
        lora_rank=tcfg["lora_rank"],
        lora_alpha=tcfg["lora_alpha"],
        lora_dropout=tcfg["lora_dropout"],
        num_cleavages=len(CLEAVAGES),
    ))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc.to(device)
    ac_dtype = _autocast_dtype(device)

    optim = torch.optim.AdamW(enc.parameters(), lr=tcfg["learning_rate"],
                              weight_decay=1e-4)

    bank = ClusterCentroidBank(dim=enc.cfg.proj_dim, momentum=0.95, device=device)

    # Cluster-level train/val split — no positive-pair leakage.
    train_cluster_set, val_cluster_set = _cluster_val_split(
        df["event_cluster_id"].fillna(-1).astype(int).tolist(), val_frac, seed,
    )
    is_val = df["event_cluster_id"].fillna(-1).astype(int).isin(val_cluster_set).to_numpy()
    train_df = df.loc[~is_val].reset_index(drop=True)
    val_df = df.loc[is_val].reset_index(drop=True)
    log.info("train=%d articles, val=%d articles (%d clusters val / %d total)",
             len(train_df), len(val_df), len(val_cluster_set),
             len(train_cluster_set) + len(val_cluster_set))

    ds_train = ArticleDataset(train_df, atoms, enc.tokenizer,
                              max_length=cfg["model"]["max_length"])
    ds_val = ArticleDataset(val_df, atoms, enc.tokenizer,
                            max_length=cfg["model"]["max_length"])

    # Cluster-aware sampler — picks m clusters x k articles per batch.
    m = int(tcfg.get("sampler_clusters_per_batch", 8))
    k = int(tcfg.get("sampler_articles_per_cluster", 4))
    n_batches = max(1, len(train_df) // (m * k))
    sampler = ClusterBatchSampler(
        cluster_ids=train_df["event_cluster_id"].fillna(-1).astype(int).tolist(),
        m_per_batch=m, k_per_cluster=k, n_batches=n_batches, seed=seed,
    )
    loader_train = DataLoader(ds_train, batch_sampler=sampler,
                              collate_fn=lambda b: collate(b, atoms), num_workers=2)
    loader_val = DataLoader(ds_val, batch_size=sampler.batch_size, shuffle=False,
                            collate_fn=lambda b: collate(b, atoms), num_workers=1)

    epochs = epochs or tcfg["epochs"]
    consensus_w = float(tcfg["consensus_weight"])
    cleav_w = float(tcfg.get("cleavage_weight", 1.0))
    host_w = float(tcfg.get("hostility_weight", 0.5))
    srcdiv_w = float(tcfg.get("source_diversity_weight", 0.1))

    # Resume if requested.
    step = 0
    best_val = math.inf
    bad_epochs = 0
    if resume is not None and Path(resume).exists():
        step, best_val = _load_ckpt(Path(resume), enc, optim, bank, device)
        log.info("resumed from %s at step=%d best_val=%.4f", resume, step, best_val)

    history: list[dict] = []
    all_s_logits: list[float] = []
    all_sources: list[str] = []

    for ep in range(epochs):
        enc.train()
        ep_losses = {"agree": 0.0, "dev": 0.0, "cleav": 0.0, "host": 0.0,
                     "srcdiv": 0.0, "n": 0, "pos_pairs": 0.0}
        pbar = tqdm(loader_train, desc=f"epoch {ep + 1}/{epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            clus = batch["cluster_id"].to(device)
            J = batch["jaccard"].to(device)
            cl_y = batch["cleavage_y"].to(device)
            h_y = batch["hostility_y"].to(device)
            cl_i = batch["cleavage_idx"].to(device)
            sources = batch["sources"]

            optim.zero_grad()

            # Encoder forward -- single pass, all heads share the backbone CLS.
            use_ac = ac_dtype is not None
            with torch.autocast(device_type="cuda", dtype=ac_dtype, enabled=use_ac):
                out = enc(input_ids, attn)
                z = out["z"]                  # [B, D] unit-norm
                s = out["trust_logit"]        # [B]
                cl_logits = out["cleavage"]   # [B, K]
                h_logits_all = out["hostility"]  # [B, K]
                # Select the dominant-cleavage column for per-article hostility.
                h_logits = h_logits_all.gather(1, cl_i.view(-1, 1)).squeeze(-1)

            # Keep the losses in fp32 either way (contrastive logsumexp is the
            # classic fp16 overflow spot — we're bf16/fp32 here, but the extra
            # .float() is cheap and keeps things deterministic.)
            z_f = z.float()
            s_f = s.float()
            L_agree = agreement_contrastive_loss(
                z_f, clus, J, tau=tcfg["contrastive_tau"],
                jaccard_weight=tcfg["jaccard_weight"],
            )
            L_dev = consensus_deviation_loss(z_f, s_f, clus, bank)
            L_cleav = bce_with_smoothing(cl_logits.float(), cl_y, smoothing=0.05)
            L_host = hostility_loss(h_logits.float(), h_y)
            L_srcdiv = source_diversity_regularizer(s_f, sources)

            L = (L_agree
                 + consensus_w * L_dev
                 + cleav_w * L_cleav
                 + host_w * L_host
                 + srcdiv_w * L_srcdiv)

            L.backward()
            torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
            optim.step()

            # IMPORTANT: update centroids AFTER the loss, not before.
            with torch.no_grad():
                gate = torch.sigmoid(s_f).clamp(0.1, 0.9)
                bank.update(z_f.detach(), gate.detach(), clus)

            pos = positives_per_batch(clus.cpu().tolist())
            ep_losses["agree"] += float(L_agree); ep_losses["dev"] += float(L_dev)
            ep_losses["cleav"] += float(L_cleav); ep_losses["host"] += float(L_host)
            ep_losses["srcdiv"] += float(L_srcdiv)
            ep_losses["n"] += 1
            ep_losses["pos_pairs"] += float(pos)
            step += 1
            pbar.set_postfix({
                "agree": f"{L_agree.item():.3f}", "dev": f"{L_dev.item():.3f}",
                "cleav": f"{L_cleav.item():.3f}", "host": f"{L_host.item():.3f}",
                "pos": pos,
            })

            if step % save_every_steps == 0:
                _save_ckpt(ckpt_dir / f"step{step:07d}.pt", enc, optim, step,
                           best_val, bank.centroids)
                _prune_top_k(ckpt_dir, top_k_ckpts)

            if ep == epochs - 1:
                all_s_logits.extend(s_f.detach().cpu().tolist())
                all_sources.extend(sources)

        # ---- Validation ----
        enc.eval()
        val_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for batch in loader_val:
                input_ids = batch["input_ids"].to(device)
                attn = batch["attention_mask"].to(device)
                clus = batch["cluster_id"].to(device)
                J = batch["jaccard"].to(device)
                out = enc(input_ids, attn)
                L_a = agreement_contrastive_loss(
                    out["z"].float(), clus, J, tau=tcfg["contrastive_tau"],
                    jaccard_weight=tcfg["jaccard_weight"],
                )
                if torch.isfinite(L_a):
                    val_sum += float(L_a)
                    val_n += 1
        val_loss = (val_sum / val_n) if val_n > 0 else float("nan")

        mean_losses = {k: (v / max(ep_losses["n"], 1) if k != "n" else v)
                       for k, v in ep_losses.items()}
        mean_losses["val"] = val_loss
        history.append(mean_losses)
        log.info("ep %d  agree=%.4f dev=%.4f cleav=%.4f host=%.4f  val=%.4f pos/b=%.1f",
                 ep, mean_losses["agree"], mean_losses["dev"], mean_losses["cleav"],
                 mean_losses["host"], val_loss, mean_losses["pos_pairs"])

        # Early stopping + best-model save.
        if not math.isnan(val_loss) and val_loss < best_val - 1e-4:
            best_val = val_loss
            bad_epochs = 0
            _save_ckpt(out_dir / "best.pt", enc, optim, step, best_val, bank.centroids)
        else:
            bad_epochs += 1
            if bad_epochs >= patience_epochs:
                log.info("early stop: val hasn't improved for %d epochs", patience_epochs)
                break

    # Persist final artifacts.
    torch.save(enc.state_dict(), out_dir / "dsfs_encoder.pt")

    tau_map = source_trust_from_logits(all_sources, torch.tensor(all_s_logits)) \
        if all_s_logits else {}
    if tau_map:
        pd.DataFrame([{"source_domain": k, "tau": v} for k, v in tau_map.items()]) \
            .sort_values("tau", ascending=False) \
            .to_parquet(out_dir / "source_trust.parquet", index=False)

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    log.info("Stage A complete. Sources scored: %d. Best val=%.4f", len(tau_map), best_val)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--articles", required=True)
    p.add_argument("--atoms", required=True)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--out", default="./artifacts/stage_a")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--save-every-steps", type=int, default=500)
    p.add_argument("--top-k-ckpts", type=int, default=3)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=2)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)
    train_stage_a(args.articles, args.atoms, args.config, args.out,
                  epochs=args.epochs, resume=args.resume,
                  save_every_steps=args.save_every_steps,
                  top_k_ckpts=args.top_k_ckpts, val_frac=args.val_frac,
                  patience_epochs=args.patience)


if __name__ == "__main__":
    main()
