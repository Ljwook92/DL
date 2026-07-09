#!/usr/bin/env python3
"""Train a compact U-Net for next-day HMS smoke impact patches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader, Dataset


DEFAULT_DATASET_DIR = Path(
    "/Volumes/Extreme SSD/4. smoke_coverage/data/analysis/"
    "temporal_unet_10km/rank1_episode_gap14"
)


class PatchDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, valid: np.ndarray, indices: np.ndarray):
        self.x = x
        self.y = y
        self.valid = valid
        self.indices = indices.astype(int)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = self.indices[item]
        return (
            torch.from_numpy(self.x[idx]),
            torch.from_numpy(self.y[idx]),
            torch.from_numpy(self.valid[idx]),
        )


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SmallUNet(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 16):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base_channels * 2, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)
        self.out = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out(d1)


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def normalize_inputs(
    x: np.ndarray,
    valid: np.ndarray,
    metadata: pd.DataFrame,
    train_indices: np.ndarray,
    channel_names: list[str],
) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    out = x.copy()
    train_valid = valid[train_indices, 0].astype(bool)
    stats: dict[str, dict[str, float]] = {}
    for channel_idx, channel in enumerate(channel_names):
        if channel == "valid_cell_mask":
            stats[channel] = {"mean": 0.0, "std": 1.0}
            continue
        values = out[train_indices, channel_idx][train_valid]
        values = values[np.isfinite(values)]
        if len(values) == 0:
            mean = 0.0
            std = 1.0
        else:
            mean = float(values.mean())
            std = float(values.std())
            if not np.isfinite(std) or std < 1e-6:
                std = 1.0
        out[:, channel_idx] = (out[:, channel_idx] - mean) / std
        out[:, channel_idx] *= valid[:, 0]
        stats[channel] = {"mean": mean, "std": std}
    if "valid_cell_mask" in channel_names:
        out[:, channel_names.index("valid_cell_mask")] = valid[:, 0]
    return out.astype(np.float32), stats


def split_indices(metadata: pd.DataFrame, train_periods: list[str], test_periods: list[str]) -> tuple[np.ndarray, np.ndarray]:
    period = metadata["period"].astype(str)
    train_idx = metadata.index[period.isin(train_periods)].to_numpy(dtype=int)
    test_idx = metadata.index[period.isin(test_periods)].to_numpy(dtype=int)
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise RuntimeError(f"Empty train/test split: train={len(train_idx)} test={len(test_idx)}")
    return train_idx, test_idx


def masked_bce_loss(
    criterion: nn.Module,
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    loss = criterion(logits, target)
    return (loss * valid).sum() / valid.sum().clamp_min(1.0)


def masked_tversky_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    alpha: float,
    beta: float,
    gamma: float = 1.0,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Tversky/Focal-Tversky loss over valid cells.

    alpha weights false positives and beta weights false negatives. For smoke
    coverage, alpha > beta is useful when the model paints the plume too wide.
    """
    prob = torch.sigmoid(logits)
    prob = prob * valid
    target = target * valid
    true_pos = (prob * target).sum()
    false_pos = (prob * (1.0 - target) * valid).sum()
    false_neg = ((1.0 - prob) * target * valid).sum()
    score = (true_pos + smooth) / (true_pos + alpha * false_pos + beta * false_neg + smooth)
    return torch.pow(1.0 - score, gamma)


def masked_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    smooth: float = 1.0,
) -> torch.Tensor:
    prob = torch.sigmoid(logits) * valid
    target = target * valid
    intersection = (prob * target).sum()
    denominator = prob.sum() + target.sum()
    score = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - score


def compute_masked_loss(
    loss_name: str,
    bce_criterion: nn.Module,
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    bce_weight: float,
    dice_weight: float,
    tversky_alpha: float,
    tversky_beta: float,
    focal_gamma: float,
) -> torch.Tensor:
    bce = masked_bce_loss(bce_criterion, logits, target, valid)
    if loss_name == "bce":
        return bce
    if loss_name == "dice":
        return masked_dice_loss(logits, target, valid)
    if loss_name == "tversky":
        return masked_tversky_loss(logits, target, valid, tversky_alpha, tversky_beta)
    if loss_name == "focal_tversky":
        return masked_tversky_loss(logits, target, valid, tversky_alpha, tversky_beta, focal_gamma)
    if loss_name == "bce_dice":
        return bce_weight * bce + dice_weight * masked_dice_loss(logits, target, valid)
    if loss_name == "bce_tversky":
        return bce_weight * bce + dice_weight * masked_tversky_loss(
            logits,
            target,
            valid,
            tversky_alpha,
            tversky_beta,
        )
    raise ValueError(f"Unsupported loss: {loss_name}")


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    probs = []
    targets = []
    valids = []
    for x, y, valid in loader:
        x = x.to(device)
        logits = model(x)
        probs.append(torch.sigmoid(logits).cpu().numpy())
        targets.append(y.numpy())
        valids.append(valid.numpy())
    return np.concatenate(probs), np.concatenate(targets), np.concatenate(valids)


def evaluate_pixels(y: np.ndarray, prob: np.ndarray, valid: np.ndarray) -> dict[str, object]:
    mask = valid[:, 0].astype(bool)
    y_flat = y[:, 0][mask].astype(int)
    p_flat = prob[:, 0][mask].astype(float)
    metrics: dict[str, object] = {
        "valid_pixels": int(len(y_flat)),
        "positive_pixels": int(y_flat.sum()),
        "positive_rate": float(y_flat.mean()) if len(y_flat) else 0.0,
        "roc_auc": float(roc_auc_score(y_flat, p_flat)) if len(np.unique(y_flat)) == 2 else None,
        "average_precision": float(average_precision_score(y_flat, p_flat)) if len(np.unique(y_flat)) == 2 else None,
        "brier": float(brier_score_loss(y_flat, p_flat)) if len(np.unique(y_flat)) >= 1 else None,
        "thresholds": [],
    }
    for threshold in [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50]:
        pred = p_flat >= threshold
        precision, recall, f1, _ = precision_recall_fscore_support(y_flat, pred, average="binary", zero_division=0)
        metrics["thresholds"].append(
            {
                "threshold": threshold,
                "predicted_positive_rate": float(pred.mean()),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )
    metrics["best_f1_threshold"] = max(metrics["thresholds"], key=lambda row: row["f1"])
    return metrics


def evaluate_patch_iou(
    y: np.ndarray,
    prob: np.ndarray,
    valid: np.ndarray,
    metadata: pd.DataFrame,
    indices: np.ndarray,
    thresholds: list[float] | None = None,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Evaluate predicted coverage as a fire-day mask overlap problem."""
    if thresholds is None:
        thresholds = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50]
    rows: list[dict[str, object]] = []
    for local_idx, global_idx in enumerate(indices):
        meta = metadata.iloc[int(global_idx)]
        mask = valid[local_idx, 0].astype(bool)
        observed = (y[local_idx, 0] >= 0.5) & mask
        observed_cells = int(observed.sum())
        prob_values = prob[local_idx, 0]
        soft_intersection = float((prob_values[mask] * observed[mask]).sum())
        soft_union = float((prob_values[mask] + observed[mask].astype(float) - prob_values[mask] * observed[mask]).sum())
        soft_iou = soft_intersection / soft_union if soft_union > 0 else np.nan
        for threshold in thresholds:
            predicted = (prob_values >= threshold) & mask
            intersection = int((predicted & observed).sum())
            union = int((predicted | observed).sum())
            predicted_cells = int(predicted.sum())
            rows.append(
                {
                    "sample_index": int(global_idx),
                    "threshold": float(threshold),
                    "period": meta.get("period"),
                    "source_date": meta.get("source_date"),
                    "target_date": meta.get("target_date"),
                    "cluster_id": meta.get("cluster_id"),
                    "fire_episode_id": meta.get("fire_episode_id"),
                    "fire_name": meta.get("fire_name"),
                    "observed_cells": observed_cells,
                    "predicted_cells": predicted_cells,
                    "intersection_cells": intersection,
                    "union_cells": union,
                    "iou": float(intersection / union) if union else np.nan,
                    "precision": float(intersection / predicted_cells) if predicted_cells else 0.0,
                    "recall": float(intersection / observed_cells) if observed_cells else 0.0,
                    "area_ratio_pred_to_obs": float(predicted_cells / observed_cells) if observed_cells else np.nan,
                    "soft_iou": soft_iou,
                }
            )
    records = pd.DataFrame(rows)
    threshold_summary = []
    for threshold, group in records.groupby("threshold", sort=True):
        inter = group["intersection_cells"].sum()
        union = group["union_cells"].sum()
        threshold_summary.append(
            {
                "threshold": float(threshold),
                "mean_iou": float(group["iou"].mean()),
                "median_iou": float(group["iou"].median()),
                "weighted_iou": float(inter / union) if union else np.nan,
                "mean_precision": float(group["precision"].mean()),
                "mean_recall": float(group["recall"].mean()),
                "median_area_ratio_pred_to_obs": float(group["area_ratio_pred_to_obs"].median()),
                "mean_area_ratio_pred_to_obs": float(group["area_ratio_pred_to_obs"].mean()),
            }
        )
    summary: dict[str, object] = {
        "samples": int(len(indices)),
        "thresholds": threshold_summary,
        "best_mean_iou_threshold": max(threshold_summary, key=lambda row: row["mean_iou"]) if threshold_summary else None,
        "best_weighted_iou_threshold": max(threshold_summary, key=lambda row: row["weighted_iou"]) if threshold_summary else None,
        "soft_iou_mean": float(records.drop_duplicates("sample_index")["soft_iou"].mean()) if not records.empty else None,
        "soft_iou_median": float(records.drop_duplicates("sample_index")["soft_iou"].median()) if not records.empty else None,
    }
    return summary, records


def write_examples(
    prob: np.ndarray,
    y: np.ndarray,
    valid: np.ndarray,
    metadata: pd.DataFrame,
    test_indices: np.ndarray,
    output_dir: Path,
    max_examples: int,
    threshold: float,
) -> list[str]:
    positive_counts = y[:, 0].sum(axis=(1, 2))
    order = np.argsort(-positive_counts)
    paths: list[str] = []
    for rank, local_idx in enumerate(order[:max_examples], start=1):
        global_idx = int(test_indices[local_idx])
        meta = metadata.iloc[global_idx]
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        observed = np.ma.masked_where(valid[local_idx, 0] == 0, y[local_idx, 0])
        predicted = np.ma.masked_where(valid[local_idx, 0] == 0, prob[local_idx, 0])
        axes[0].imshow(observed, cmap="Reds", vmin=0, vmax=1)
        axes[0].scatter([meta["source_col"]], [meta["source_row"]], s=32, c="#111111", marker="*")
        axes[0].set_title("Observed D+1 HMS smoke")
        im = axes[1].imshow(predicted, cmap="magma", vmin=0, vmax=max(0.25, float(np.nanmax(prob[local_idx, 0]))))
        axes[1].scatter([meta["source_col"]], [meta["source_row"]], s=32, c="#33a02c", marker="*")
        axes[1].set_title("Predicted probability")
        axes[2].imshow(np.ma.masked_where(valid[local_idx, 0] == 0, prob[local_idx, 0] >= threshold), cmap="Blues", vmin=0, vmax=1)
        axes[2].imshow(observed, cmap="Reds", alpha=0.45, vmin=0, vmax=1)
        axes[2].scatter([meta["source_col"]], [meta["source_row"]], s=32, c="#111111", marker="*")
        axes[2].set_title(f"p>={threshold:.0%} over observed")
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        title = (
            f"{meta.get('fire_name', '')} | source {meta.get('source_date', '')} -> target {meta.get('target_date', '')} | "
            f"observed cells {int(meta.get('positive_cells', 0))}"
        )
        fig.suptitle(title, fontsize=10)
        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        fig.tight_layout()
        path = output_dir / f"unet_prediction_example_{rank:02d}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(str(path))
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--dataset-npz", type=Path)
    parser.add_argument("--metadata-csv", type=Path)
    parser.add_argument("--channels-json", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--train-periods", nargs="+", default=["202406", "202407", "202408", "202409", "202410"])
    parser.add_argument("--test-periods", nargs="+", default=["202506", "202507", "202508", "202509", "202510"])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--pos-weight-cap", type=float, default=20.0)
    parser.add_argument(
        "--loss",
        choices=["bce", "dice", "tversky", "focal_tversky", "bce_dice", "bce_tversky"],
        default="bce",
    )
    parser.add_argument("--bce-weight", type=float, default=0.3)
    parser.add_argument("--dice-weight", type=float, default=0.7)
    parser.add_argument("--tversky-alpha", type=float, default=0.7)
    parser.add_argument("--tversky-beta", type=float, default=0.3)
    parser.add_argument("--focal-gamma", type=float, default=1.33)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-examples", type=int, default=4)
    parser.add_argument("--example-threshold", type=float, default=0.50)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_npz = args.dataset_npz or args.dataset_dir / "temporal_unet_patch_dataset.npz"
    metadata_csv = args.metadata_csv or args.dataset_dir / "temporal_unet_patch_metadata.csv"
    channels_json = args.channels_json or args.dataset_dir / "temporal_unet_channels.json"
    output_dir = args.output_dir or args.dataset_dir / "unet_model"
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = np.load(dataset_npz)
    x = payload["x"].astype(np.float32)
    y = payload["y"].astype(np.float32)
    valid = payload["valid"].astype(np.float32)
    metadata = pd.read_csv(metadata_csv)
    channels_payload = json.loads(channels_json.read_text(encoding="utf-8"))
    channel_names = channels_payload["channels"]

    train_idx, test_idx = split_indices(metadata, args.train_periods, args.test_periods)
    x, normalization = normalize_inputs(x, valid, metadata, train_idx, channel_names)

    rng = np.random.default_rng(args.random_state)
    np.random.seed(args.random_state)
    torch.manual_seed(args.random_state)
    train_idx = rng.permutation(train_idx)
    device = choose_device(args.device)

    train_ds = PatchDataset(x, y, valid, train_idx)
    test_ds = PatchDataset(x, y, valid, test_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    pos = float((y[train_idx] * valid[train_idx]).sum())
    neg = float(valid[train_idx].sum() - pos)
    pos_weight = min(max(neg / max(pos, 1.0), 1.0), args.pos_weight_cap)
    model = SmallUNet(in_channels=x.shape[1], base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor([pos_weight], device=device).view(1, 1, 1, 1))

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_valid = 0.0
        for xb, yb, vb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            vb = vb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = compute_masked_loss(
                args.loss,
                criterion,
                logits,
                yb,
                vb,
                args.bce_weight,
                args.dice_weight,
                args.tversky_alpha,
                args.tversky_beta,
                args.focal_gamma,
            )
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * float(vb.sum().cpu())
            total_valid += float(vb.sum().cpu())
        epoch_loss = total_loss / max(total_valid, 1.0)
        history.append({"epoch": epoch, "train_loss": epoch_loss})
        print(f"epoch {epoch}/{args.epochs} train_loss={epoch_loss:.5f}", flush=True)

    train_prob, train_y, train_valid = predict(model, DataLoader(train_ds, batch_size=args.batch_size, shuffle=False), device)
    test_prob, test_y, test_valid = predict(model, test_loader, device)
    metrics = {
        "dataset_npz": str(dataset_npz),
        "metadata_csv": str(metadata_csv),
        "train_periods": args.train_periods,
        "test_periods": args.test_periods,
        "device": str(device),
        "samples": int(len(metadata)),
        "train_samples": int(len(train_idx)),
        "test_samples": int(len(test_idx)),
        "channels": channel_names,
        "channel_count": int(x.shape[1]),
        "patch_size": int(x.shape[-1]),
        "pos_weight": float(pos_weight),
        "pos_weight_cap": float(args.pos_weight_cap),
        "loss": args.loss,
        "bce_weight": float(args.bce_weight),
        "dice_weight": float(args.dice_weight),
        "tversky_alpha": float(args.tversky_alpha),
        "tversky_beta": float(args.tversky_beta),
        "focal_gamma": float(args.focal_gamma),
        "history": history,
        "train_eval": evaluate_pixels(train_y, train_prob, train_valid),
        "test_eval": evaluate_pixels(test_y, test_prob, test_valid),
    }
    train_patch_iou, train_patch_records = evaluate_patch_iou(train_y, train_prob, train_valid, metadata, train_idx)
    test_patch_iou, test_patch_records = evaluate_patch_iou(test_y, test_prob, test_valid, metadata, test_idx)
    train_patch_records.to_csv(output_dir / "temporal_unet_train_patch_iou_by_sample.csv", index=False)
    test_patch_records.to_csv(output_dir / "temporal_unet_test_patch_iou_by_sample.csv", index=False)
    metrics["train_patch_iou_eval"] = train_patch_iou
    metrics["test_patch_iou_eval"] = test_patch_iou

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "channels": channel_names,
            "normalization": normalization,
            "metrics": metrics,
            "base_channels": args.base_channels,
        },
        output_dir / "temporal_unet_smoke_coverage.pt",
    )
    metrics["example_images"] = write_examples(
        test_prob,
        test_y,
        test_valid,
        metadata,
        test_idx,
        output_dir,
        args.max_examples,
        args.example_threshold,
    )
    (output_dir / "temporal_unet_smoke_coverage_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
