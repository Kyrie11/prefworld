from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch


@dataclass
class CalibrationBins:
    """Online accumulator for reliability diagrams / ECE."""

    sum_conf: torch.Tensor  # [B]
    sum_acc: torch.Tensor   # [B]
    count: torch.Tensor     # [B]

    @property
    def n_bins(self) -> int:
        return int(self.count.numel())

    @property
    def total(self) -> torch.Tensor:
        return self.count.sum()


def init_bins(n_bins: int, *, device: torch.device) -> CalibrationBins:
    n = int(max(1, n_bins))
    z = torch.zeros((n,), device=device, dtype=torch.float32)
    return CalibrationBins(sum_conf=z.clone(), sum_acc=z.clone(), count=z.clone())


@torch.no_grad()
def update_bins(
    bins: CalibrationBins,
    confidence: torch.Tensor,  # [...]
    correct: torch.Tensor,     # [...], bool or 0/1
) -> CalibrationBins:
    """Accumulate confidence/correctness into equal-width bins on [0,1]."""
    conf = confidence.detach().to(torch.float32).clamp(0.0, 1.0).flatten()
    cor = correct.detach().to(torch.float32).flatten()
    if conf.numel() == 0:
        return bins

    n_bins = bins.n_bins
    # Bin index in [0, n_bins-1]
    idx = torch.clamp((conf * n_bins).to(torch.int64), 0, n_bins - 1)
    bins.count.scatter_add_(0, idx, torch.ones_like(conf, dtype=torch.float32))
    bins.sum_conf.scatter_add_(0, idx, conf)
    bins.sum_acc.scatter_add_(0, idx, cor)
    return bins


@torch.no_grad()
def finalize_bins(bins: CalibrationBins) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (bin_conf, bin_acc, ece)."""
    denom = bins.count.clamp_min(1.0)
    bin_conf = bins.sum_conf / denom
    bin_acc = bins.sum_acc / denom

    total = bins.total.clamp_min(1.0)
    ece = (bins.count / total * (bin_acc - bin_conf).abs()).sum()
    return bin_conf, bin_acc, ece


def save_reliability_artifacts(
    bin_conf: torch.Tensor,
    bin_acc: torch.Tensor,
    bin_count: torch.Tensor,
    out_dir: str | Path,
    *,
    tag: str = "maneuver",
) -> Optional[Path]:
    """Save reliability diagram data (+ an image if matplotlib is available)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / f"{tag}_reliability_bins.pt"
    torch.save(
        {
            "bin_conf": bin_conf.detach().cpu(),
            "bin_acc": bin_acc.detach().cpu(),
            "bin_count": bin_count.detach().cpu(),
        },
        data_path,
    )

    # Optional plot
    try:
        import matplotlib.pyplot as plt  # type: ignore

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot([0.0, 1.0], [0.0, 1.0])
        ax.plot(bin_conf.detach().cpu().numpy(), bin_acc.detach().cpu().numpy(), marker="o")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Reliability diagram ({tag})")
        img_path = out_dir / f"{tag}_reliability.png"
        fig.savefig(img_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return img_path
    except Exception:
        return None
