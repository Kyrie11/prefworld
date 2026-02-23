from __future__ import annotations

import argparse
import glob
import json
import random
from pathlib import Path
from typing import List

from prefworld.utils.config import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Create train/val split files from a directory of nuPlan .db logs")
    parser.add_argument("--db_dir", type=str, required=True, help="Directory containing .db files")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for split lists")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dbs = sorted(glob.glob(str(Path(args.db_dir) / "*.db")))
    if len(dbs) == 0:
        raise RuntimeError(f"No .db files found under {args.db_dir}")

    random.seed(args.seed)
    random.shuffle(dbs)
    n_val = int(round(len(dbs) * args.val_ratio))
    val = dbs[:n_val]
    train = dbs[n_val:]

    out_dir = Path(args.out_dir)
    ensure_dir(str(out_dir))
    (out_dir / "train_dbs.json").write_text(json.dumps(train, indent=2), encoding="utf-8")
    (out_dir / "val_dbs.json").write_text(json.dumps(val, indent=2), encoding="utf-8")
    print(f"Wrote {len(train)} train and {len(val)} val db paths to {out_dir}")


if __name__ == "__main__":
    main()
