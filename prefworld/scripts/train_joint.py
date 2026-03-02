"""Alias for stage-3 joint fine-tuning.

Some projects prefer a separate entrypoint named `train_joint.py`.
This file simply forwards to `train_stage3_joint.py`.
"""

from prefworld.scripts.train_stage3_joint import main


if __name__ == "__main__":
    main()
