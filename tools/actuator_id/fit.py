# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 nyxrobotics

"""Fit a bam actuator model against logs recorded by record.py.

Thin wrapper around bam's own `python -m bam.fit` CLI: bam.fit looks up
--actuator in bam.actuators.actuators, which has no "xc330" entry (see
xc330_actuator.py's docstring), so this registers it there before handing
off to bam.fit's own argparse/optimization loop - every other flag (see
bam.fit --help, e.g. --model, --trials, --method) passes through unchanged.

Needs optuna and wandb, which bam.fit imports unconditionally but which
aren't project dependencies (only record.py's rustypot/bam.trajectory path
is): `uv run --with optuna --with wandb python3 tools/actuator_id/fit.py ...`

Usage (record.py's raw logs are first resampled at a fixed dt by
bam.process, which bam.fit needs):
    mkdir -p tools/actuator_id/logs_processed
    uv run --group sim python3 -m bam.process \\
        --raw tools/actuator_id/logs --logdir tools/actuator_id/logs_processed
    uv run --with optuna --with wandb python3 tools/actuator_id/fit.py \\
        --logdir tools/actuator_id/logs_processed --actuator xc330 --model m6 \\
        --output xc330_params.json
"""

import sys

import bam.actuators
from bam.testbench import Pendulum

from xc330_actuator import XC330Actuator

bam.actuators.actuators["xc330"] = lambda: XC330Actuator(Pendulum)

if __name__ == "__main__":
    sys.argv[0] = "bam.fit"
    import bam.fit  # noqa: F401  (argparse + fitting run at import time)
