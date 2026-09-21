# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

"""Fit a bam actuator model against logs recorded by record.py.

Thin wrapper around bam's own `python -m bam.fit` CLI: bam.fit looks up
--actuator in bam.actuators.actuators, which has no "xc330" entry (see
xc330_actuator.py's docstring), so this registers it there before handing
off to bam.fit's own argparse/optimization loop - every other flag (see
bam.fit --help, e.g. --model, --trials, --method) passes through unchanged.

Needs optuna and wandb, which bam.fit imports unconditionally but which
aren't project dependencies (only record.py's rustypot/bam.trajectory path
is): `uv run --with optuna --with wandb python3 tools/actuator_id/fit.py ...`

Usage:
    uv run --with optuna --with wandb python3 tools/actuator_id/fit.py \\
        --logdir tools/actuator_id/logs --actuator xc330 --model m6 \\
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
