"""Entry point: runs every experiment in sequence and saves each one's
console output to a timestamped log file under logs/, in addition to
printing it. Usage: python run.py
"""

import contextlib
import io
import os
import sys

from experiments import exp0_entity_discovery
from experiments import exp1_causal_flip
from experiments import exp2_conjunctive_rule
from experiments import exp3_planning
from experiments import exp4_curiosity
from experiments import exp5_window_sweep
from experiments import exp6_habit_formation
from experiments import exp7_multi_concept
from experiments import exp8_perception
from experiments import exp9_sequence_scaling
from experiments import exp10_context_scaling
from experiments import baseline_qlearning

EXPERIMENTS = [
    ("exp0_entity_discovery", exp0_entity_discovery.main),
    ("exp1_causal_flip", exp1_causal_flip.main),
    ("exp2_conjunctive_rule", exp2_conjunctive_rule.main),
    ("exp3_planning", exp3_planning.main),
    ("exp4_curiosity", exp4_curiosity.main),
    ("exp5_window_sweep", exp5_window_sweep.main),
    ("exp6_habit_formation", exp6_habit_formation.main),
    ("exp7_multi_concept", exp7_multi_concept.main),
    ("exp8_perception", exp8_perception.main),
    ("exp9_sequence_scaling", exp9_sequence_scaling.main),
    ("exp10_context_scaling", exp10_context_scaling.main),
    ("baseline_qlearning", baseline_qlearning.main),
]


class _Tee(io.TextIOBase):
    """Writes to two streams at once, so console output and the saved log
    file always stay identical -- no separate code path that could drift.
    """

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for stream in self.streams:
            stream.write(s)
        return len(s)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def run_and_log(name, main_fn, log_dir):
    log_path = os.path.join(log_dir, f"{name}.log")
    with open(log_path, "w", encoding="utf-8") as log_file:
        tee = _Tee(sys.stdout, log_file)
        with contextlib.redirect_stdout(tee):
            main_fn()
    return log_path


if __name__ == "__main__":
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)

    for name, main_fn in EXPERIMENTS:
        run_and_log(name, main_fn, log_dir)

    print(f"\nAll experiment logs saved to {log_dir}")
