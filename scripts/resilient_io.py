"""Retry transient Windows sharing violations without changing experiment semantics.

Usage: python -m scripts.resilient_io scripts.run_experiment [its arguments...]
The numerical implementation and evaluation digest are unchanged, including
when resuming a partially completed test evaluation.
"""

import os
import runpy
import sys
import time

_replace = os.replace


def replace_with_retry(source, destination, **kwargs):
    for attempt in range(12):
        try:
            return _replace(source, destination, **kwargs)
        except PermissionError:
            if attempt == 11:
                raise
            time.sleep(min(0.02 * 2**attempt, 0.5))


def main():
    if len(sys.argv) < 2 or not sys.argv[1].startswith("scripts."):
        raise SystemExit("Specify a project scripts.* module")
    module = sys.argv.pop(1)
    os.replace = replace_with_retry
    # Spawned workers must be able to import functions defined in the target
    # script's __main__, notably tune_unet.score_job on Windows.
    runpy.run_module(module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
