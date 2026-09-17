"""
Daily end-of-day pipeline for the Indian Stock Trader project.

Pipeline:
1. Incrementally refresh NSE OHLCV and Nifty 50 data.
2. Generate daily, weekly, monthly predictions.
3. Apply the final 9-condition entry engine.
4. Save a run record and full console log.

Run manually:
    python daily_pipeline.py

Schedule this script with Windows Task Scheduler after NSE close.
"""

import logging
import os
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
LOG_DIR = PROJECT_DIR / "logs"

# The scripts execute in this exact order.
PIPELINE_STEPS = [
    "refresh_market_data.py",
    "predict_weekly_monthly.py",
    "entry_decision_engine.py",

    # Manage existing paper positions before opening new ones.
    "exit_decision_engine.py",

    # Mark positions, open new eligible paper positions, save snapshot.
    "portfolio_engine.py",
]


def configure_logging(run_id: str) -> Path:
    """Configure console and timestamped file logs."""
    LOG_DIR.mkdir(exist_ok=True)

    log_file = LOG_DIR / f"daily_pipeline_{run_id}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(
                log_file,
                encoding="utf-8",
            ),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )

    return log_file


def run_step(script_name: str) -> None:
    """
    Execute one Python file using the same Python interpreter that
    launched this orchestrator.
    """
    script_path = PROJECT_DIR / script_name

    if not script_path.exists():
        raise FileNotFoundError(
            f"Required pipeline file not found: {script_path}"
        )

    logging.info("=" * 80)
    logging.info("STARTING STEP: %s", script_name)
    logging.info("=" * 80)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PROJECT_DIR),
        text=True,
        capture_output=True,
        check=False,
    )

    if result.stdout:
        logging.info(
            "%s STDOUT:\n%s",
            script_name,
            result.stdout,
        )

    if result.stderr:
        logging.warning(
            "%s STDERR:\n%s",
            script_name,
            result.stderr,
        )

    if result.returncode != 0:
        raise RuntimeError(
            f"{script_name} failed with exit code "
            f"{result.returncode}"
        )

    logging.info("COMPLETED STEP: %s", script_name)


def main() -> None:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = configure_logging(run_id)

    started_at = datetime.now()

    logging.info("=" * 80)
    logging.info("DAILY MARKET PIPELINE STARTED")
    logging.info("Run ID: %s", run_id)
    logging.info("Project directory: %s", PROJECT_DIR)
    logging.info("Python executable: %s", sys.executable)
    logging.info("=" * 80)

    pipeline_success = False
    failed_step = None

    try:
        for script_name in PIPELINE_STEPS:
            failed_step = script_name
            run_step(script_name)

        pipeline_success = True

    except Exception as exc:
        logging.error(
            "PIPELINE FAILED at %s: %s",
            failed_step,
            exc,
        )

        logging.error(
            "TRACEBACK:\n%s",
            traceback.format_exc(),
        )

        raise

    finally:
        finished_at = datetime.now()
        duration = finished_at - started_at

        logging.info("=" * 80)
        logging.info(
            "DAILY MARKET PIPELINE %s",
            "SUCCEEDED" if pipeline_success else "FAILED",
        )
        logging.info("Started:  %s", started_at)
        logging.info("Finished: %s", finished_at)
        logging.info("Duration: %s", duration)
        logging.info("Log file: %s", log_file)
        logging.info("=" * 80)


if __name__ == "__main__":
    main()