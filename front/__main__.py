import logging
import os
import sys

from loguru import logger

from front.pipeline import main as run_pipeline

# Fichier de log permanent (en plus de la console) — indispensable pour diagnostiquer
# les sessions sur la machine distante. Chemin surchargeable via JERRY_LOG_FILE.
_LOG_FILE = os.environ.get("JERRY_LOG_FILE", "jerry.log")


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO)
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(_LOG_FILE, level="DEBUG", rotation="10 MB", retention=5, enqueue=True)


def main() -> None:
    _configure_logging()
    logger.info("front ready")
    run_pipeline()


if __name__ == "__main__":
    main()
