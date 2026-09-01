import logging

from front.pipeline import main as run_pipeline


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("front").info("front ready")
    run_pipeline()


if __name__ == "__main__":
    main()
