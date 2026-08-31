import logging


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("front").info("front ready")


if __name__ == "__main__":
    main()
