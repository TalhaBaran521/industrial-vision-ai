"""Configure logging at the application entry point."""

import logging


def configure_logging(log_level: str) -> None:
    """Send logs to stderr using the application log level."""
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s %(name)s: %(message)s",
    )
