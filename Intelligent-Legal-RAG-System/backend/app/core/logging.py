"""Configure structured logging for backend services."""

import logging


def configure_logging(log_level: str) -> None:
    """Configure application logging using the requested log level."""
    raise NotImplementedError


def get_logger(name: str) -> logging.Logger:
    """Return a named logger for application modules."""
    raise NotImplementedError

