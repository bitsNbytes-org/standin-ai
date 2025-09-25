import logging
import os
def setup_logging() -> logging.Logger:
    """Setup application-wide logging configuration."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_format = os.getenv(
        "LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            # Add file handler if LOG_FILE environment variable is set
            *(
                [logging.FileHandler(os.getenv("LOG_FILE"))]
                if os.getenv("LOG_FILE")
                else []
            ),
        ],
    )

    return logging.getLogger(__name__)
def getLogger():
    logger = setup_logging()
    return logger
