import logging

logger = logging.getLogger(__name__)


def setup_logging(level=logging.INFO):
    """
    Set up the logging configuration for the application.
    """
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=level
    )
    logger.info("Logging is set up with level: %s", logging.getLevelName(level))
    logger.debug("Debug logging is enabled.")
