import logging
import os

class Logger:
    _logger = None  # Class-level variable to ensure a singleton logger

    @staticmethod
    def get_logger(log_file="logs/app.log"):
        """Set up a logger (singleton)."""
        if Logger._logger is None:
            # Ensure the logs directory exists
            os.makedirs(os.path.dirname(log_file), exist_ok=True)

            logger = logging.getLogger("ThyroidCancerDetection")
            logger.setLevel(logging.DEBUG)

            # File handler
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_format)

            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_format = logging.Formatter('%(levelname)s: %(message)s')
            console_handler.setFormatter(console_format)

            logger.addHandler(file_handler)
            logger.addHandler(console_handler)

            # Assign to the singleton logger
            Logger._logger = logger

        return Logger._logger
