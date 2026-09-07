import logging
from logging.handlers import TimedRotatingFileHandler
import os

def setup_logging():
    # Create logs directory
    if not os.path.exists("logs"):
        os.makedirs("logs")

    # Configuration for the Root Logger
    # This is where the "magic" happens for all sub-loggers
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Add the specific handler to the root logger
    file_handler = TimedRotatingFileHandler(
        "logs/app_log.log", 
        when="D", 
        interval=1, 
        backupCount=7
    )

    # -----------------------------
    # Third-party noise control
    # -----------------------------
    NOISY_LOGGERS = [
        "httpx",
        "httpcore",
        "urllib3",
        "asyncio",
        "trafilatura",
        "hugging_face",
        "sentence_transformers"
    ]

    for logger_name in NOISY_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    logging.getLogger().addHandler(file_handler)

# Call the setup immediately so it configures the root logger 
# as soon as this file is imported.
setup_logging()
