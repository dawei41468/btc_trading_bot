import logging

class Monitoring:
    def __init__(self):
        """Initialize logging to file and console."""
        logging.basicConfig(filename='trade_log.log', level=logging.INFO,
                            format='%(asctime)s - %(message)s')

    def log(self, message):
        """Log message to file and print to console."""
        logging.info(message)
        print(message)