import logging
import time

today = time.strftime("%Y-%m-%d", time.localtime())


class Logger(logging.Logger):
    def __init__(self, log_path):
        super(Logger, self).__init__(__name__)
        self.log_path = log_path
        logger = logging.getLogger()
        logger.setLevel(level=logging.INFO)
        handler = logging.FileHandler(self.log_path, mode="w")
        handler.setLevel(level=logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)

        logger.addHandler(handler)
        logger.addHandler(console)


logger = Logger(log_path=f"./log/model_train_{today}.log")