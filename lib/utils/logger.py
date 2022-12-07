import logging
import colorlog
import os, time


class Logging:

    def __init__(self):
        self.format_str = ">>>>>>%(asctime)s-%(levelname)s-%(filename)s-Line:%(lineno)d-Message:%(message)s"

    def make_log_dir(self, dir_name='log'):
        log_dir = os.path.join(dir_name)
        log_dir = os.path.normpath(log_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    def make_log_file(self, dir_name='log'):
        log_dir = self.make_log_dir(dir_name)
        log_name = f"{time.strftime('%Y-%m-%d-%H', time.localtime())}.log"
        log_path = os.path.join(log_dir, log_name)
        log_path = os.path.normpath(log_path)
        return log_path

    def logger(self, level='DEBUG'):
        logger = logging.getLogger()
        level = logging.DEBUG
        logger.setLevel(level)

        fh = logging.FileHandler(self.make_log_file(), 'a', 'utf-8')
        sh = logging.StreamHandler()

        formatter = logging.Formatter(fmt=self.format_str)

        fh.setFormatter(formatter)
        sh.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(sh)
        return logger


logger = Logging().logger()

if __name__ == '__main__':
    init_logger = Logging()
    logger.warning('test')