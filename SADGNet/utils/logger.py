import logging
import os
import sys


class MyLogging:
    def __init__(self, name='mylog', log_level=logging.INFO):
        self.formatter = '%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
        self.date_format = '%Y-%m-%d %H:%M:%S'
        self.logging = logging
        self.log_level = log_level
        self.name = name

    def init_stdout_handle(self, formatter):
        stdoutHandler = self.logging.StreamHandler(sys.stdout)
        stdoutHandler.setLevel(self.log_level)
        stdoutHandler.setFormatter(formatter)
        return stdoutHandler

    def init_logger(self):
        formatter = self.logging.Formatter(self.formatter)
        logger = self.logging.getLogger(self.name)
        logger.setLevel(self.log_level)
        logger.addHandler(self.init_stdout_handle(formatter))
        return logger


name = os.path.basename(__file__)
my_logger = MyLogging(name=name).init_logger()

