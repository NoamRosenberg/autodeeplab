import os
import time
import logging
import os.path as osp

import torch.distributed as dist


def setup_logger(logpth):
    logfile = 'Deeplab_v3plus-{}.log'.format(time.strftime('%Y-%m-%d-%H-%M-%S'))
    logfile = osp.join(logpth, logfile)
    FORMAT = '%(levelname)s %(filename)s(%(lineno)d): %(message)s'
    log_level = logging.INFO
    if dist.is_initialized() and dist.get_rank()!=0:
        log_level = logging.WARNING
    logging.basicConfig(level=log_level, format=FORMAT, filename=logfile)
    logging.root.addHandler(logging.StreamHandler())


class Logger(object):
    def __init__(self, args, logger_str):
        self._logger_name = args.save_path
        # if os.path
        self._logger_str = logger_str
        self._save_path = os.path.join(
            self._logger_name, self._logger_str+'.txt')
        # self._save_path = os.path.abspath(_save_path)
        self._file = open(self._save_path, 'w')

    def log(self, string, save=True):
        print(string)
        if save:
            self._file.write('{:}\n'.format(string))
            self._file.flush()

    def close(self):
        self._file.close()
