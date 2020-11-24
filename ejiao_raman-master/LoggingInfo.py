import logging
import time
import os
import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='logging Demo')
    parser.add_argument('--not-save', default='True', action='store_true',
                        help='if yes,only output to terminal')
    parser.add_argument('--work-dir', default='./work_dir',
                        help='the work folder for storing results')
    return parser

def loadLogger(args):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt="[ %(asctime)s ] %(message)s",
                                  datefmt="%a %b %d %H:%M:%S %Y")

    sHandler = logging.StreamHandler()
    sHandler.setFormatter(formatter)

    logger.addHandler(sHandler)

    if not args.not_save:
        work_dir = os.path.join(args.work_dir,
                                time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()))
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)

        fHandler = logging.FileHandler(work_dir + '/log.txt', mode='w')
        fHandler.setLevel(logging.DEBUG)
        fHandler.setFormatter(formatter)

        logger.addHandler(fHandler)

    return logger

