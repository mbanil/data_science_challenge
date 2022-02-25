import logging
import time

import sys

try:
    timestr = time.strftime("%Y%m%d%H%M%S")
    LOG = logging
    LOG.basicConfig(
                    format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.INFO,
                    handlers=[
                        logging.FileHandler(
                                filename='./src/logger/log/log_{}.txt'.format(timestr),
                                mode='a'
                            ),
                        logging.StreamHandler()
                    ])
except Exception :
    logging.exception("Something went wrong while initializing the custom logger")
    sys.exit("Something went wrong while initializing the custom logger")