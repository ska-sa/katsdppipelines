import argparse
import logging
import os
import os.path

import six

import OTObit
import ObitTask

from aips_context import aips_context
from aips_parser import aips_cfg

logging.basicConfig(level=logging.INFO)

def create_parser():
    """ Argument Parser """
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("-c", "--config", required=True)
    return parser

def create_mf_task(args):
    """ Create MF Image """
    task = ObitTask.ObitTask("MFImage")

    path, file  = os.path.split(args.input)
    base_filename, ext = os.path.splitext(file)

    cfg = aips_cfg(args.config, args.input)

    # Dump configuration options onto the task
    for k, v in six.iteritems(cfg):
        setattr(task, k, v)

    return task

with aips_context():
    task = create_mf_task(create_parser().parse_args())
    task.g

