#!/usr/bin/env python

import argparse
import logging
import os
import os.path

import six

import OTObit
import ObitTask

from katsdpcontim.aips_context import aips_context
from katsdpcontim.aips_parser import aips_cfg, apply_cfg_to_task

logging.basicConfig(level=logging.INFO)

def create_parser():
    """ Argument Parser """
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("-c", "--config", required=True)
    return parser

def create_task(args):
    """ Create MF Image """
    task = ObitTask.ObitTask("MFImage")

    path, file  = os.path.split(args.input)
    base_filename, ext = os.path.splitext(file)

    # Obtain configuration options
    cfg = aips_cfg(args.config, args.input)

    # Create a representative log file for this task
    cfg['taskLog'] = ''.join(('mfimage-', base_filename, '.log'))

    # Apply configuration options to the task
    apply_cfg_to_task(task, cfg)

    return task

with aips_context():
    task = create_task(create_parser().parse_args())
    task.g

