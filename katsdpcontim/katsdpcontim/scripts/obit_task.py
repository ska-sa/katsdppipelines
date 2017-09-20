#!/usr/bin/env python

import argparse
import logging

from katsdpcontim.obit_context import obit_context
from katsdpcontim.util import task_factory, parse_python_assigns

log = logging.getLogger('katsdpcontim')

def create_parser():
    """ Argument Parser """
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("-c", "--config")
    parser.add_argument("-kw", "--keywords", type=parse_python_assigns)
    return parser

args = create_parser().parse_args()

with obit_context():
    task = task_factory(args.task, args.config, **args.keywords)

    try:
        task.go()
    except Exception as e:
        log.exception("Task exception")
        log.error("Please consult '%s'" % task.taskLog)
