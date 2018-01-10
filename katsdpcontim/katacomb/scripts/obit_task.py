#!/usr/bin/env python

import argparse
import logging

from katacomb.obit_context import obit_context
from katacomb.util import task_factory, parse_python_assigns

log = logging.getLogger('katacomb')

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
