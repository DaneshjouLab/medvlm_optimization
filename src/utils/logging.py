# This source file is part of the Daneshjou Lab project
#
# SPDX-FileCopyrightText: 2025 Stanford University and the project authors (see AUTHORS.md)
#
# SPDX-License-Identifier: MIT

import datetime
import sys
import os


class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()


def setup_logging(exp_dir, experiment_name, args):
    os.makedirs(exp_dir, exist_ok=True)

    args_str = "_".join([f"{k}-{str(v).replace('/', '-')}" for k, v in vars(args).items()])
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{exp_dir}/{experiment_name}_{args_str}_{timestamp}.txt"

    logfile = open(log_filename, "w")
    sys.stdout = Tee(sys.__stdout__, logfile)
    sys.stderr = Tee(sys.__stderr__, logfile)

    return log_filename