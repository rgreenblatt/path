#!/usr/bin/env python3

import sys
import json
import re
import os
import shutil

import matplotlib.pyplot as plt


def extract_info(json_data):
    return json_data["benchmarks"]


def get_name(run):
    m = re.search('(.*)_[0-9]*_([0-9]*x[0-9]*)/?.*', run['run_name'])
    return m.group(1) + '_' + m.group(2)


def main(csv_files):
    benchmark_dicts = [json.load(open(f))["benchmarks"] for f in csv_files]

    names = set()

    for runs in benchmark_dicts:
        for run in runs:
            names.add(get_name(run))

    directory = 'benchmark_plots'
    shutil.rmtree(directory)
    os.makedirs(directory, exist_ok=True)
    for name in names:
        file_names = []
        for file_name, runs in zip(csv_files, benchmark_dicts):
            file_names.append(file_name)
            times = []
            errors = []
            for run in runs:
                if get_name(run) == name:
                    times.append(run['real_time'])
                    errors.append(run['error'])

            plt.plot(times, errors)
        plt.xlabel('time')
        plt.ylabel('error')
        plt.yscale('log')
        plt.legend(file_names)
        plt.savefig(os.path.join(directory, name + '.png'))
        plt.clf()


if __name__ == "__main__":
    main(sys.argv[1:])
