#!/usr/bin/env python3

from collections import defaultdict, namedtuple
import os
import sys

import matplotlib.pyplot as plt

from benchmark_utils import get_benchmark_dicts, setup_directory

Instance = namedtuple('Instance', 'item_type reduction_factor')
SubInstance = namedtuple(
    'SubInstance', 'file_name impl block_size sub_warp_size items_per_thread')


# these mappings need to kept up to date with cpp....
def str_item_type(n):
    return ['float'][int(n)]


def str_reduce_impl(n):
    return ['MineGeneric', 'CUBGeneric', 'CUBSum', 'Mine'][int(n)]


def get_instance(run):
    return Instance(str_item_type(run['item_type']),
                    int(run['reduction_factor']))


def get_sub_instance(file_name, run):
    return SubInstance(file_name, str_reduce_impl(run['impl']),
                       int(run['block_size']), int(run['sub_warp_size']),
                       int(run['items_per_thread']))


def format_legend(file_names, sub_instance):
    items = [*sub_instance]
    if len(file_names) == 1:
        items = items[1:]
    return str(items)


def main(json_files):
    json_files, benchmark_dicts = get_benchmark_dicts(json_files)
    directory = setup_directory('reduce')

    instances = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: [None] * 2)))
    name_to_info = {}
    for file_name, runs in zip(json_files, benchmark_dicts):
        for run in runs:
            run_type = run['run_type']
            run_name = run['run_name']
            if run_type == 'iteration':
                instance = get_instance(run)
                sub_instance = get_sub_instance(file_name, run)
                size = int(run['size'])
                name_to_info[run['run_name']] = (instance, sub_instance, size)
                continue
            assert run_type == 'aggregate'
            instance, sub_instance, size = name_to_info[run_name]

            # if (sub_instance.items_per_thread != 1
            #         or sub_instance.block_size != 512):
            #     continue
            item = instances[instance][sub_instance][size]
            if run['aggregate_name'] == 'mean':
                item[0] = run['real_time']
            elif run['aggregate_name'] == 'stddev':
                item[1] = 2 * run['real_time']
    for instance, sub_instances in instances.items():
        for size_dict in sub_instances.values():
            sizes, times, errs = zip(*((size, time, err)
                                       for size, (time,
                                                  err) in size_dict.items()))
            assert not any(size is None for size in sizes)
            assert not any(time is None for time in times)
            assert not any(err is None for err in errs)
            plt.errorbar(sizes, times, yerr=errs, capsize=10.0)

        plt.gcf().set_size_inches(15, 10)
        legend = [
            format_legend(json_files, sub_instance)
            for sub_instance in sub_instances.keys()
        ]
        plt.xlabel('size')
        plt.ylabel('time (ms)')
        # fig.yscale('log')
        plt.legend(legend, loc='lower right', ncol=1)
        plt.savefig(
            os.path.join(
                directory, '{}_{}.png'.format(instance.item_type,
                                              instance.reduction_factor)))
        plt.clf()


if __name__ == "__main__":
    main(sys.argv[1:])
