#!/usr/bin/env python3

from collections import defaultdict, namedtuple
import os
import sys

import matplotlib.pyplot as plt

from benchmark_utils import get_benchmark_dicts, setup_directory

Instance = namedtuple('Instance', 'item_type reduction_factor')
Constants = namedtuple('Constants',
                       'block_size sub_warp_size items_per_thread')
Implementation = namedtuple('Implementation', 'file_name impl')


# these mappings need to kept up to date with cpp....
def str_item_type(n):
    return ['float', 'eigen_array3f'][int(n)]


def str_reduce_impl(n):
    return [
        'MineGeneric', 'CUBGeneric', 'CUBSum', 'Mine', 'GeneralKernelLaunch'
    ][int(n)]


def get_instance(run):
    return Instance(str_item_type(run['item_type']),
                    int(run['reduction_factor']))


def get_constants(run):
    return Constants(int(run['block_size']), int(run['sub_warp_size']),
                     int(run['items_per_thread']))


def get_implementation(file_name, run):
    return Implementation(file_name, str_reduce_impl(run['impl']))


def format_legend(file_names, sub_instance):
    items = [*sub_instance]
    if len(file_names) == 1:
        items = items[1:]
    return str(items)


def main(json_files):
    json_files, benchmark_dicts = get_benchmark_dicts(json_files)
    directory = setup_directory('reduce')

    instances_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(
        lambda: defaultdict(lambda: [None] * 2))))
    name_to_info = {}
    for file_name, runs in zip(json_files, benchmark_dicts):
        for run in runs:
            run_type = run['run_type']
            run_name = run['run_name']
            if run_type == 'iteration':
                instance = get_instance(run)
                constants = get_constants(run)
                impl = get_implementation(file_name, run)
                size = int(run['size'])
                name_to_info[run['run_name']] = (instance, constants, impl,
                                                 size)
                continue
            assert run_type == 'aggregate'
            instance, constants, impl, size = name_to_info[run_name]

            # if (sub_instance.items_per_thread != 1
            #         or sub_instance.block_size != 512):
            #     continue
            item = instances_dict[instance][constants][impl][size]
            if run['aggregate_name'] == 'mean':
                item[0] = run['real_time']
            elif run['aggregate_name'] == 'stddev':
                item[1] = 2 * run['real_time']

    compare_all_like = False
    for instance, constants_dict in instances_dict.items():

        def setup_save_fig(legend, file_name_no_ext):
            size = (15, 10) if compare_all_like else (10, 8)
            plt.gcf().set_size_inches(*size)
            plt.xlabel('size')
            plt.ylabel('time (ms)')
            # fig.yscale('log')
            plt.legend(legend, loc='lower right', ncol=1)
            plt.savefig(
                os.path.join(directory, '{}.png'.format(file_name_no_ext)))
            plt.clf()

        legend_items = []
        base_file_name = '{}_factor_{}'.format(instance.item_type,
                                               instance.reduction_factor)
        for constants, impls_dict in constants_dict.items():
            if not compare_all_like:
                legend_items.clear()
            for impl, size_dict in impls_dict.items():
                sizes, times, errs = zip(
                    *((size, time, err)
                      for size, (time, err) in size_dict.items()))
                assert not any(size is None for size in sizes)
                assert not any(time is None for time in times)
                assert not any(err is None for err in errs)
                plt.errorbar(sizes, times, yerr=errs, capsize=10.0)

                def formatting(named_tuple):
                    return dict(named_tuple._asdict())

                def remove_name_if(impl_dict):
                    if len(json_files) == 1:
                        del impl_dict['file_name']
                    return impl_dict

                legend_items.append(
                    (formatting(constants), remove_name_if(formatting(impl))))

            if not compare_all_like:
                legend = [str(impl) for _, impl in legend_items]

                f_name = '{}_block_size_{}_{}items_per_thread_{}'.format(
                    base_file_name, constants.block_size,
                    'sub_warp_size_{}_'.format(constants.sub_warp_size)
                    if constants.sub_warp_size != 0 else '',
                    constants.items_per_thread)

                setup_save_fig(legend, f_name)

        if compare_all_like:
            legend = [str(v) for v in legend_items]
            setup_save_fig(legend, base_file_name)


if __name__ == "__main__":
    main(sys.argv[1:])
