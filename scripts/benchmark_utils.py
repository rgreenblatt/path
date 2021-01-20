import json
import shutil
import os


def get_benchmark_dicts(json_files):
    json_files = list(
        filter(lambda x: x != "compile_commands.json", json_files))

    return json_files, [json.load(open(f))["benchmarks"] for f in json_files]


def get_directory(sub_name):
    return os.path.join('benchmark_plots', sub_name)

def setup_directory(sub_name):
    directory = get_directory(sub_name)
    shutil.rmtree(directory, ignore_errors=True)
    os.makedirs(directory, exist_ok=True)

    return directory
