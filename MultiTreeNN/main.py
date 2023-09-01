"""
# urllib3 >= 1.21.1, <= 1.26
# chardet >= 3.0.2, < 5.0.0
"""

import argparse
import gc
import shutil
from argparse import ArgumentParser

from gensim.models.word2vec import Word2Vec

import config
# from preprocessing import *
import preprocessing as pre

PATHS = config.Paths()
FILES = config.Files()
DEVICE = FILES.get_device()


def prepare_task():
    context = config.Create()
    raw = pre.read(PATHS.raw, FILES.raw)
    filtered = pre.apply_filter(raw, pre.select)
    filtered = pre.clean(filtered)
    pre.drop(filtered, ["commit_id", "project"])
    slices = pre.slice_frame(filtered, context.slice_size)
    # s是序号(0,1,2,...), slice是切片，slice_size行n列的数据
    slices = [(s, slice.apply(lambda x: x)) for s, slice in slices]

    cpg_files = []
    # Create CPG binary files
    for s, slice in slices:
        pre.to_files(slice, PATHS.joern)
        cpg_file = pre.joern_parse(context.joern_cli_dir, PATHS.joern, PATHS.cpg, f"{s}_{FILES.cpg}")
        cpg_files.append(cpg_file)
        print(f"Dataset {s} to cpg.")
        shutil.rmtree(PATHS.joern)  # 无条件地删除指定的目录及其内容，包括所有的子目录和文件，且无法恢复。
    # Create CPG with graphs json files
    json_files = pre.joern_create(context.joern_cli_dir, PATHS.cpg, PATHS.cpg, cpg_files)
    for (s, slice), json_file in zip(slices, json_files):
        graphs = pre.json_process(PATHS.cpg, json_file)
        if graphs is None:
            print(f"Dataset chunk {s} not processed.")
            continue
        dataset = pre.create_with_index(graphs, ["Index", "cpg"])
        dataset = pre.inner_join_by_index(slice, dataset)
        print(f"Writing cpg dataset chunk {s}.")
        pre.write(dataset, PATHS.cpg, f"{s}_{FILES.cpg}.pkl")
        del dataset
        gc.collect()
    pass


def embed_task():
    print('embed')
    pass


def train_task(stopping):
    print('train', stopping)
    pass


def main():
    """
    main function that executes tasks based on command-line options
    """
    parser: ArgumentParser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prepare', help='Prepare dataset', action='store_true')
    parser.add_argument('-e', '--embed', help='Embed dataset', action='store_true')
    parser.add_argument('-t', '--train', help='Train model', action='store_true')
    parser.add_argument('-ts', '--train_stopping', help='Train model with early stopping', action='store_true')

    args = parser.parse_args()

    if args.prepare:
        prepare_task()
    if args.embed:
        embed_task()
    if args.train:
        train_task(False)
    if args.train_stopping:
        train_task(True)


if __name__ == '__main__':
    main()
