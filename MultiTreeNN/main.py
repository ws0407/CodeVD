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
import preprocessing as pre
from utils.functions import parse_to_nodes
from utils.embeddings import nodes_to_input
from utils.process import *
from model import *

PATHS = config.Paths()
FILES = config.Files()
DEVICE = FILES.get_device()


def prepare_task():
    context = config.Create()
    raw = pre.read_json(PATHS.raw, FILES.raw)
    filtered = pre.apply_filter(raw, pre.select)
    filtered = pre.clean(filtered)
    pre.drop(filtered, ["commit_id"])   # "project"
    slices = pre.slice_frame(filtered, context.slice_size)
    # s是序号(0,1,2,...), slice是切片，slice_size行n列的数据
    slices = [(s, slice.apply(lambda x: x)) for s, slice in slices]

    cpg_files = []
    # Create CPG binary files
    for s, slice in slices:
        pre.to_files(slice, PATHS.workspace, PATHS.header)
        cpg_file = pre.joern_parse(context.joern_cli_dir, PATHS.workspace, PATHS.cpg, f"{s}_{FILES.cpg}")
        cpg_files.append(cpg_file)
        print(f"Dataset {s} to cpg.")
        shutil.rmtree(PATHS.workspace)  # 删除原有c文件
    # Create CPG with graphs json files
    json_files = pre.joern_create(context.joern_cli_dir, context.script, PATHS.cpg, PATHS.cpg, cpg_files)
    for (s, slice), json_file in zip(slices, json_files):
        graphs = pre.json_process(PATHS.cpg, json_file)
        if graphs is None:
            print(f"Dataset chunk {s} not processed.")
            continue
        dataset = pre.create_with_index(graphs, ["Index", "cpg"])
        dataset = pre.inner_join_by_index(slice, dataset)
        print(f"Writing cpg dataset chunk {s}.")
        pre.df_to_pickle(dataset, PATHS.cpg, f"{s}_{FILES.cpg}.pkl")
        del dataset
        gc.collect()


def embed_task():
    context = config.Embed()
    # Tokenize source code into tokens
    dataset_files = pre.get_directory_files(PATHS.cpg)
    w2vmodel = Word2Vec(**context.w2v_args)
    w2v_init = True
    for pkl_file in dataset_files:
        file_name = pkl_file.split(".")[0]
        cpg_dataset = pre.load_pickle(PATHS.cpg, pkl_file)
        tokens_dataset = pre.tokenize(cpg_dataset)
        pre.df_to_pickle(tokens_dataset, PATHS.tokens, f"{file_name}_{FILES.tokens}")
        # word2vec used to learn the initial embedding of each token
        w2vmodel.build_vocab(sentences=tokens_dataset.tokens, update=not w2v_init)
        w2vmodel.train(tokens_dataset.tokens, total_examples=w2vmodel.corpus_count, epochs=1)
        if w2v_init:
            w2v_init = False
        # Embed cpg to node representation and pass to graph data structure
        cpg_dataset["nodes"] = cpg_dataset.apply(lambda row: parse_to_nodes(row.cpg, context.nodes_dim), axis=1)
        # remove rows with no nodes
        cpg_dataset = cpg_dataset.loc[cpg_dataset.nodes.map(len) > 0]
        cpg_dataset["input"] = cpg_dataset.apply(
            lambda row: nodes_to_input(row.nodes, row.target, context.nodes_dim,
                                       w2vmodel.wv, context.edge_type), axis=1)
        pre.drop(cpg_dataset, ["nodes"])
        print(f"Saving input dataset {file_name} with size {len(cpg_dataset)}.")
        pre.df_to_pickle(cpg_dataset[["input", "target"]], PATHS.input, f"{file_name}_{FILES.input}")
        del cpg_dataset
        gc.collect()
    print("Saving w2vmodel.")
    w2vmodel.save(f"{PATHS.w2v}/{FILES.w2v}")


def train_task(stopping):
    context = config.Process()
    devign = config.Devign()
    model_path = PATHS.model + FILES.model
    model = Devign(path=model_path, device=DEVICE, model=devign.model, learning_rate=devign.learning_rate,
                   weight_decay=devign.weight_decay, loss_lambda=devign.loss_lambda)
    train = Train(model, context.epochs)
    input_dataset = pre.loads(PATHS.input)
    # split the dataset and pass to DataLoader with batch size
    train_loader, val_loader, test_loader = list(
        map(lambda x: x.get_loader(context.batch_size, shuffle=context.shuffle),
            train_val_test_split(input_dataset, shuffle=context.shuffle)))
    train_loader_step = LoaderStep("Train", train_loader, DEVICE)
    val_loader_step = LoaderStep("Validation", val_loader, DEVICE)
    test_loader_step = LoaderStep("Test", test_loader, DEVICE)

    if stopping:
        early_stopping = EarlyStopping(model, patience=context.patience)
        train(train_loader_step, val_loader_step, early_stopping)
        model.load()
    else:
        train(train_loader_step, val_loader_step)
        model.save()

    predict(model, test_loader_step)


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

    # print(args.prepare, args.embed, args.train, args.train_stopping)

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
