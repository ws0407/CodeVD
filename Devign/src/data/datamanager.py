import glob
import pandas as pd
import numpy as np
import os
from ..utils.functions import parse
from ..utils.objects.input_dataset import InputDataset
from sklearn.model_selection import train_test_split


def read(path, json_file):
    """读取path+json_file文件并返回DataFrame格式
    :param path: str
    :param json_file: str
    :return DataFrame
    """
    return pd.read_json(path + json_file)


def get_ratio(dataset, ratio):
    """从前往后截取ratio比例的数据
    :param dataset:
    :param ratio: 0~1
    :return:
    """
    approx_size = int(len(dataset) * ratio)
    return dataset[:approx_size]


def load(path, pickle_file, ratio=1):
    """加载path+pickle_file的dataset文件
    :param path: str
    :param pickle_file: str
    :param ratio: 0~1
    :return: dataset
    """
    dataset = pd.read_pickle(path + pickle_file)
    dataset.info(memory_usage='deep')
    if ratio < 1:
        dataset = get_ratio(dataset, ratio)
    return dataset


def write(data_frame: pd.DataFrame, path, file_name):
    """将数据集导出为pickle文件
    :param data_frame: pd.DataFrame
    :param path: str
    :param file_name: str
    """
    data_frame.to_pickle(path + file_name)


def apply_filter(data_frame: pd.DataFrame, filter_func):
    """根据过滤函数过滤数据集
    :param data_frame: pd.DataFrame
    :param filter_func: func 过滤函数
    """
    return filter_func(data_frame)


def rename(data_frame: pd.DataFrame, old, new):
    """DataFrame的某一列重命名
    :param data_frame: DataFrame
    :param old: str
    :param new: str
    """
    return data_frame.rename(columns={old: new})


def tokenize(data_frame: pd.DataFrame):
    data_frame.func = data_frame.func.apply(parse.tokenizer)
    # Change column name
    data_frame = rename(data_frame, 'func', 'tokens')
    # Keep just the tokens
    return data_frame[["tokens"]]


def to_files(data_frame: pd.DataFrame, out_path):
    """将数据集内每个函数导出为单个的c文件
    :param data_frame: 是一个slice，数据集的一部分
    :param out_path: str
    """
    # path = f"{self.out_path}/{self.dataset_name}/"
    os.makedirs(out_path)

    for idx, row in data_frame.iterrows():
        file_name = f"{idx}.c"
        with open(out_path + file_name, 'w') as f:
            f.write(row.func)


def create_with_index(data, columns):
    data_frame = pd.DataFrame(data, columns=columns)
    data_frame.index = list(data_frame["Index"])

    return data_frame


def inner_join_by_index(df1, df2):
    return pd.merge(df1, df2, left_index=True, right_index=True)


def train_val_test_split(data_frame: pd.DataFrame, shuffle=True):
    print("Splitting Dataset")

    false = data_frame[data_frame.target == 0]
    true = data_frame[data_frame.target == 1]

    train_false, test_false = train_test_split(false, test_size=0.2, shuffle=shuffle)
    test_false, val_false = train_test_split(test_false, test_size=0.5, shuffle=shuffle)
    train_true, test_true = train_test_split(true, test_size=0.2, shuffle=shuffle)
    test_true, val_true = train_test_split(test_true, test_size=0.5, shuffle=shuffle)

    train = train_false.append(train_true)
    val = val_false.append(val_true)
    test = test_false.append(test_true)

    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)
    test = test.reset_index(drop=True)

    return InputDataset(train), InputDataset(test), InputDataset(val)


def get_directory_files(directory):
    return [os.path.basename(file) for file in glob.glob(f"{directory}/*.pkl")]


def loads(data_sets_dir, ratio=1):
    data_sets_files = sorted([f for f in os.listdir(data_sets_dir) if os.path.isfile(os.path.join(data_sets_dir, f))])

    if ratio < 1:
        data_sets_files = get_ratio(data_sets_files, ratio)

    dataset = load(data_sets_dir, data_sets_files[0])
    data_sets_files.remove(data_sets_files[0])

    for ds_file in data_sets_files:
        dataset = dataset.append(load(data_sets_dir, ds_file))

    return dataset


def clean(data_frame: pd.DataFrame):
    """去除重复的函数（subset="func"）"""
    return data_frame.drop_duplicates(subset="func", keep=False)


def drop(data_frame: pd.DataFrame, keys):
    """移除keys的列"""
    for key in keys:
        del data_frame[key]


def slice_frame(data_frame: pd.DataFrame, size: int):
    """根据size切片数据集，结果是按顺序每size个数据组成的DataFrame
    >>> np.arange(10)
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> np.arange(10) // 2
    array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    """
    data_frame_size = len(data_frame)
    return data_frame.groupby(np.arange(data_frame_size) // size)
