import json
import glob
import re
import subprocess
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os

from utils import parse
from utils.objects import InputDataset


def prepare_folder(input_json, key, out_path):
    """原始数据集作为输入，提取其中的函数，然后保存为单个的cpp文件，一个文件夹对应一个函数"""
    with open(input_json, 'r') as f:
        data = json.load(f)
    for idx, sample in tqdm(enumerate(data)):
        func = sample[key]
        func_name = func[:func.find("(")].split()[-1]
        if not os.path.exists(out_path + f"{idx}-{func_name}"):
            os.makedirs(out_path + f"{idx}-{func_name}")
        with open(out_path + f"{idx}-{func_name}/{idx}-{func_name}.cpp", 'w') as f:
            f.write(func)
        # print(func)
        # print(func_name)
        # break


def read_json(path, json_file):
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


def load_pickle(path, pickle_file, ratio=1):
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


def loads(data_sets_dir, ratio=1):
    """从给定目录加载并组合多个数据集"""
    data_sets_files = sorted([f for f in os.listdir(data_sets_dir) if os.path.isfile(os.path.join(data_sets_dir, f))])
    if ratio < 1:
        data_sets_files = get_ratio(data_sets_files, ratio)
    dataset = load_pickle(data_sets_dir, data_sets_files[0])    # ? 为什么要先加载第一个文件
    data_sets_files.remove(data_sets_files[0])
    for ds_file in data_sets_files:                             # ? 为什么加载一个后可以直接append
        dataset = dataset.append(load_pickle(data_sets_dir, ds_file))
    return dataset


def df_to_pickle(data_frame: pd.DataFrame, path, file_name):
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


def create_with_index(data, columns):
    """根据提供的数据和列名创建一个DataFrame，并根据"Index"列设置索引到数据中"""
    data_frame = pd.DataFrame(data, columns=columns)
    data_frame.index = list(data_frame["Index"])
    return data_frame


def inner_join_by_index(df1, df2):
    """按照索引进行内连接（内部联接），即只保留两个DataFrame中索引相同的行。
    最终形成的df有['target', 'func', 'Index', 'cpg']"""
    return pd.merge(df1, df2, left_index=True, right_index=True)


def joern_parse(joern_path, input_path, output_path, file_name):
    """将每个slice下的所有c文件一同处理成一个bin文件
    :param joern_path: str
    :param input_path: str
    :param output_path: str 输出文件夹路径
    :param file_name: str 输出文件名，为slice索引
    :return: str
    """
    out_file = file_name + ".bin"
    joern_parse_call = subprocess.run(["./" + joern_path + "joern-parse", input_path, "--output", output_path + out_file],
                                      stdout=subprocess.PIPE, text=True, check=True)
    print(str(joern_parse_call))
    return out_file


def joern_create(joern_path, in_path, out_path, cpg_files):
    """将所有的cpg的bin文件处理成json的格式，通过执行joern命令打开交互窗口，并运行script脚本处理
    :param joern_path:
    :param in_path: 输入路径
    :param out_path: 输出路径
    :param cpg_files: bin文件，list
    :return:
    """
    joern_process = subprocess.Popen(["./" + joern_path + "joern"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    json_files = []
    for cpg_file in cpg_files:
        json_file_name = f"{cpg_file.split('.')[0]}.json"
        json_files.append(json_file_name)

        print(in_path+cpg_file)
        if os.path.exists(in_path+cpg_file):
            json_out = f"{os.path.abspath(out_path)}/{json_file_name}"
            import_cpg_cmd = f"importCpg(\"{os.path.abspath(in_path)}/{cpg_file}\")\r".encode()
            script_path = f"{os.path.dirname(os.path.abspath(joern_path))}/graph-for-funcs.sc"
            run_script_cmd = f"cpg.runScript(\"{script_path}\").toString() |> \"{json_out}\"\r".encode()
            joern_process.stdin.write(import_cpg_cmd)
            print(joern_process.stdout.readline().decode())
            joern_process.stdin.write(run_script_cmd)
            print(joern_process.stdout.readline().decode())
            joern_process.stdin.write("delete\r".encode())
            print(joern_process.stdout.readline().decode())
    try:
        outs, errs = joern_process.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        joern_process.kill()
        outs, errs = joern_process.communicate()
    if outs is not None:
        print(f"Outs: {outs.decode()}")
    if errs is not None:
        print(f"Errs: {errs.decode()}")
    return json_files


# def funcs_to_graphs(funcs_path):
#     client = CPGClientWrapper()
#     # query the cpg for the dataset
#     print(f"Creating CPG.")
#     graphs_string = client(funcs_path)
#     # removes unnecessary namespace for object references
#     graphs_string = re.sub(r"io\.shiftleft\.codepropertygraph\.generated\.", '', graphs_string)
#     graphs_json = json.loads(graphs_string)
#
#     return graphs_json["functions"]


def graph_indexing(graph):
    """根据graph文件名提取graph的索引，并返回索引+函数"""
    idx = int(graph["file"].split(".c")[0].split("/")[-1])
    del graph["file"]
    return idx, {"functions": [graph]}


def json_process(in_path, json_file):
    """处理json文件，移除无用字段，并返回函数的列表+索引"""
    if os.path.exists(in_path + json_file):
        with open(in_path + json_file) as jf:
            cpg_string = jf.read()
            # 替换：io.shiftleft.codepropertygraph.generated.nodes.Block[label=BLOCK; id=2305843009213694036]
            # 为：nodes.Block@2305843009213694036
            cpg_string = re.sub(r'io\.shiftleft\.codepropertygraph\.generated\.', '', cpg_string)
            cpg_string = re.sub(r'(\[label)(\D+)(\d+)\]', lambda match: f'@{match.group(3)}', cpg_string)
            cpg_json = json.loads(cpg_string)
            container = [graph_indexing(graph) for graph in cpg_json["functions"] if graph["file"] != "N/A"]
            return container
    return None


def get_directory_files(directory):
    """glob.glob获取所有pkl文件路径，使用os.path.basename只保留文件名"""
    return [os.path.basename(file) for file in glob.glob(f"{directory}/*.pkl")]


def tokenize(data_frame: pd.DataFrame):
    """转化成token序列，按照关键字、运算符、括号等分割，并将自定义 变量/函数 替换为 VAR# 和 FUN# """
    data_frame.func = data_frame.func.apply(parse.tokenizer)
    # Change column name
    data_frame = rename(data_frame, 'func', 'tokens')
    # Keep just the tokens
    return data_frame[["tokens"]]


def select(dataset):
    """测试使用，选取FFmpeg中的200个函数作为测试"""
    result = dataset.loc[dataset['project'] == "FFmpeg"]
    len_filter = result.func.str.len() < 1200
    result = result.loc[len_filter]
    # print(len(result))
    # result = result.iloc[11001:]
    # print(len(result))
    result = result.head(200)
    return result


if __name__ == '__main__':
    prepare_folder('/data/data/ws/CodeVD/IVDetect/data/Reveal/vulnerables.json', 'code',
                   '/data/data/ws/CodeVD/IVDetect/data/Reveal/vulnerable/')
    # prepare_folder('/data/data/ws/CodeVD/IVDetect/data/FFMPeg_Qemu/function.json')

    # _code = 'digraph v4l2_free_buffer {  \r\n\"1000100\" [label = \"(METHOD,v4l2_free_buffer)\" ]\r\n\"1000101\" [label = \"(PARAM,void *opaque)\" ]\r\n\"1000102\" [label = \"(PARAM,uint8_t *unused)\" ]\r\n\"1000103\" [label = \"(BLOCK,,)\" ]\r\n\"1000104\" [label = \"(LOCAL,avbuf: V4L2Buffer *)\" ]\r\n\"1000105\" [label = \"(<operator>.assignment,* avbuf = opaque)\" ]\r\n\"1000106\" [label = \"(IDENTIFIER,avbuf,* avbuf = opaque)\" ]\r\n\"1000107\" [label = \"(IDENTIFIER,opaque,* avbuf = opaque)\" ]\r\n\"1000108\" [label = \"(LOCAL,s: V4L2m2mContext *)\" ]\r\n\"1000109\" [label = \"(<operator>.assignment,*s = buf_to_m2mctx(avbuf))\" ]\r\n\"1000110\" [label = \"(IDENTIFIER,s,*s = buf_to_m2mctx(avbuf))\" ]\r\n\"1000111\" [label = \"(buf_to_m2mctx,buf_to_m2mctx(avbuf))\" ]\r\n\"1000112\" [label = \"(IDENTIFIER,avbuf,buf_to_m2mctx(avbuf))\" ]\r\n\"1000113\" [label = \"(CONTROL_STRUCTURE,if (atomic_fetch_sub(&avbuf->context_refcount, 1) == 1),if (atomic_fetch_sub(&avbuf->context_refcount, 1) == 1))\" ]\r\n\"1000114\" [label = \"(<operator>.equals,atomic_fetch_sub(&avbuf->context_refcount, 1) == 1)\" ]\r\n\"1000115\" [label = \"(atomic_fetch_sub,atomic_fetch_sub(&avbuf->context_refcount, 1))\" ]\r\n\"1000116\" [label = \"(<operator>.addressOf,&avbuf->context_refcount)\" ]\r\n\"1000117\" [label = \"(<operator>.indirectFieldAccess,avbuf->context_refcount)\" ]\r\n\"1000118\" [label = \"(IDENTIFIER,avbuf,atomic_fetch_sub(&avbuf->context_refcount, 1))\" ]\r\n\"1000119\" [label = \"(FIELD_IDENTIFIER,context_refcount,context_refcount)\" ]\r\n\"1000120\" [label = \"(LITERAL,1,atomic_fetch_sub(&avbuf->context_refcount, 1))\" ]\r\n\"1000121\" [label = \"(LITERAL,1,atomic_fetch_sub(&avbuf->context_refcount, 1) == 1)\" ]\r\n\"1000122\" [label = \"(BLOCK,,)\" ]\r\n\"1000123\" [label = \"(atomic_fetch_sub_explicit,atomic_fetch_sub_explicit(&s->refcount, 1, memory_order_acq_rel))\" ]\r\n\"1000124\" [label = \"(<operator>.addressOf,&s->refcount)\" ]\r\n\"1000125\" [label = \"(<operator>.indirectFieldAccess,s->refcount)\" ]\r\n\"1000126\" [label = \"(IDENTIFIER,s,atomic_fetch_sub_explicit(&s->refcount, 1, memory_order_acq_rel))\" ]\r\n\"1000127\" [label = \"(FIELD_IDENTIFIER,refcount,refcount)\" ]\r\n\"1000128\" [label = \"(LITERAL,1,atomic_fetch_sub_explicit(&s->refcount, 1, memory_order_acq_rel))\" ]\r\n\"1000129\" [label = \"(IDENTIFIER,memory_order_acq_rel,atomic_fetch_sub_explicit(&s->refcount, 1, memory_order_acq_rel))\" ]\r\n\"1000130\" [label = \"(CONTROL_STRUCTURE,if (s->reinit),if (s->reinit))\" ]\r\n\"1000131\" [label = \"(<operator>.indirectFieldAccess,s->reinit)\" ]\r\n\"1000132\" [label = \"(IDENTIFIER,s,if (s->reinit))\" ]\r\n\"1000133\" [label = \"(FIELD_IDENTIFIER,reinit,reinit)\" ]\r\n\"1000134\" [label = \"(BLOCK,,)\" ]\r\n\"1000135\" [label = \"(CONTROL_STRUCTURE,if (!atomic_load(&s->refcount)),if (!atomic_load(&s->refcount)))\" ]\r\n\"1000136\" [label = \"(<operator>.logicalNot,!atomic_load(&s->refcount))\" ]\r\n\"1000137\" [label = \"(atomic_load,atomic_load(&s->refcount))\" ]\r\n\"1000138\" [label = \"(<operator>.addressOf,&s->refcount)\" ]\r\n\"1000139\" [label = \"(<operator>.indirectFieldAccess,s->refcount)\" ]\r\n\"1000140\" [label = \"(IDENTIFIER,s,atomic_load(&s->refcount))\" ]\r\n\"1000141\" [label = \"(FIELD_IDENTIFIER,refcount,refcount)\" ]\r\n\"1000142\" [label = \"(sem_post,sem_post(&s->refsync))\" ]\r\n\"1000143\" [label = \"(<operator>.addressOf,&s->refsync)\" ]\r\n\"1000144\" [label = \"(<operator>.indirectFieldAccess,s->refsync)\" ]\r\n\"1000145\" [label = \"(IDENTIFIER,s,sem_post(&s->refsync))\" ]\r\n\"1000146\" [label = \"(FIELD_IDENTIFIER,refsync,refsync)\" ]\r\n\"1000147\" [label = \"(CONTROL_STRUCTURE,else,else)\" ]\r\n\"1000148\" [label = \"(CONTROL_STRUCTURE,if (avbuf->context->streamon),if (avbuf->context->streamon))\" ]\r\n\"1000149\" [label = \"(<operator>.indirectFieldAccess,avbuf->context->streamon)\" ]\r\n\"1000150\" [label = \"(<operator>.indirectFieldAccess,avbuf->context)\" ]\r\n\"1000151\" [label = \"(IDENTIFIER,avbuf,if (avbuf->context->streamon))\" ]\r\n\"1000152\" [label = \"(FIELD_IDENTIFIER,context,context)\" ]\r\n\"1000153\" [label = \"(FIELD_IDENTIFIER,streamon,streamon)\" ]\r\n\"1000154\" [label = \"(ff_v4l2_buffer_enqueue,ff_v4l2_buffer_enqueue(avbuf))\" ]\r\n\"1000155\" [label = \"(IDENTIFIER,avbuf,ff_v4l2_buffer_enqueue(avbuf))\" ]\r\n\"1000156\" [label = \"(av_buffer_unref,av_buffer_unref(&avbuf->context_ref))\" ]\r\n\"1000157\" [label = \"(<operator>.addressOf,&avbuf->context_ref)\" ]\r\n\"1000158\" [label = \"(<operator>.indirectFieldAccess,avbuf->context_ref)\" ]\r\n\"1000159\" [label = \"(IDENTIFIER,avbuf,av_buffer_unref(&avbuf->context_ref))\" ]\r\n\"1000160\" [label = \"(FIELD_IDENTIFIER,context_ref,context_ref)\" ]\r\n\"1000161\" [label = \"(METHOD_RETURN,static void)\" ]\r\n  \"1000100\" -> \"1000101\"  [ label = \"AST: \"] \r\n  \"1000100\" -> \"1000102\"  [ label = \"AST: \"] \r\n  \"1000100\" -> \"1000103\"  [ label = \"AST: \"] \r\n  \"1000100\" -> \"1000161\"  [ label = \"AST: \"] \r\n  \"1000103\" -> \"1000104\"  [ label = \"AST: \"] \r\n  \"1000103\" -> \"1000105\"  [ label = \"AST: \"] \r\n  \"1000103\" -> \"1000108\"  [ label = \"AST: \"] \r\n  \"1000103\" -> \"1000109\"  [ label = \"AST: \"] \r\n  \"1000103\" -> \"1000113\"  [ label = \"AST: \"] \r\n  \"1000105\" -> \"1000106\"  [ label = \"AST: \"] \r\n  \"1000105\" -> \"1000107\"  [ label = \"AST: \"] \r\n  \"1000109\" -> \"1000110\"  [ label = \"AST: \"] \r\n  \"1000109\" -> \"1000111\"  [ label = \"AST: \"] \r\n  \"1000111\" -> \"1000112\"  [ label = \"AST: \"] \r\n  \"1000113\" -> \"1000114\"  [ label = \"AST: \"] \r\n  \"1000113\" -> \"1000122\"  [ label = \"AST: \"] \r\n  \"1000114\" -> \"1000115\"  [ label = \"AST: \"] \r\n  \"1000114\" -> \"1000121\"  [ label = \"AST: \"] \r\n  \"1000115\" -> \"1000116\"  [ label = \"AST: \"] \r\n  \"1000115\" -> \"1000120\"  [ label = \"AST: \"] \r\n  \"1000116\" -> \"1000117\"  [ label = \"AST: \"] \r\n  \"1000117\" -> \"1000118\"  [ label = \"AST: \"] \r\n  \"1000117\" -> \"1000119\"  [ label = \"AST: \"] \r\n  \"1000122\" -> \"1000123\"  [ label = \"AST: \"] \r\n  \"1000122\" -> \"1000130\"  [ label = \"AST: \"] \r\n  \"1000122\" -> \"1000156\"  [ label = \"AST: \"] \r\n  \"1000123\" -> \"1000124\"  [ label = \"AST: \"] \r\n  \"1000123\" -> \"1000128\"  [ label = \"AST: \"] \r\n  \"1000123\" -> \"1000129\"  [ label = \"AST: \"] \r\n  \"1000124\" -> \"1000125\"  [ label = \"AST: \"] \r\n  \"1000125\" -> \"1000126\"  [ label = \"AST: \"] \r\n  \"1000125\" -> \"1000127\"  [ label = \"AST: \"] \r\n  \"1000130\" -> \"1000131\"  [ label = \"AST: \"] \r\n  \"1000130\" -> \"1000134\"  [ label = \"AST: \"] \r\n  \"1000130\" -> \"1000147\"  [ label = \"AST: \"] \r\n  \"1000131\" -> \"1000132\"  [ label = \"AST: \"] \r\n  \"1000131\" -> \"1000133\"  [ label = \"AST: \"] \r\n  \"1000134\" -> \"1000135\"  [ label = \"AST: \"] \r\n  \"1000135\" -> \"1000136\"  [ label = \"AST: \"] \r\n  \"1000135\" -> \"1000142\"  [ label = \"AST: \"] \r\n  \"1000136\" -> \"1000137\"  [ label = \"AST: \"] \r\n  \"1000137\" -> \"1000138\"  [ label = \"AST: \"] \r\n  \"1000138\" -> \"1000139\"  [ label = \"AST: \"] \r\n  \"1000139\" -> \"1000140\"  [ label = \"AST: \"] \r\n  \"1000139\" -> \"1000141\"  [ label = \"AST: \"] \r\n  \"1000142\" -> \"1000143\"  [ label = \"AST: \"] \r\n  \"1000143\" -> \"1000144\"  [ label = \"AST: \"] \r\n  \"1000144\" -> \"1000145\"  [ label = \"AST: \"] \r\n  \"1000144\" -> \"1000146\"  [ label = \"AST: \"] \r\n  \"1000147\" -> \"1000148\"  [ label = \"AST: \"] \r\n  \"1000148\" -> \"1000149\"  [ label = \"AST: \"] \r\n  \"1000148\" -> \"1000154\"  [ label = \"AST: \"] \r\n  \"1000149\" -> \"1000150\"  [ label = \"AST: \"] \r\n  \"1000149\" -> \"1000153\"  [ label = \"AST: \"] \r\n  \"1000150\" -> \"1000151\"  [ label = \"AST: \"] \r\n  \"1000150\" -> \"1000152\"  [ label = \"AST: \"] \r\n  \"1000154\" -> \"1000155\"  [ label = \"AST: \"] \r\n  \"1000156\" -> \"1000157\"  [ label = \"AST: \"] \r\n  \"1000157\" -> \"1000158\"  [ label = \"AST: \"] \r\n  \"1000158\" -> \"1000159\"  [ label = \"AST: \"] \r\n  \"1000158\" -> \"1000160\"  [ label = \"AST: \"] \r\n  \"1000105\" -> \"1000111\"  [ label = \"CFG: \"] \r\n  \"1000109\" -> \"1000119\"  [ label = \"CFG: \"] \r\n  \"1000111\" -> \"1000109\"  [ label = \"CFG: \"] \r\n  \"1000114\" -> \"1000161\"  [ label = \"CFG: \"] \r\n  \"1000114\" -> \"1000127\"  [ label = \"CFG: \"] \r\n  \"1000115\" -> \"1000114\"  [ label = \"CFG: \"] \r\n  \"1000116\" -> \"1000115\"  [ label = \"CFG: \"] \r\n  \"1000117\" -> \"1000116\"  [ label = \"CFG: \"] \r\n  \"1000119\" -> \"1000117\"  [ label = \"CFG: \"] \r\n  \"1000123\" -> \"1000133\"  [ label = \"CFG: \"] \r\n  \"1000124\" -> \"1000123\"  [ label = \"CFG: \"] \r\n  \"1000125\" -> \"1000124\"  [ label = \"CFG: \"] \r\n  \"1000127\" -> \"1000125\"  [ label = \"CFG: \"] \r\n  \"1000131\" -> \"1000141\"  [ label = \"CFG: \"] \r\n  \"1000131\" -> \"1000152\"  [ label = \"CFG: \"] \r\n  \"1000133\" -> \"1000131\"  [ label = \"CFG: \"] \r\n  \"1000136\" -> \"1000146\"  [ label = \"CFG: \"] \r\n  \"1000136\" -> \"1000160\"  [ label = \"CFG: \"] \r\n  \"1000137\" -> \"1000136\"  [ label = \"CFG: \"] \r\n  \"1000138\" -> \"1000137\"  [ label = \"CFG: \"] \r\n  \"1000139\" -> \"1000138\"  [ label = \"CFG: \"] \r\n  \"1000141\" -> \"1000139\"  [ label = \"CFG: \"] \r\n  \"1000142\" -> \"1000160\"  [ label = \"CFG: \"] \r\n  \"1000143\" -> \"1000142\"  [ label = \"CFG: \"] \r\n  \"1000144\" -> \"1000143\"  [ label = \"CFG: \"] \r\n  \"1000146\" -> \"1000144\"  [ label = \"CFG: \"] \r\n  \"1000149\" -> \"1000154\"  [ label = \"CFG: \"] \r\n  \"1000149\" -> \"1000160\"  [ label = \"CFG: \"] \r\n  \"1000150\" -> \"1000153\"  [ label = \"CFG: \"] \r\n  \"1000152\" -> \"1000150\"  [ label = \"CFG: \"] \r\n  \"1000153\" -> \"1000149\"  [ label = \"CFG: \"] \r\n  \"1000154\" -> \"1000160\"  [ label = \"CFG: \"] \r\n  \"1000156\" -> \"1000161\"  [ label = \"CFG: \"] \r\n  \"1000157\" -> \"1000156\"  [ label = \"CFG: \"] \r\n  \"1000158\" -> \"1000157\"  [ label = \"CFG: \"] \r\n  \"1000160\" -> \"1000158\"  [ label = \"CFG: \"] \r\n  \"1000100\" -> \"1000105\"  [ label = \"CFG: \"] \r\n  \"1000142\" -> \"1000161\"  [ label = \"DDG: sem_post(&s->refsync)\"] \r\n  \"1000105\" -> \"1000161\"  [ label = \"DDG: opaque\"] \r\n  \"1000156\" -> \"1000161\"  [ label = \"DDG: &avbuf->context_ref\"] \r\n  \"1000114\" -> \"1000161\"  [ label = \"DDG: atomic_fetch_sub(&avbuf->context_refcount, 1) == 1\"] \r\n  \"1000123\" -> \"1000161\"  [ label = \"DDG: atomic_fetch_sub_explicit(&s->refcount, 1, memory_order_acq_rel)\"] \r\n  \"1000114\" -> \"1000161\"  [ label = \"DDG: atomic_fetch_sub(&avbuf->context_refcount, 1)\"] \r\n  \"1000154\" -> \"1000161\"  [ label = \"DDG: ff_v4l2_buffer_enqueue(avbuf)\"] \r\n  \"1000123\" -> \"1000161\"  [ label = \"DDG: &s->refcount\"] \r\n  \"1000123\" -> \"1000161\"  [ label = \"DDG: memory_order_acq_rel\"] \r\n  \"1000154\" -> \"1000161\"  [ label = \"DDG: avbuf\"] \r\n  \"1000102\" -> \"1000161\"  [ label = \"DDG: unused\"] \r\n  \"1000111\" -> \"1000161\"  [ label = \"DDG: avbuf\"] \r\n  \"1000101\" -> \"1000161\"  [ label = \"DDG: opaque\"] \r\n  \"1000109\" -> \"1000161\"  [ label = \"DDG: s\"] \r\n  \"1000142\" -> \"1000161\"  [ label = \"DDG: &s->refsync\"] \r\n  \"1000136\" -> \"1000161\"  [ label = \"DDG: !atomic_load(&s->refcount)\"] \r\n  \"1000156\" -> \"1000161\"  [ label = \"DDG: av_buffer_unref(&avbuf->context_ref)\"] \r\n  \"1000137\" -> \"1000161\"  [ label = \"DDG: &s->refcount\"] \r\n  \"1000109\" -> \"1000161\"  [ label = \"DDG: buf_to_m2mctx(avbuf)\"] \r\n  \"1000115\" -> \"1000161\"  [ label = \"DDG: &avbuf->context_refcount\"] \r\n  \"1000136\" -> \"1000161\"  [ label = \"DDG: atomic_load(&s->refcount)\"] \r\n  \"1000100\" -> \"1000101\"  [ label = \"DDG: \"] \r\n  \"1000100\" -> \"1000102\"  [ label = \"DDG: \"] \r\n  \"1000101\" -> \"1000105\"  [ label = \"DDG: opaque\"] \r\n  \"1000100\" -> \"1000105\"  [ label = \"DDG: \"] \r\n  \"1000111\" -> \"1000109\"  [ label = \"DDG: avbuf\"] \r\n  \"1000100\" -> \"1000109\"  [ label = \"DDG: \"] \r\n  \"1000105\" -> \"1000111\"  [ label = \"DDG: avbuf\"] \r\n  \"1000100\" -> \"1000111\"  [ label = \"DDG: \"] \r\n  \"1000115\" -> \"1000114\"  [ label = \"DDG: &avbuf->context_refcount\"] \r\n  \"1000115\" -> \"1000114\"  [ label = \"DDG: 1\"] \r\n  \"1000100\" -> \"1000115\"  [ label = \"DDG: \"] \r\n  \"1000100\" -> \"1000114\"  [ label = \"DDG: \"] \r\n  \"1000100\" -> \"1000123\"  [ label = \"DDG: \"] \r\n  \"1000137\" -> \"1000136\"  [ label = \"DDG: &s->refcount\"] \r\n  \"1000123\" -> \"1000137\"  [ label = \"DDG: &s->refcount\"] \r\n  \"1000111\" -> \"1000154\"  [ label = \"DDG: avbuf\"] \r\n  \"1000100\" -> \"1000154\"  [ label = \"DDG: \"] \r\n  \"1000114\" -> \"1000125\"  [ label = \"CDG: \"] \r\n  \"1000114\" -> \"1000131\"  [ label = \"CDG: \"] \r\n  \"1000114\" -> \"1000127\"  [ label = \"CDG: \"] \r\n  \"1000114\" -> \"1000158\"  [ label = \"CDG: \"] \r\n  \"1000114\" -> \"1000156\"  [ label = \"CDG: \"] \r\n  \"1000114\" -> \"1000123\"  [ label = \"CDG: \"] \r\n  \"1000114\" -> \"1000124\"  [ label = \"CDG: \"] \r\n  \"1000114\" -> \"1000160\"  [ label = \"CDG: \"] \r\n  \"1000114\" -> \"1000133\"  [ label = \"CDG: \"] \r\n  \"1000114\" -> \"1000157\"  [ label = \"CDG: \"] \r\n  \"1000131\" -> \"1000153\"  [ label = \"CDG: \"] \r\n  \"1000131\" -> \"1000137\"  [ label = \"CDG: \"] \r\n  \"1000131\" -> \"1000141\"  [ label = \"CDG: \"] \r\n  \"1000131\" -> \"1000152\"  [ label = \"CDG: \"] \r\n  \"1000131\" -> \"1000150\"  [ label = \"CDG: \"] \r\n  \"1000131\" -> \"1000139\"  [ label = \"CDG: \"] \r\n  \"1000131\" -> \"1000136\"  [ label = \"CDG: \"] \r\n  \"1000131\" -> \"1000149\"  [ label = \"CDG: \"] \r\n  \"1000131\" -> \"1000138\"  [ label = \"CDG: \"] \r\n  \"1000136\" -> \"1000143\"  [ label = \"CDG: \"] \r\n  \"1000136\" -> \"1000142\"  [ label = \"CDG: \"] \r\n  \"1000136\" -> \"1000146\"  [ label = \"CDG: \"] \r\n  \"1000136\" -> \"1000144\"  [ label = \"CDG: \"] \r\n  \"1000149\" -> \"1000154\"  [ label = \"CDG: \"] \r\n}\r\n'
    #
    # print(_code)
