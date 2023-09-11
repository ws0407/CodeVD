import csv
import json
import os
from torch.utils.data import DataLoader
import pickle
import pandas as pd
import re

from tqdm import tqdm

keywords = ['__asm', '__builtin', '__cdecl', '__declspec', '__except', '__export', '__far16', '__far32',
            '__fastcall', '__finally', '__import', '__inline', '__int16', '__int32', '__int64', '__int8',
            '__leave', '__optlink', '__packed', '__pascal', '__stdcall', '__system', '__thread', '__try',
            '__unaligned', '_asm', '_Builtin', '_Cdecl', '_declspec', '_except', '_Export', '_Far16',
            '_Far32', '_Fastcall', '_finally', '_Import', '_inline', '_int16', '_int32', '_int64',
            '_int8', '_leave', '_Optlink', '_Packed', '_Pascal', '_stdcall', '_System', '_try', 'alignas',
            'alignof', 'and', 'and_eq', 'asm', 'auto', 'bitand', 'bitor', 'bool', 'break', 'case',
            'catch', 'char', 'char16_t', 'char32_t', 'class', 'compl', 'const', 'const_cast', 'constexpr',
            'continue', 'decltype', 'default', 'delete', 'do', 'double', 'dynamic_cast', 'else', 'enum',
            'explicit', 'export', 'extern', 'false', 'final', 'float', 'for', 'friend', 'goto', 'if',
            'inline', 'int', 'long', 'mutable', 'namespace', 'new', 'noexcept', 'not', 'not_eq', 'nullptr',
            'operator', 'or', 'or_eq', 'override', 'private', 'protected', 'public', 'register',
            'reinterpret_cast', 'return', 'short', 'signed', 'sizeof', 'static', 'static_assert',
            'static_cast', 'struct', 'switch', 'template', 'this', 'thread_local', 'throw', 'true', 'try',
            'typedef', 'typeid', 'typename', 'union', 'unsigned', 'using', 'virtual', 'void', 'volatile',
            'wchar_t', 'while', 'xor', 'xor_eq', 'NULL',
            # others
            'int8_t', 'int16_t', 'int32_t', 'int64_t', 'uint8_t', 'uint16_t', 'uint32_t', 'uint64_t',
            'BOOL', 'DWORD',
            # FFmpeg
            'av_always_inline', 'always_inline', 'av_cold', 'cold', 'av_extern_inline', 'av_warn_unused_result',
            'av_noinline', 'av_pure', 'av_const', 'av_flatten', 'av_unused', 'av_used', 'av_alias',
            'av_uninit', 'av_builtin_constant_p', 'av_builtin_constant_p', 'av_builtin_constant_p',
            'av_noreturn', 'CUDAAPI', 'attribute_align_arg',
            # qemu
            'coroutine_fn', 'always_inline', 'WINAPI', 'QEMU_WARN_UNUSED_RESULT', 'QEMU_NORETURN', 'CALLBACK']


def print_func(func: str = ""):
    func = func.replace('\n\n', '\n')
    func = re.sub(r'(/\*([^*]|(\*+[^*\/]))*\*+\/)|(\/\/.*)', '', func)
    func = re.sub(r'["]([^"\\\n]|\\.|\\\n)*["]', '""', func)
    func = re.sub(r'[\']([^\'\\\n]|\\.|\\\n)*[\']', '\'\'', func)
    print(func)


def count_dataset(filepath="/data/data/ws/CodeVD/MultiTreeNN/data/raw/dataset.json"):
    data = pd.read_json(filepath)
    print(data)
    print(data.columns)
    print(data["target"].value_counts())


def out_dataset_errors(
        input_json="/data/data/ws/CodeVD/IVDetect/data/FFMPeg_Qemu/function.json",
        key="func",
):
    """
    标准标识符：[800, 814, 1134, 13]
    err1: 函数无返回类型，joern解析没问题
    err2：代码括号不匹配，爬取代码不完整! target=0有43个, =1有757个, FFmpeg=355, qemu=455
    err3 & 4：需引入宏文件，加上宏文件FFmpeg和Qemu后可清除后两种解析bug
    """
    with open(input_json, 'r') as f:
        data = json.load(f)
    all_type = [0, 0, 0, 0]
    for idx, sample in tqdm(enumerate(data)):
        # if sample["project"] != "FFmpeg":
        #     continue
        func: str = sample[key]
        while '\n\n' in func:
            func = func.replace('\n\n', '\n')
        func = re.sub(r'(/\*([^*]|(\*+[^*\/]))*\*+\/)|(\/\/.*)', '', func)  # 去除/**/ //--注释
        func = re.sub(r'"([^"\\\n]|\\.|\\\n)*"', '""', func)                # 去除""包的字符串
        func = re.sub(r'\'([^\'\\\n]|\\.|\\\n)*\'', '\'\'', func)           # 去除''包的字符
        func_head = func[:func.find("(")]
        if func.count('(') != func.count(')') or func.count('{') != func.count('}'):
            print('{} brackets not match {}'.format('*' * 10, '*' * 10))
            print('[{}][target={}] {}'.format(sample['project'], sample['target'], func_head))
            all_type[0] += 1
            continue
        func_token = func_head.split()
        if len(func_token) <= 1:
            # print('{} len(func_token) <= 1 {}:'.format('*' * 10, '*' * 10))
            # print('[{}] {}'.format(sample['project'], func_head))
            all_type[1] += 1
            continue
        if len(func_token) == 4:
            if 'inline' in func_token or 'enum' in func_token or 'struct' in func_token or 'const' in func_token \
                    or '*' in func_token or 'void*' in func_token:
                continue
            if func_token[0] not in keywords or func_token[1] not in keywords or func_token[2] not in keywords:
                # print('{} len(func_token) == 4 {}:'.format('*' * 10, '*' * 10))
                # print('[{}] {}'.format(sample['project'], func_head))
                all_type[2] += 1
                continue
        if len(func_token) > 4:
            if 'inline' in func_token or 'enum' in func_token or 'struct' in func_token or 'const' in func_token \
                    or '*' in func_token:
                continue
            if func_token[0] not in keywords or func_token[1] not in keywords or \
                    func_token[2] not in keywords or func_token[3] not in keywords:
                # print('{} len(func_token) > 4 {}:'.format('*' * 10, '*' * 10))
                # print('[{}] {}'.format(sample['project'], func_head))
                all_type[3] += 1
                continue
        # func_name = func[:func.find("(")].split()[-1]
        # func_type = func.split()[0]
    print(len(data))
    print(all_type)


out_dataset_errors()
#
# count_dataset()
#
#
# def format_id(string):
#     string = re.sub(r'io\.shiftleft\.codepropertygraph\.generated\.', '', string)
#     string = re.sub(r'(\[label)(\D+)(\d+)\]', lambda match: f'@{match.group(3)}', string)
#     # string = re.sub(r'\d+', lambda match: f'@{match.group()}', string)
#     return string
#
#
# # 测试示例
# print(format_id("io.shiftleft.codepropertygraph.generated.edges.Ast@2b423"))
# print(format_id("\"io.shiftleft.codepropertygraph.generated.edges.Ast@2b062\","))
# s = """"function" : "ff_af_queue_init",
#       "id" : "io.shiftleft.codepropertygraph.generated.nodes.Method[label=METHOD; id=2305843009213694033]",
#       "AST" : [
#         {
#           "id" : "io.shiftleft.codepropertygraph.generated.nodes.MethodParameterIn[label=METHOD_PARAMETER_IN; id=2305843009213694034]",
#           "edges" : [
#             {
#               "id" : "io.shiftleft.codepropertygraph.generated.edges.Ast@2b062",
#               "in" : "io.shiftleft.codepropertygraph.generated.nodes.MethodParameterIn[label=METHOD_PARAMETER_IN; id=2305843009213694034]",
#               "out" : "io.shiftleft.codepropertygraph.generated.nodes.Method[label=METHOD; id=2305843009213694033]"
#             }
#           ],
#           "properties" : [
#             {
#               "key" : "ORDER",
#               "value" : "1"
#             },
#             {
#               "key" : "CODE",
#               "value" : "AVCodecContext *avctx"
#             }
#           ]
#         },"""
# print(s)
# print(format_id(s))

# from IVDetect.main import MyDatset, collate_batch
#
# test_path = '/data/data/ws/CodeVD/IVDetect/data/Fan_et_al/test_data/'
# test_files = [f for f in os.listdir(test_path) if
#               os.path.isfile(os.path.join(test_path, f))]
# test_dataset = MyDatset(test_files, test_path)
#
# test_loader = DataLoader(test_dataset, collate_fn=collate_batch, shuffle=False)
#
# print('len:', len(test_loader))

# import pickle
# import json
#
# with open('data/cpg/0_cpg.pkl', 'rb') as f:
#     data: pd.DataFrame = pickle.load(f)
#
# print(data.columns)
# print(data.head(2))
#
# with open('data/cpg/test.json', 'w', encoding="utf-8") as f:
#     json.dump(data.head(10).to_json(), f, ensure_ascii=False)


# print(cpg['cpg'].iloc[6])
# print(len(cpg))


# import pickle
#
# from utils.functions import parse_to_nodes
#
# with open('/data/data/ws/CodeVD/MultiTreeNN/data/cpg/0_cpg.pkl', 'rb') as f:
#     data: pd.DataFrame = pickle.load(f)
# # data["nodes"] = data.apply(lambda row: parse_to_nodes(row.cpg, 205), axis=1)
# # print()
# for funcs in data["cpg"]:
#     for k, v in funcs.items():
#         # print(len(v))
#         if 'ASN1_TYPE_set' in str(v[0]):
#             print(k, v[0])
#             break
