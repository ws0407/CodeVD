import csv
import os
from torch.utils.data import DataLoader
import pickle
import pandas as pd
import re


def count_dataset(filepath="/data/data/ws/CodeVD/MultiTreeNN/data/raw/dataset.json"):
    data = pd.read_json(filepath)
    print(data)
    print(data.columns)
    print(data["target"].value_counts())


count_dataset()



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
