import csv
import os

from torch.utils.data import DataLoader
import pickle
import pandas as pd

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

import pickle
import json

with open('data/cpg/0_cpg.pkl', 'rb') as f:
    data: pd.DataFrame = pickle.load(f)

print(data.columns)
# print(data.head(2))

with open('data/cpg/test.json', 'w', encoding="utf-8") as f:
    json.dump(data.head(10).to_json(), f, ensure_ascii=False)


# print(cpg['cpg'].iloc[6])
# print(len(cpg))