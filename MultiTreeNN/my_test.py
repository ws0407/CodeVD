import csv
import os

from torch.utils.data import DataLoader

from IVDetect.main import MyDatset, collate_batch

test_path = '/data/data/ws/CodeVD/IVDetect/data/Fan_et_al/test_data/'
test_files = [f for f in os.listdir(test_path) if
              os.path.isfile(os.path.join(test_path, f))]
test_dataset = MyDatset(test_files, test_path)

test_loader = DataLoader(test_dataset, collate_fn=collate_batch, shuffle=False)

print('len:', len(test_loader))
