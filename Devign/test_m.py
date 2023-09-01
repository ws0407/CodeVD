import pickle
import json

with open('data/cpg/0_cpg.pkl', 'rb') as f:
    cpg = pickle.load(f)

with open('data/cpg/test.json', 'w', encoding="utf-8") as f:
    json.dump(cpg['cpg'].iloc[6], f, ensure_ascii=False)


# print(cpg['cpg'].iloc[6])
# print(len(cpg))