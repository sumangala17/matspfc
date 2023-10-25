import json
import os
import pickle

import numpy as np
import pandas as pd

from run_example_cbss import Results

dataset_names = ["ht_chantry", "maze-32-32-2", "room-32-32-4", "den312d", "empty-16-16"]

res_mat = {}
for d in dataset_names:
    res_mat[d] = {
        'N': [], 'M': [], 'K': [],
        'useH': [], 'total_time': [], 'ntsp': [],
        'cost': [], 'last_ts': []
    }

def read_from_np(dname):

    file_path = "/home/biorobotics/matspfc/results/" + dname + "/numpyfiles/"
    # for file in os.walk("/home/biorobotics/matspfc/results/" + dname + "/numpyfiles/"):
    for filename in os.listdir(file_path):
        # filename = str(file).split('/')[-1]
        filename1 = filename
        filename = filename.replace(dname + "_N", "")
        filename = filename.replace('M', '')
        filename = filename.replace('K', '')
        n, m, k, h, _ = filename.split('.')[0].split('_')
        h = h.replace('h', '')
        res_mat[dname]['N'].append(int(n))
        res_mat[dname]['M'].append(int(m))
        res_mat[dname]['K'].append(int(k))
        res_mat[dname]['useH'].append(int(h))

        with open(file_path + filename1, 'rb') as f:
            res = pickle.load(f)
        # print(res.print_stats())
        res_mat[dname]['total_time'].append(res.total_time)
        res_mat[dname]['ntsp'].append(res.ntsp)
        res_mat[dname]['cost'].append(int(res.cost))
        res_mat[dname]['last_ts'].append(int(res.max_step))

    df = pd.DataFrame(res_mat["room-32-32-4"])#.groupby(['N','M','K','useH','total_time','ntsp','cost','last_ts'])
    print(df)
    # print(df["room-32-32-4"].columns)
    return res_mat["room-32-32-4"]

if __name__ == '__main__':
    # for dname in dataset_names:
    print(read_from_np(dataset_names[2]))


