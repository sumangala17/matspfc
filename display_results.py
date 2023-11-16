import json
import os
import pickle

import numpy as np
import pandas as pd
from pretty_html_table import build_table

from util_data_structs import Unit

dataset_names = ["maze-32-32-2", "room-32-32-4", "empty-16-16"]

res_mat = {}
for d in dataset_names:
    res_mat[d] = {
        'N': [], 'M': [], 'K': [], 'Astar_time': [],
        'total_time_cbssc': [], 'total_time_hr': [], 'ntsp_cbssc': [], 'ntsp_hr': [],
        'cost_cbssc': [], 'cost_hr': [], 'last_ts_cbssc': [], 'last_ts_hr': [],
        'conflicts_cbssc': [], 'conflicts_hr': [],
        'cost up %': [], 'ntsp time down %': []
    }

def read_from_np(dname):

    file_path = "/home/biorobotics/matspfc/results_nov/" + dname + "/numpyfiles/"
    # for file in os.walk("/home/biorobotics/matspfc/results/" + dname + "/numpyfiles/"):
    for filename in os.listdir(file_path):
        if "01" not in filename:
            continue
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
        # res_mat[dname]['useH'].append(int(h))

        with open(file_path + filename1, 'rb') as f:
            unit: Unit = pickle.load(f)
        # print(res.print_stats())
        res_mat[dname]['Astar_time'].append(unit.res_hr.Astar_time)
        res_mat[dname]['total_time_cbssc'].append(unit.res_cbss_c.total_time)
        res_mat[dname]['total_time_hr'].append(unit.res_hr.total_time)
        res_mat[dname]['ntsp_cbssc'].append(unit.res_cbss_c.ntsp)
        res_mat[dname]['ntsp_hr'].append(unit.res_hr.ntsp)
        res_mat[dname]['cost_cbssc'].append(int(unit.res_cbss_c.cost))
        res_mat[dname]['cost_hr'].append(int(unit.res_hr.cost))
        res_mat[dname]['last_ts_cbssc'].append(int(unit.res_cbss_c.max_step))
        res_mat[dname]['last_ts_hr'].append(int(unit.res_hr.max_step))
        res_mat[dname]['conflicts_cbssc'].append(int(unit.res_cbss_c.num_conflicts))
        res_mat[dname]['conflicts_hr'].append(int(unit.res_hr.num_conflicts))

        res_mat[dname]['cost up %'].append((unit.res_hr.cost - unit.res_cbss_c.cost) * 100 / unit.res_cbss_c.cost)
        res_mat[dname]['ntsp time down %'].append((unit.res_cbss_c.ntsp - unit.res_hr.ntsp) * 100 / unit.res_cbss_c.ntsp)

    df = pd.DataFrame(res_mat[dname])#.groupby(['N','M','K','useH','total_time','ntsp','cost','last_ts'])
    # print(df["room-32-32-4"].columns)
    return df#[df['M']>10]


if __name__ == '__main__':

    with open('styled_table.html', 'w') as f:
        for dname in dataset_names:
            print('_'*190)
            print(' '*95, dname.upper(), ' '*95)
            print('_'*190)
            df = read_from_np(dname)
            print(df)
            img_path = f'/home/biorobotics/matspfc/{dname}.png'
            html_table_blue_light = build_table(df, 'blue_dark')
            # f.write(f"<html><font color = #00b300><h2>MAP: {dname}</h2></font></html>")
            f.write(f"<html><h2 style='color:DodgerBlue; text-align: center;'>MAP: {dname}</h2></html>")
            f.write('<center><img src="' + img_path + '" height="70" alt="Italian Trulli"></center>')
            f.write(html_table_blue_light)
            # with pd.ExcelWriter("/home/biorobotics/matspfc/results_mini2/res123.xlsx", mode='a', if_sheet_exists='replace') as writer:
            #     df.to_excel(writer, sheet_name=f"Sheet_{dname}")


