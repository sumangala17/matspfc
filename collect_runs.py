import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_example_cbss import call_CBSS_c
from util_data_structs import Unit

dataset_names = ["random-32-32-10", "maze-32-32-2", "den312d", "room-32-32-4",  "empty-16-16"]  # den312d, ht_chantry
map_size_x = [32, 32, 16]  # 141, 81
map_size_y = [32, 32, 16]  # 162, 65


results_path_parent = "/home/biorobotics/matspfc/res_feb/"


def create_problem_instances(dataset_name, num_agents, num_targets, num_clusters, num_instances=1):

    results_path = results_path_parent + dataset_name + "/"

    scen_folder = f'/home/biorobotics/matspfc/datasets/{dataset_name}.map-scen-random/scen-random/'

    grid_file = f'/home/biorobotics/matspfc/datasets/{dataset_name}.map'
    grid_file_new = f'/home/biorobotics/matspfc/datasets/{dataset_name}_binary.map'

    with open(grid_file) as file:
        for _ in range(4):
            next(file)
        newText = file.read().replace('@', '1 ')
        newText = newText.replace('.', '0 ')
        newText = newText.replace('T', '1 ')

    with open(grid_file_new, 'w') as file:
        file.write(newText)

    grids = np.loadtxt(grid_file_new)

    sz = grids.shape[1]

    cost_list1 = []
    ntsp_list1 = []
    conf_list1 = []
    node_list1 = []

    cost_list0 = []
    ntsp_list0 = []
    conf_list0 = []
    node_list0 = []

    cost_list3 = []
    ntsp_list3 = []
    conf_list3 = []
    node_list3 = []

    cost_list5 = []
    ntsp_list5 = []
    conf_list5 = []
    node_list5 = []

    cost_list4 = []
    ntsp_list4 = []
    conf_list4 = []
    node_list4 = []

    # cost_list1 = []
    # cost_list2 = []
    # ntsp_list1 = []
    # ntsp_list2 = []

    # conf_list1 = []
    # conf_list2 = []
    # node_list1 = []
    # node_list2 = []

    # cost_list3 = []
    # cost_list4 = []
    # ntsp_list3 = []
    # ntsp_list4 = []

    stats = np.zeros((1,20))

    print(f"_________________________N, M, K = {num_agents}, {num_targets, num_clusters}_____________________________")

    i = 1
    pass_c, fail_c, pass_ch, fail_ch = 0,0,0,0
    while i < num_instances + 1:
        # print("INSTANCE", i)
        id = i % 25 + 1
        scen_file = f'{dataset_name}-random-{id}.scen'
        path_to_dataset = scen_folder + scen_file

        with open(path_to_dataset) as f:
            # next(f)
            df = pd.read_csv(path_to_dataset, delimiter='\t', header=None, index_col=False, skiprows=1)

        max_rows = df.shape[0]

        selected_loc = random.sample(range(max_rows), num_agents + num_targets)
        start_x = np.array(df.iloc[selected_loc, 4])
        start_y = np.array(df.iloc[selected_loc, 5])
        goal_x = np.array(df.iloc[selected_loc, 6])
        goal_y = np.array(df.iloc[selected_loc, 7])

        starts, targets, dests = [], [], []
        for p in range(num_agents):
            starts.append(sz * start_y[p] + start_x[p])
            dests.append(sz * goal_y[p] + goal_x[p])
        for p in range(num_agents, num_agents + num_targets):
            targets.append(sz * start_y[p] + start_x[p])

        clusters = np.random.randint(0, high=num_clusters, size=num_targets)
        idx = 0
        while not set(np.unique(clusters)) == set(range(num_clusters)):
            clusters = np.random.randint(0, high=num_clusters, size=num_targets)
            idx += 1
            if idx > 200:
                clusters = np.arange(num_targets) % num_clusters
                np.random.shuffle(clusters)
                break

        ac_dict = dict()
        ri = 0
        for k in targets:
            ac_dict[k] = {ri, ri + 1}
            ri += 1
            if ri >= len(starts) - 1:
                break
        ri = 0
        for k in dests:
            ac_dict[k] = {ri}
            ri += 1

        txt_file = results_path + f"{dataset_name}_N{num_agents}_M{num_targets}_K{num_clusters}_{i}"


        try:
            arr = call_CBSS_c(starts, dests, targets, ac_dict, grids, clusters, sz, txt_file)
            # (c1, t1, cn1, tn1, ch0, th0, chn0, thn0, ch3, th3, chn3, thn3, ch4, th4, chn4, thn4, ch5,
            #  th5, chn5, thn5) = call_CBSS_c(starts, dests, targets, ac_dict, grids, clusters, sz, txt_file)
            # success = call_CBSS_c(starts, dests, targets, ac_dict, grids, clusters, sz, txt_file)
            # c1,c2,t1,t2 = call_CBSS_c(starts, dests, targets, ac_dict, grids, clusters, sz, txt_file)
            # c1,c2,t1,t2, cn1,cn2,n1,n2 = call_CBSS_c(starts, dests, targets, ac_dict, grids, clusters, sz, txt_file)
            # t1, t2, t3, t4, c1, c2, c3, c4 = call_CBSS_c(starts, dests, targets, ac_dict, grids, clusters, sz, txt_file)
        except:
            # print("we should not be here")
            continue
        else:
            i += 1
            if i%10 == 0:
                print(f"we have collected {i} runs")

            stats += arr

            # pass_c += success[0]
            # fail_c += success[1]
            # pass_ch += success[2]
            # fail_ch += success[3]
            # cost_list1.append(c1)
            # cost_list2.append(c2)
            # cost_list3.append(c3)
            # cost_list4.append(c4)
            #
            # ntsp_list1.append(t1)
            # ntsp_list2.append(t2)
            # ntsp_list3.append(t3)
            # ntsp_list4.append(t4)

            # cost_list1.append(c1)
            # ntsp_list1.append()
        #
        #     conf_list1.append(cn1)
        #     conf_list2.append(cn2)
        #     node_list1.append(n1)
        #     node_list2.append(n2)
    # return pass_c, fail_c, pass_ch, fail_ch

    # n = len(cost_list1)
    # assert n == num_instances
    # cost1 = sum(cost_list1)/n
    # cost2 = sum(cost_list2)/n
    # cost3 = sum(cost_list3) / n
    # cost4 = sum(cost_list4) / n
    # ntsp1 = sum(ntsp_list1)/n
    # ntsp2 = sum(ntsp_list2)/n
    # ntsp3 = sum(ntsp_list3) / n
    # ntsp4 = sum(ntsp_list4) / n
    # conf1 = sum(conf_list1)/n
    # conf2 = sum(conf_list2)/n
    #
    # node1 = sum(node_list1)/n
    # node2 = sum(node_list2)/n

    stats = stats / num_instances

    print(stats.reshape(5,4))

    # print(cost_list1)
    # print(cost_list2)
    # print(ntsp_list1)
    # print(ntsp_list2)
    # print(ntsp_list3)
    # print(ntsp_list4)

    # return ntsp1, ntsp2, ntsp3, ntsp4, cost1, cost2, cost3, cost4
    return stats
    # return cost1, cost2, ntsp1, ntsp2, conf1, conf2, node1, node2



if __name__ == '__main__':
    # with open('res_feb/excellent.txt', 'w') as f1:
    #     f1.write('\n\nResults\n\n')
    for d in [3]:# range(4, len(dataset_names)):
        print(f'=============={dataset_names[d]}==============')
        # bar_c1, bar_c2 = [], []
        # bar_t1, bar_t2 = [], []
        instances = [(2,25,25)]
        # instances = [(2,40,15)]  #(2,10,10),(3,15,15),(5,20,20),(7,35,35),

        # with open('res_feb/excellentscale.txt', 'a') as f1:
        #     f1.write('\n\n' + dataset_names[d] + '\n\n')

        for (N,M,K) in instances:
            # c1,c2,t1,t2,cn1,cn2,n1,n2 = create_problem_instances(dataset_names[d], N,M,K, 50)
            # pass_c, fail_c, pass_ch, fail_ch = create_problem_instances(dataset_names[d], N,M,K, 50)
            # t1, t2, t3, t4, c1, c2, c3, c4 = create_problem_instances(dataset_names[d], N,M,K, 50)
            # print(f'\n\ncost={c1, c2, c3, c4}, time={t1, t2, t3, t4}\n')
            # print(f'\n\n{N,M,K}:\tcost={c1, c2}, time={t1, t2}\n')
            stats = create_problem_instances(dataset_names[d], N,M,K, 50)
            # (c1, t1, cn1, tn1, ch0, th0, chn0, thn0, ch3, th3, chn3, thn3, ch4, th4, chn4, thn4, ch5,
            #  th5, chn5, thn5) = create_problem_instances(dataset_names[d], N,M,K, 50)

            with open('res_feb/heuristic_analysis.txt', 'a') as f1:
                f1.write(f'\n{(N,M,K)}\n\n')
                for row in stats:
                    for num in row:
                        # f1.write(" ".join(map(str, num)))
                        f1.write(f'{num}\n')
                f1.write('\n\n')
                # f1.write(f'{c1}\n{c2}\n{t1}\n{t2}\n\n')
                # f1.write(f'{c1}\n{c2}\n{t1}\n{t2}\n{cn1}\n{cn2}\n{n1}\n{n2}\n\n')
                # f1.write(f'{c1}\n{c2}\n{c3}\n{c4}\n{t1}\n{t2}\n{t3}\n{t4}\n\n')
            # bar_c1.append(c1)
            # bar_c2.append(c2)
            # bar_t1.append(t1)
            # bar_t2.append(t2)
        #
        # df = pd.DataFrame({'c1': bar_c1, 'c2': bar_c2})
        # fig, ax = plt.subplots()
        # index = np.arange(5)
        # bar_width = 0.3
        # cost_c = ax.bar(index, df["c1"], bar_width, label="Cost CBSS-c")
        # cost_ch = ax.bar(index + bar_width, df["c2"], bar_width, label="Cost CBSS-ch")
        #
        # ax.set_xlabel('Instance')
        # ax.set_ylabel('Cost')
        # ax.set_xticks(index + bar_width / 2)
        # ax.set_xticklabels(instances)
        # ax.legend()

        plt.show()

        # dft = pd.DataFrame({'t1': bar_t1, 't2': bar_t2})
        # fig, ax = plt.subplots()
        # index = np.arange(5)
        # bar_width = 0.35
        # time_c = ax.bar(index, dft["t1"], bar_width,log=True, label="ntsp CBSS-c")
        # time_ch = ax.bar(index + bar_width, dft["t2"], bar_width,log=True, label="ntsp CBSS-ch")
        #
        # ax.set_xlabel('Instance')
        # ax.set_ylabel('nTSP Time')
        # # ax.set_title('Crime incidence by season, type')
        # ax.set_xticks(index + bar_width / 2)
        # ax.set_xticklabels(instances)
        # ax.legend()
        #
        # plt.show()
