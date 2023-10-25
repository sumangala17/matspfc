import pickle
import random

import numpy as np
import pandas as pd

from run_example_cbss import call_CBSS_c, Results

dataset_names = ["ht_chantry", "maze-32-32-2", "room-32-32-4", "den312d", "empty-16-16"]
map_size_x = [141, 32, 32, 81, 16]
map_size_y = [162, 32, 32, 65, 16]

# N = [5, 10, 20]                 # num_agents
# M = [10, 15, 25, 45, 70]        # num_targets
# K = [2, 5, 10]                  # num_clusters

#results mini
N = [3, 6, 12]                 # num_agents
M = [5, 10, 35]        # num_targets
K = [3, 8, 15]                  # num_clusters

results_path_parent = "/home/biorobotics/matspfc/results_mini/"

class Unit:
    def __init__(self, cbss_c: Results, hr: Results):
        self.res_cbss_c = cbss_c
        self.res_hr: Results = hr


def create_problem_instances(dataset_name, map_size, num_instances=1):

    results_path = results_path_parent + dataset_name + "/"

    scen_folder = f'/home/biorobotics/matspfc/datasets/{dataset_name}.map-scen-random/scen-random/'

    grid_file = f'/home/biorobotics/matspfc/datasets/{dataset_name}.map'
    grid_file_new = f'/home/biorobotics/matspfc/datasets/{dataset_name}_binary.map'

    with open(grid_file) as file:
        for _ in range(4):
            next(file)
        newText = file.read().replace('@', '1 ')
        newText = newText.replace('.', '0 ')
        newText = newText.replace('T', '0 ')

    with open(grid_file_new, 'w') as file:
        file.write(newText)

    grids = np.loadtxt(grid_file_new)

    for i in range(1, num_instances + 1):
        assert i <= 25
        scen_file = f'{dataset_name}-random-{i}.scen'
        path_to_dataset = scen_folder + scen_file

        with open(path_to_dataset) as f:
            # next(f)
            df = pd.read_csv(path_to_dataset, delimiter='\t', header=None, index_col=False, skiprows=1)

        max_rows = df.shape[0]

        for num_agents in N:
            for num_targets in M:
                for num_clusters in K:

                    # one entry in our results table (n agents, m targets, k clusters)

                    if num_targets <= num_clusters:
                        continue

                    selected_loc = random.sample(range(max_rows), num_agents + num_targets)
                    # selected_loc = range(30, 30+num_agents + num_targets)
                    # random.shuffle(selected_loc)

                    start_x = np.array(df.iloc[selected_loc, 4])
                    start_y = np.array(df.iloc[selected_loc, 5])
                    goal_x = np.array(df.iloc[selected_loc, 6])
                    goal_y = np.array(df.iloc[selected_loc, 7])

                    starts, targets, dests = [], [], []
                    for p in range(num_agents):
                        starts.append(map_size * start_y[p] + start_x[p])
                        dests.append(map_size * goal_y[p] + goal_x[p])

                    for p in range(num_agents, num_agents + num_targets):
                        targets.append(map_size * start_y[p] + start_x[p])

                    clusters = np.random.randint(0, high=num_clusters, size=num_targets)
                    while not set(np.unique(clusters)) == set(range(num_clusters)):
                        clusters = np.random.randint(0, high=num_clusters, size=num_targets)
                    # clusters = np.arange(num_targets)

                    ac_dict = dict()
                    ri = 0
                    for k in targets:
                        ac_dict[k] = set([ri, ri + 1])
                        ri += 1
                        if ri >= len(starts) - 1:
                            break
                    ri = 0
                    for k in dests:
                        ac_dict[k] = set([ri])
                        ri += 1


                    try:
                    # print("_________________________NEW RUN______________________________")
                        res_cbss, res_hr = call_CBSS_c(starts, dests, targets, ac_dict, grids, clusters, map_size)
                        fpath = results_path + f"numpyfiles/{dataset_name}_N{num_agents}_M{num_targets}_K{num_clusters}"
                        with open(fpath + f"_h0_run{i}.npy", 'wb') as f:
                            pickle.dump(res_cbss, f, pickle.HIGHEST_PROTOCOL)
                        with open(fpath + f"_h1_run{i}.npy", 'wb') as f:
                            pickle.dump(res_hr, f, pickle.HIGHEST_PROTOCOL)

                        unit = Unit(res_cbss, res_hr)
                        with open(fpath + f"_h01_run{i}.npy", 'wb') as f:
                            pickle.dump(unit, f, pickle.HIGHEST_PROTOCOL)

                        with open(results_path + f"{dataset_name}_N{num_agents}_M{num_targets}_K{num_clusters}_h0.txt", "w") as f:
                            for line in str(res_cbss.__dict__):
                                f.write(line)
                        with open(results_path + f"{dataset_name}_N{num_agents}_M{num_targets}_K{num_clusters}_h1.txt", "w") as f:
                            for line in str(res_hr.__dict__):
                                f.write(line)

                    except Exception as e:
                        print(f"We caught an exception with {num_agents} agents, {num_targets} targets and {num_clusters} clusters!", e, "\nMoving on..")






        # print(df)

if __name__ == '__main__':
    for d in range(len(dataset_names)):
        map_size = max(map_size_x[d], map_size_y[d])
        create_problem_instances(dataset_names[d], map_size, 1)
