import numpy as np
import pandas as pd
# import pandas
# from pandas import read_fwf

def get_world(num_agents, num_targets, num_clusters):

    scen_folder = '/home/biorobotics/matspfc/datasets/Berlin_1_256.map-scen-random/scen-random/'
    scen_file = 'Berlin_1_256-random-1.scen'

    path_to_dataset = scen_folder + scen_file

    with open(path_to_dataset) as f:
        # next(f)
        df = pd.read_csv(path_to_dataset, delimiter='\t', header=None, index_col=False, skiprows=1)

    # print(df)

    start_x = df.iloc[:,4]
    start_y = df.iloc[:,5]
    goal_x = df.iloc[:,6]
    goal_y = df.iloc[:,7]
    # print("hi",start_x,"y", start_y.shape)

    # First 10 are robot starts and dests
    sz = 256
    starts = []
    dests = []
    # num_agents = 20
    for i in range(num_agents):
        x, y = start_x[i], start_y[i]
        starts.append(sz * y + x)
        dests.append(sz * goal_y[i] + goal_x[i])

    print("Starts:", starts)
    print("Dests:", dests)

    # num_targets = 20
    # Next 20 are targets
    targets = []
    for i in range(50, num_targets // 2 + 50):
        targets.append(sz * start_y[i] + start_x[i])
        targets.append(sz * goal_y[i] + goal_x[i])
    if num_targets % 2:
        i = num_targets // 2 + 51
        targets.append(sz * start_y[i] + start_x[i])

    print("Targets:", targets)

    # grids = np.zeros((256, 256))
    grid_file = '/home/biorobotics/matspfc/datasets/Berlin_1_256.map'
    grid_file_new = '/home/biorobotics/matspfc/datasets/Berlin_1_256_binary.map'

    with open(grid_file) as file:
        for i in range(4):
            next(file)
        newText = file.read().replace('@', '1 ')
        newText = newText.replace('.', '0 ')
        newText = newText

    with open(grid_file_new, 'w') as file:
        file.write(newText)

    grids = np.loadtxt(grid_file_new)
    # print(grids, grids.shape)

    # clusters = np.array([2, 8, 8, 6, 6, 9, 6, 5, 5, 3, 4, 3, 1, 6, 3, 9, 3, 3, 4, 4, 8, 3, 2, 9, 6, 7, 3, 0, 0, 2]) # np.random.randint(0, high=10, size=num_targets)
    # clusters = [1, 2, 1, 1, 2, 1, 1, 2, 0, 2] # np.arange(num_targets)
    # clusters = [0,1,2, 0]
    # clusters = [1,0,2, 0]
    clusters = np.random.randint(0, high=num_clusters, size=num_targets)
    while not set(np.unique(clusters)) ==set(range(num_clusters)):
        clusters = np.random.randint(0, high=num_clusters, size=num_targets)
    clusters = np.arange(num_targets)
    print("Clusters:",clusters)

    return starts, dests, targets, grids, clusters