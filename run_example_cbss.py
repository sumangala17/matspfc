"""
Author: Zhongqiang (Richard) Ren
All Rights Reserved.
ABOUT: Entrypoint to the code.
Oeffentlich fuer: RSS22
"""
from math import inf

import context
import time
import numpy as np
import random
import libmcpf.cbss_mcpf as cbss_mcpf

import common as cm


def get_paths(res_path):
  agent_paths = {}
  for agent in res_path:
    agent_paths[agent] = [p for p in list(zip(res_path[agent][0], res_path[agent][1]))]
    # print("Agent {}'s path is ".format(agent), [p for p in list(zip(res_path[agent][0], res_path[agent][1]))], "at times",
    #       res_path[agent][2])
  return agent_paths


def test_targets_visited(path, ac_dict, targets, clusters, num_agents, sz):
  # num_clusters = len(np.unique(clusters))
  # visited = [False for _ in range(num_clusters)]
  visited = [False for _ in range(len(targets))]
  agent_path = get_paths(path)
  t = 0
  for i in range(len(targets)):
    target = targets[i]
    txy = (target % sz, int(target / sz))
    allowed_ag = ac_dict[target] if target in ac_dict else np.arange(num_agents)
    for ag in allowed_ag:
      if txy in agent_path[ag]:
        # visited[clusters[i]] = True
        visited[i] = True
        print(f"Agent {ag} visited target {target} at timestep {agent_path[ag].index(txy)}")
        break
    t += 1
  assert sum(visited) == len(targets)


def hard_coded_degenerate_cluster_test():
  starts = [11, 22, 33, 88, 99]
  targets = [72, 83, 40, 38, 27, 66, 70]
  dests = [46, 69, 19, 28, 37]
  clusters = np.arange(len(targets))
  ac_dict = {72: {0}, 83: {2}, 40: {3}, 38: {4}, 46: {0}, 69: {1}, 19: {2}, 28: {3}, 37: {4}}
  configs = dict()
  configs["problem_str"] = "msmp"
  configs["tsp_exe"] = "./pytspbridge/tsp_solver/LKH-2.0.10/LKH"
  configs["time_limit"] = 60
  configs["eps"] = 0.0
  ny = 10
  nx = 10
  grids = np.zeros((ny, nx))
  grids[5, 3:7] = 1  # obstacles

  res_dict = cbss_mcpf.RunCbssMCPF(grids, starts, targets, dests, clusters, ac_dict, configs)

  assert res_dict['best_g_value'] == 75, f"best g value was {res_dict['best_g_value']}"
  assert res_dict['num_nodes_transformed_graph'] == 29

  assert res_dict['path_set'] == {0: [[1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 4, 5, 6, 6], [1, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 4, 4, 4, 4, 4],
  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, inf]], 1: [[2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 9, 9, 9], [2, 2, 2, 2, 2, 2, 2, 2, 3, 4, 5, 6, 6],
  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, inf]], 2: [[3, 3, 2, 2, 2, 3, 3, 3, 3, 3, 4, 5, 6, 7, 7, 7, 8, 8, 8, 8, 9, 9],
  [3, 4, 4, 5, 6, 6, 7, 8, 7, 6, 6, 6, 6, 6, 5, 4, 4, 3, 2, 1, 1, 1], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, inf]],
  3: [[8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 6, 5, 4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, inf]], 4: [[9, 8, 8, 8, 8, 8, 8, 8, 7, 7],
  [9, 9, 8, 7, 6, 5, 4, 3, 3, 3], [0, 1, 2, 3, 4, 5, 6, 7, 8, inf]]}

  targets = [72, 96, 83, 40, 38, 27, 66, 70]
  clusters = [0, 1, 2, 3, 4, 5, 6, 1]
  res_dict1 = cbss_mcpf.RunCbssMCPF(grids, starts, targets, dests, clusters, ac_dict, configs)

  assert res_dict1['best_g_value'] == 75, f"best g value was {res_dict1['best_g_value']}"
  assert res_dict1['num_nodes_transformed_graph'] == 34

  assert res_dict1['path_set'] == res_dict['path_set']
  # {0: [[1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 4, 5, 6, 6], [1, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 4, 4, 4, 4, 4],
  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, inf]], 1: [[2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 9, 9, 9], [2, 2, 2, 2, 2, 2, 2, 2, 3, 4, 5, 6, 6],
  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, inf]], 2: [[3, 3, 2, 2, 2, 3, 3, 3, 3, 3, 4, 5, 6, 7, 7, 7, 8, 8, 8, 8, 9, 9],
  # [3, 4, 4, 5, 6, 6, 7, 8, 7, 6, 6, 6, 6, 6, 5, 4, 4, 3, 2, 1, 1, 1], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, inf]],
  # 3: [[8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 6, 5, 4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, inf]], 4: [[9, 8, 8, 8, 8, 8, 8, 8, 7, 7], [9, 9, 8, 7, 6, 5, 4, 3, 3, 3],
  # [0, 1, 2, 3, 4, 5, 6, 7, 8, inf]]}

  targets = [72, 81, 83, 40, 38, 27, 66, 70]
  res_dict2 = cbss_mcpf.RunCbssMCPF(grids, starts, targets, dests, clusters, ac_dict, configs)

  assert res_dict2['best_g_value'] == 75, f"best g value was {res_dict2['best_g_value']}"
  assert res_dict2['num_nodes_transformed_graph'] == 34

  assert res_dict2['path_set'] == res_dict['path_set']



def hard_coded_test(res, num_agents, cost_mat, ac_dict):
  agent_path = get_paths(res["path_set"])
  true_paths = {0: [(1, 1), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 6), (2, 5), (2, 5), (2, 4), (3, 4), (4, 4), (5, 4),
  (6, 4), (6, 4)], 1: [(2, 2), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 7), (1, 6), (2, 6), (3, 6), (4, 6), (5, 6), (6, 6), (7, 6), (8, 6),
  (9, 6), (9, 6)], 2: [(3, 3), (3, 4), (2, 4), (2, 5), (2, 6), (3, 6), (3, 7), (3, 8), (3, 7), (3, 6), (4, 6), (5, 6), (6, 6), (7, 6), (7, 5), (7, 4), (8, 4),
  (8, 3), (8, 2), (8, 1), (9, 1), (9, 1)], 3: [(8, 8), (7, 8), (7, 7), (7, 6), (7, 5), (7, 4), (6, 4), (5, 4), (4, 4), (3, 4), (2, 4), (1, 4), (0, 4), (0, 3),
  (0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2), (7, 2), (8, 2), (8, 2)], 4: [(9, 9), (8, 9), (8, 8), (8, 7), (8, 6),
  (8, 5), (8, 4), (8, 3), (7, 3), (7, 3)]}
  for i in range(num_agents):
    assert agent_path[i] == true_paths[i], i

  true_cost_mat = np.load("hard_coded_cost_mat.npy")
  assert np.array_equal(cost_mat, true_cost_mat)

  assert res["num_nodes_transformed_graph"] == 25

  assert ac_dict == {72: {0}, 81: {1}, 83: {2}, 40: {3}, 38: {4}, 46: {0}, 69: {1}, 19: {2}, 28: {3}, 37: {4}}

def run_CBSS_MCPF():
  """
  With assignment constraints.
  """
  print("------run_CBSS_MCPF------")
  ny = 10
  nx = 10
  grids = np.zeros((ny,nx))
  grids[5,3:7] = 1 # obstacles

  # grid_file = '/home/biorobotics/matspfc/datasets/maze-32-32-2.map'
  # grid_file_new = '/home/biorobotics/matspfc/datasets/maze-32-32-2_binary.map'
  #
  # with open(grid_file) as file:
  #   for i in range(4):
  #     next(file)
  #   newText = file.read().replace('@', '1 ')
  #   newText = newText.replace('.', '0 ')
  #   newText = newText
  #
  # with open(grid_file_new, 'w') as file:
  #   file.write(newText)
  #
  # grids = np.loadtxt(grid_file_new)
  # print("grid size", grids.shape)

  # starts = [11,22,33,88,99]
  # # targets = [72,81,83,40,38,27,66,73]
  # targets = [72,96,83,40,38,27,66,70]
  # dests = [46,69,19,28,37]

  starts = [11, 22, 33, 88, 99]
  targets = [72, 96, 83, 40, 38, 27, 66, 70]
  dests = [46, 69, 19, 28, 37]
  clusters = np.arange(len(targets))
  clusters = [0,1,2,3,4,5,6,1]

  # clusters = [0,1,2,3,4,5,6,1] #np.arange(len(targets))

  # starts = [79, 613, 372, 555, 755]
  # targets = [854, 191, 417, 810, 528, 141, 95, 50, 607, 377, 74, 653, 741, 843, 650]
  # dests = [865, 654, 993, 323, 1006]

  # ac_dict = dict()
  # ri = 0
  # for k in targets:
  #   ac_dict[k] = {ri}  # set([ri,ri+1])
  #   ri += 1
  #   if ri >= len(starts):#-1:
  #     break
  # ri = 0
  # for k in dests:
  #   ac_dict[k] = set([ri])
  #   ri += 1
  ac_dict = {72: {0}, 83: {2}, 40: {3}, 38: {4}, 46: {0}, 69: {1}, 19: {2}, 28: {3}, 37: {4}}
  print("Assignment constraints : ", ac_dict)

  configs = dict()
  configs["problem_str"] = "msmp"
  configs["tsp_exe"] = "./pytspbridge/tsp_solver/LKH-2.0.10/LKH"
  configs["time_limit"] = 60
  configs["eps"] = 0.0

  res_dict = cbss_mcpf.RunCbssMCPF(grids, starts, targets, dests, clusters, ac_dict, configs)

  print('n_tsp_time \t best_g_value\t num_nodes_transformed_graph')
  print(res_dict['n_tsp_time'], '\t', res_dict['best_g_value'], '\t', res_dict['num_nodes_transformed_graph'])

  path = res_dict["path_set"]
  print("path is", path)

  print(res_dict["target_assignment"])
  print(res_dict["cluster_target_selection"])
  test_targets_visited(path, ac_dict, targets, [], len(starts), nx)
  
  hard_coded_test(res_dict, len(starts), res_dict["cost_mat"], ac_dict)

  return 


if __name__ == '__main__':
  print("begin of main")

  hard_coded_degenerate_cluster_test()

  # run_CBSS_MCPF()

  print("end of main")
