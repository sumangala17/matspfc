"""
Author: Zhongqiang (Richard) Ren
All Rights Reserved.
ABOUT: Entrypoint to the code.
Oeffentlich fuer: RSS22
"""

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
        break
    t += 1
  assert sum(visited) == len(targets)


def hard_coded_test(agent_path, num_agents, cost_mat):
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

  starts = [11,22,33,88,99]
  targets = [72,81,83,40,38,27,66]
  dests = [46,69,19,28,37]

  # starts = [79, 613, 372, 555, 755]
  # targets = [854, 191, 417, 810, 528, 141, 95, 50, 607, 377, 74, 653, 741, 843, 650]
  # dests = [865, 654, 993, 323, 1006]

  ac_dict = dict()
  ri = 0
  for k in targets:
    ac_dict[k] = {ri}  # set([ri,ri+1])
    ri += 1
    if ri >= len(starts):#-1:
      break
  ri = 0
  for k in dests:
    ac_dict[k] = set([ri])
    ri += 1
  print("Assignment constraints : ", ac_dict)

  configs = dict()
  configs["problem_str"] = "msmp"
  configs["tsp_exe"] = "./pytspbridge/tsp_solver/LKH-2.0.10/LKH"
  configs["time_limit"] = 60
  configs["eps"] = 0.0

  res_dict = cbss_mcpf.RunCbssMCPF(grids, starts, targets, dests, ac_dict, configs)

  path = res_dict["path_set"]

  print(get_paths(res_dict["path_set"]))
  agent_path = get_paths(res_dict["path_set"])
  test_targets_visited(path, ac_dict, targets, [], len(starts), nx)
  
  hard_coded_test(agent_path, len(starts), res_dict["cost_mat"])

  return 


if __name__ == '__main__':
  print("begin of main")

  run_CBSS_MCPF()

  print("end of main")
