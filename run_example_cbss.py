"""
Author: Zhongqiang (Richard) Ren
All Rights Reserved.
ABOUT: Entrypoint to the code.
Oeffentlich fuer: RSS22
"""
import copy
import math
import pickle
from math import inf

import context
import time
import numpy as np
import random
import libmcpf.cbss_mcpf as cbss_mcpf

import common as cm

import libmcpf.heuristics as heuristics
from util_data_structs import Results, Unit


def get_paths(res_path):
  agent_paths = {}
  max_step = 0
  for agent in res_path:
    agent_paths[agent] = [p for p in list(zip(res_path[agent][0], res_path[agent][1]))]
    max_step = max(max_step, res_path[agent][2][-2])
    # print("Agent {}'s path is ".format(agent), [p for p in list(zip(res_path[agent][0], res_path[agent][1]))], "at times",
    #       res_path[agent][2])
  return agent_paths, max_step


def test_targets_visited(path, ac_dict, targets, clusters, num_agents, sz):
  # num_clusters = len(np.unique(clusters))
  # visited = [False for _ in range(num_clusters)]
  visited = [False for _ in range(len(targets))]
  visited_clusters = [False for _ in range(max(clusters) + 1)]
  agent_path, m = get_paths(path)
  # print(agent_path)
  t = 0
  for i in range(len(targets)):
    target = targets[i]
    txy = (target % sz, int(target / sz))
    allowed_ag = ac_dict[target] if target in ac_dict else np.arange(num_agents)
    for ag in allowed_ag:
      if txy in agent_path[ag]:
        visited_clusters[clusters[i]] = True
        visited[i] = True
        # print(f"Agent {ag} visited target {target} at timestep {agent_path[ag].index(txy)}")
        break
    t += 1

  assert sum(visited_clusters) == max(clusters) + 1, f"{sum(visited)} and {max(clusters)}"


def hard_coded_degenerate_cluster_test():
  starts = [11, 22, 33, 88, 99]
  targets = [72, 83, 40, 38, 27, 66, 70]
  dests = [46, 69, 19, 28, 37]
  clusters = np.arange(len(targets))
  ac_dict = {72: {0}, 83: {2}, 40: {3}, 38: {4}, 46: {0}, 69: {1}, 19: {2}, 28: {3}, 37: {4}}
  configs = dict()
  configs["problem_str"] = "msmp"
  configs["tsp_exe"] = "./pytspbridge/tsp_solver/LKH-2.0.10/LKH"
  configs["time_limit"] = 60 * 30
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

  starts = [11, 22, 33, 88, 99]
  targets = [72, 96, 83, 40, 38, 27, 66, 70]
  dests = [46, 69, 19, 28, 37]
  clusters = np.arange(len(targets))
  clusters = [0,1,2,3,4,5,6,1]


  ac_dict = {72: {0}, 83: {2}, 40: {3}, 38: {4}, 46: {0}, 69: {1}, 19: {2}, 28: {3}, 37: {4}}
  print("Assignment constraints : ", ac_dict)

  configs = dict()
  configs["problem_str"] = "msmp"
  configs["tsp_exe"] = "./pytspbridge/tsp_solver/LKH-2.0.10/LKH"
  configs["time_limit"] = 60 / 2
  configs["eps"] = 0.0

  spMat = cm.getTargetGraph(grids, starts, targets, dests)

  res_dict = cbss_mcpf.RunCbssMCPF(grids, starts, targets, dests, clusters, ac_dict, configs, copy.deepcopy(spMat))

  print('n_tsp_time \t best_g_value\t num_nodes_transformed_graph')
  print(res_dict['n_tsp_time'], '\t', res_dict['best_g_value'], '\t', res_dict['num_nodes_transformed_graph'])

  path = res_dict["path_set"]
  test_targets_visited(path, ac_dict, targets, clusters, len(starts), nx)


def save_hr_result(starts, dests, targets, ac_dict, grids, clusters, sz, configs, spMat, results_path, dname):
  print("Welcome to Heuristic!")
  ac_dict_new = heuristics.PathHeuristic(starts, targets, dests, ac_dict, grids, clusters, spMat,
                                         0).get_updated_ac_dict()
  res_dict_hr = cbss_mcpf.RunCbssMCPF(grids, starts, targets, dests, clusters, ac_dict_new, configs, spMat)
  print("TRUE Heuristic", res_dict_hr['best_g_value'])
  path = res_dict_hr['path_set']
  max_step = 0
  for agent in path:
    max_step = max(max_step, path[agent][2][-2])
  agent_paths = get_paths(path)
  test_targets_visited(path, ac_dict, targets, clusters, len(starts), sz)

  num_agents = len(starts)
  num_targets = len(targets)
  num_clusters = len(clusters)

  # write to file
  fpath = results_path + f"numpyfiles/{dname}_N{num_agents}_M{num_targets}_K{num_clusters}"
  with open(fpath + f"_h1.npy", 'wb') as f:
    pickle.dump(res_dict_hr, f, pickle.HIGHEST_PROTOCOL)

  with open(results_path + f"{dname}_N{num_agents}_M{num_targets}_K{num_clusters}_h1.txt", "w") as f:
    print(res_dict_hr, file=f)
    print(agent_paths, file=f)




def call_CBSS_c(starts, dests, targets, acd, grids, clusters, sz, f):
  configs = dict()
  configs["tsp_exe"] = "./pytspbridge/tsp_solver/LKH-2.0.10/LKH"
  configs["time_limit"] = 60 * 3
  configs["eps"] = 0.0

  ac_dict = copy.deepcopy(acd)

  # Astar_time = time.time()
  spMat = cm.getTargetGraph(grids, starts, targets, dests)
  # Astar_time = time.time() - Astar_time
  spMat_copy = copy.deepcopy(spMat)

  # CBSS-c
  # configs["eps"] = 0.0
  # success = [0,0,0,0]

  res_dict = cbss_mcpf.RunCbssMCPF(grids, starts, targets, dests, clusters, ac_dict, configs, spMat)
  # if res_dict['best_g_value'] > 0:
  #   success[0] = 1
  # elif res_dict['best_g_value'] == -1:
  #   success[1] = -1

    # print("CBSS-c COST, nTSP, CONFLICTS, nodes",
  #       res_dict['best_g_value'], res_dict['n_tsp_time'], res_dict['num_conflicts'],
  #       res_dict["num_nodes_transformed_graph"])
  c1, t1 = res_dict['best_g_value'], res_dict['n_tsp_time']
  cn1, tn1 = res_dict["num_conflicts"], res_dict["num_nodes_transformed_graph"]

  # cn1, n1 = res_dict['num_conflicts'], res_dict['num_nodes_transformed_graph']

  # configs["eps"] = 0.01
  # res_dict = cbss_mcpf.RunCbssMCPF(grids, starts, targets, dests, clusters, ac_dict, configs, spMat)
  # c2, t2 = res_dict['best_g_value'], res_dict['n_tsp_time']
  # print("EPS=0.01: CBSS-c COST, nTSP, CONFLICTS",
  #       res_dict['best_g_value'], res_dict['n_tsp_time'], res_dict['num_conflicts'])
  #
  # configs["eps"] = 0.1
  # res_dict = cbss_mcpf.RunCbssMCPF(grids, starts, targets, dests, clusters, ac_dict, configs, spMat)
  # c3, t3 = res_dict['best_g_value'], res_dict['n_tsp_time']
  # print("EPS=0.1: CBSS-c COST, nTSP, CONFLICTS",
  #       res_dict['best_g_value'], res_dict['n_tsp_time'], res_dict['num_conflicts'])
  #
  # configs["eps"] = math.inf
  # res_dict = cbss_mcpf.RunCbssMCPF(grids, starts, targets, dests, clusters, ac_dict, configs, spMat)
  # c4, t4 = res_dict['best_g_value'], res_dict['n_tsp_time']
  # print("EPS=inf: CBSS-c COST, nTSP, CONFLICTS",
  #       res_dict['best_g_value'], res_dict['n_tsp_time'], res_dict['num_conflicts'])

  # return t1, t2, t3, t4, c1, c2, c3, c4

  test_targets_visited(res_dict['path_set'], ac_dict, targets, clusters, len(starts), sz)

  # with open(f + '_h0.txt', "w") as file:
  #   file.write("\n\nCBSS-C STATS")
  #   file.write('\nstarts:'+str(starts) + '\ntargets:' + str(targets) + '\ndests:'+str(dests))
  #   file.write('\nac_dict:' + str(ac_dict) + '\nclusters:' + str(clusters) + '\n\n')
  #   for k in res_dict:
  #     if 'mat' not in k.lower() and 'path' not in k:
  #       file.write(k + ': ' + str(res_dict[k]) + '\n')
  #   file.write('agent_paths:' + str(agent_paths))

  # print("CBSS-c", unit.res_cbss_c.cost, unit.res_cbss_c.ntsp)

  # Heuristic

  ac_dict_new = heuristics.PathHeuristic(starts,targets,dests,acd,grids,clusters,spMat_copy,
                                         0).get_updated_ac_dict()
  res_dict = cbss_mcpf.RunCbssMCPF(grids, starts, targets, dests, clusters, ac_dict_new, configs, spMat_copy)
  # print("a=0 | CBSS-ch COST, nTSP, conflicts, nodes:", res_dict['best_g_value'], res_dict['n_tsp_time'], res_dict["num_conflicts"],
  #       res_dict["num_nodes_transformed_graph"])
  test_targets_visited(res_dict['path_set'], ac_dict, targets, clusters, len(starts), sz)
  ch0, th0 = res_dict['best_g_value'], res_dict['n_tsp_time']
  chn0, thn0 = res_dict["num_conflicts"], res_dict["num_nodes_transformed_graph"]

  ac_dict_new = heuristics.PathHeuristic(starts, targets, dests, ac_dict, grids, clusters, spMat,
                                         3).get_updated_ac_dict()
  res_dict = cbss_mcpf.RunCbssMCPF(grids, starts, targets, dests, clusters, ac_dict_new, configs, spMat_copy)
  test_targets_visited(res_dict['path_set'], ac_dict, targets, clusters, len(starts), sz)
  ch3, th3 = res_dict['best_g_value'], res_dict['n_tsp_time']
  chn3, thn3 = res_dict["num_conflicts"], res_dict["num_nodes_transformed_graph"]


  ac_dict_new = heuristics.PathHeuristic(starts, targets, dests, ac_dict, grids, clusters, spMat,
                                         4).get_updated_ac_dict()
  res_dict = cbss_mcpf.RunCbssMCPF(grids, starts, targets, dests, clusters, ac_dict_new, configs, spMat_copy)
  test_targets_visited(res_dict['path_set'], ac_dict, targets, clusters, len(starts), sz)
  ch4, th4 = res_dict['best_g_value'], res_dict['n_tsp_time']
  chn4, thn4 = res_dict["num_conflicts"], res_dict["num_nodes_transformed_graph"]


  ac_dict_new = heuristics.PathHeuristic(starts, targets, dests, ac_dict, grids, clusters, spMat,
                                         5).get_updated_ac_dict()
  res_dict = cbss_mcpf.RunCbssMCPF(grids, starts, targets, dests, clusters, ac_dict_new, configs, spMat_copy)
  ch5, th5 = res_dict['best_g_value'], res_dict['n_tsp_time']
  chn5, thn5 = res_dict["num_conflicts"], res_dict["num_nodes_transformed_graph"]
  test_targets_visited(res_dict['path_set'], ac_dict, targets, clusters, len(starts), sz)

  # print("a=5 | CBSS-ch COST, nTSP, conflicts, nodes:", res_dict['best_g_value'], res_dict['n_tsp_time'],
  #       res_dict["num_conflicts"],
  #       res_dict["num_nodes_transformed_graph"])
  # c2, t2 = res_dict['best_g_value'], res_dict['n_tsp_time']
  # cn2, n2 = res_dict['num_conflicts'], res_dict['num_nodes_transformed_graph']
  # max_step = 0
  # for agent in path:
  #   max_step = max(max_step, path[agent][2][-2])


  arr = np.array([c1, t1, cn1, tn1, ch3, th3, chn3, thn3,ch0, th0, chn0, thn0,
                  ch4, th4, chn4, thn4, ch5, th5, chn5, thn5])
  return arr

  # print(c1,c2,'\t\t',t1,t2)
  return (c1, t1, cn1, tn1, ch3, th3, chn3, thn3,ch0, th0, chn0, thn0,
          ch4, th4, chn4, thn4, ch5, th5, chn5, thn5)
  # return c1,c2, t1,t2, cn1,cn2,n1,n2


if __name__ == '__main__':
  print("begin of main")

  # targets = [141, 545, 734, 472, 1015, 1014, 962, 69, 358, 244, 645, 255, 656, 643, 334]
  # clusters = [1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0]

  starts = [84, 689, 634, 522, 358]
  targets = [727, 86, 360, 333, 839, 1022, 150, 644, 937, 172, 570, 271, 673, 272, 313, 314, 989, 746, 667, 830, 74, 639, 996, 957, 329, 565, 545, 370, 857, 338]
  dests =[688, 364, 125, 348, 68]
  clusters = [0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]
  grids = np.loadtxt(f'/home/biorobotics/matspfc/datasets/maze-32-32-2_binary.map')
  acd = {727: {0, 1}, 86: {1, 2}, 360: {2, 3}, 333: {3, 4}, 688: {0}, 364: {1}, 125: {2}, 348: {3}, 68: {4}} #pickle.load(open('nov16/maze-32-32-2/numpyfiles/maze-32-32-2_N5_M30_K10_h01_run1.npy', 'rb')).res_cbss_c.ac_dict
  tgid = None #[(3, 4), (27, 20), (28, 3), (12, 28), (9, 4), (7, 23), (7, 26), (25, 26), (4, 23), (30, 25), (13, 13), (9, 10), (29, 7),
          # (31, 30), (5, 13), (10, 31), (11, 17), (28, 14), (20, 29), (31, 6), (27, 19), (13, 24), (20, 24), (19, 25), (18, 1), (11, 18),
          # (13, 18), (13, 14), (26, 6), (5, 31)]

  print("initial acd:", acd)

  for _ in range(1):
    u = call_CBSS_c(starts, dests, targets, tgid, acd, grids, clusters, 32)
    print(u.res_hr.cost, u.res_cbss_c.cost)
    print(u.res_hr.ac_dict, '\n', u.res_cbss_c.ac_dict)
    print("----the later acds----\n", acd)
    print('__________________________________________________')



  # res = call_CBSS_c()

  # res = call_CBSS_c(starts=[86, 271, 180, 40, 346, 456, 815, 245, 834, 939],
  #             targets=targets, dests=[435, 910, 355, 622, 514, 1021, 627, 921, 566, 790], ac_dict={141: {0}, 545: {1, 2},
  # 734: {3}, 472: {4}, 1015: {5}, 1014: {5, 6}, 962: {7}, 69: {7}, 358: {8}, 435: {0}, 910: {1}, 355: {2}, 622: {3}, 514: {4},
  # 1021: {5}, 627: {6}, 921: {7}, 566: {8}, 790: {9}, 244: {0, 1, 2, 3, 7}, 645: {0, 1, 4, 5, 7, 8}, 255: {0}, 656: {0, 4, 6, 8},
  # 643: {0, 1, 4, 5, 7, 8}, 334: {0, 1, 2, 3, 7, 8}}, clusters=clusters,
  #             grids=pickle.load(open('results_nov/maze-32-32-2/numpyfiles/maze-32-32-2_N10_M15_K2_h1_run1.npy', 'rb')).grids, sz=32)
  # # run_CBSS_MCPF()

  print("end of main")
