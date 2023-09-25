"""
Author: Zhongqiang (Richard) Ren
All Rights Reserved.
ABOUT: Entrypoint to the code.
Oeffentlich fuer: RSS22
"""
import copy

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import context
import time
import numpy as np
import random
import cbss_msmp
import libmcpf.cbss_mcpf as cbss_mcpf

import common as cm
import imageio

from visualize import create_gif
from libmcpf import heuristics
from dataset import get_world

visited = set()

DATASET_GIVEN = True

TEST = True


def test_ac(path, ac_dict, targets, clusters, num_agents):
  num_clusters = len(np.unique(clusters))
  visited = [False for _ in range(num_clusters)]
  agent_path = [[] for _ in range(num_agents)]
  i = 0
  for agent in path:
    agent_path[i] = [p for p in list(zip(path[agent][0], path[agent][1]))]
    i += 1
  t = 0
  for i in range(len(targets)):
    target = targets[i]
    txy = (target%256, int(target/256))
    allowed_ag = ac_dict[target] if target in ac_dict else np.arange(num_agents)
    for ag in allowed_ag:
      if txy in agent_path[ag]:
        visited[clusters[i]] = True
        break
    t += 1

  print("VISITED, clusters", sum(visited), clusters)

def calculate_A_star_mat(grids, starts, targets, dests):
  print("Computing spMat")
  t1 = time.time()
  spMat = cm.getTargetGraph(grids, starts, targets, dests)
  print("A* cost mat done in time = ", time.time() - t1)
  return spMat


def get_heuristic_ac_dict(grids, starts, targets, dests, ac_dict, spMat):
  print('_______________________________________________________________________________\n\n')
  print("CBSS Heuristic")
  print('_______________________________________________________________________________\n\n')

  time_heuristic = time.time()
  ac_dict_H = heuristics.PathHeuristic(starts, targets, dests, ac_dict, grids,
                                       np.arange(len(targets)), spMat).get_updated_ac_dict()
  print("time taken by heuristic ac_dict computation = ", time.time() - time_heuristic)
  print("AC DICT NEW", ac_dict_H)
  return ac_dict_H

def call_CBSS(grids, starts, targets, dests, clusters, ac_dict, configs, spMat):
  res = {}

  for use_heuristic in [False, True]:

    if use_heuristic:
      ac_dict_new = get_heuristic_ac_dict(grids, starts, targets, dests, ac_dict, copy.deepcopy(spMat))
    else:
      ac_dict_new = copy.deepcopy(ac_dict)

    print('_______________________________________________________________________________\n\n')
    print("CBSS || Heuristic used =", use_heuristic)
    print('_______________________________________________________________________________\n\n')
    t1 = time.time()

    # call CBSS
    res_dict = cbss_mcpf.RunCbssMCPF(grids, starts, targets, dests, clusters, ac_dict_new, configs, copy.deepcopy(spMat))

    print("Time taken by CBSS = ", time.time() - t1)
    print('n_tsp_time \t best_g_value\t num_nodes_transformed_graph')
    print(res_dict['n_tsp_time'], '\t', res_dict['best_g_value'], '\t', res_dict['num_nodes_transformed_graph'])

    path = res_dict['path_set']
    max_step = 0
    for agent in path:
      max_step = max(max_step, path[agent][2][-2])
    print("Max step = ", max_step)
    print_paths(path)
    test_ac(path, ac_dict, targets, clusters, len(starts))

    res[int(use_heuristic)] = res_dict

  return res

def print_paths(res_path):
  agent_paths = {}
  for agent in res_path:
    agent_paths[agent] = [p for p in list(zip(res_path[agent][0], res_path[agent][1]))]
    # print("Agent {}'s path is ".format(agent), [p for p in list(zip(res_path[agent][0], res_path[agent][1]))], "at times",
    #       res_path[agent][2])
  return agent_paths


def run_tests(grids, starts, targets, dests, clusters, ac_dict, configs, res_dict):

  # TARGET ASSIGNMENT TEST
  target_assignment = res_dict["target_assignment"]
  print("target assignment dict values", target_assignment)
  ac_dict_ta = target_assignment  #copy.deepcopy(ac_dict)
  targets_ta = []
  for t in target_assignment.keys():
    # print("KEY", t)
    # ac_dict_ta[t] = target_assignment[t]  # should overwrite targets but leave dests unmodified
    if t in targets:
      targets_ta.append(t)
  spMat = calculate_A_star_mat(grids, starts, targets_ta, dests)
  print("TEST TARGET ASSIGNMENT\nAC_DICT\t=\t", ac_dict_ta)
  res_dict_ta = cbss_mcpf.RunCbssMCPF(grids, starts, targets_ta, dests, None, ac_dict_ta, configs, spMat, test=True)
  print("TEST paths:\n", print_paths(res_dict_ta["path_set"]))
  print("Original paths\n", print_paths(res_dict["path_set"]))
  assert res_dict_ta['path_set'] == res_dict["path_set"]

  print("COST COMPARISON", res_dict_ta["best_g_value"], res_dict["best_g_value"])

  # CLUSTER TARGET SELECTION TEST
  cluster_target_selection = res_dict["cluster_target_selection"]
  targets_cts = []
  for c in cluster_target_selection.keys():
    targets_cts.append(cluster_target_selection[c])

  ac_dict_cts = copy.deepcopy(ac_dict)
  for t in ac_dict.keys():
    if t in targets and t not in targets_cts:
      del ac_dict_cts[t]
  spMat = calculate_A_star_mat(grids, starts, targets_cts, dests)
  res_dict_cts = cbss_mcpf.RunCbssMCPF(grids, starts, targets_cts, dests, None, ac_dict_cts, configs, spMat, test=True)
  print("TEST paths:\n", print_paths(res_dict_cts["path_set"]))
  print("Original paths\n", print_paths(res_dict["path_set"]))

  print("COST COMPARISON", res_dict_cts["best_g_value"], res_dict["best_g_value"])
  assert res_dict_cts['path_set'] == res_dict_ta["path_set"]


def run_CBSS_MCPF():
  """
  With assignment constraints.
  """

  if DATASET_GIVEN:
    starts, dests, targets, grids, clusters = get_world(num_agents=3, num_targets=10)
  else:
    ny = 10
    nx = 10
    starts = [11, 22, 33, 88, 99]
    targets = [72, 81, 83, 40, 38, 27, 66]
    dests = [46, 69, 19, 28, 37]
    grids = np.zeros((ny, nx))
    grids[5, 3:7] = 1  # obstacles
    clusters = np.array([0,1,2,2,2,2,2])  # np.arange(len(targets))
  print("------run_CBSS_MCPF------")
  print("SETUP AT START")

  ac_dict = dict()
  ri = 0
  for k in targets:
    ac_dict[k] = set([ri,ri+1])
    ri += 1
    if ri >= len(starts)-1:
      break
  ri = 0
  for k in dests:
    ac_dict[k] = set([ri])
    ri += 1
  print("Assignment constraints : ", ac_dict)

  configs = dict()
  configs["problem_str"] = "msmp"
  configs["mtsp_fea_check"] = 1
  configs["mtsp_atLeastOnce"] = 1
    # this determines whether the k-best TSP step will visit each node for at least once or exact once.
  configs["tsp_exe"] = "./pytspbridge/tsp_solver/LKH-2.0.10/LKH"
  configs["time_limit"] = 60 * 60 * 24
  configs["eps"] = 0.0

  spMat = calculate_A_star_mat(grids, starts, targets, dests)
  res = call_CBSS(grids, starts, targets, dests, clusters, ac_dict, configs, spMat)

  res_dict = res[0]


  if TEST:
    run_tests(grids, starts, targets, dests, clusters, ac_dict, configs, res_dict)




  # create_gif(grids, targets, dests, ac_dict, clusters=None, path=res_dict['path_set'], name='heuristic_ano')
  # print(res_dict)

  # visualize_grid(grids, starts, targets, dests, res_dict['path_set'])
  # create_gif(grids, targets, dests, ac_dict, clusters=None, path=res_dict['path_set'])

  return


if __name__ == '__main__':
  print("begin of main")

  run_CBSS_MCPF()

  print("end of main")
