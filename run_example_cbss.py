"""
Author: Zhongqiang (Richard) Ren
All Rights Reserved.
ABOUT: Entrypoint to the code.
Oeffentlich fuer: RSS22
"""
import copy
import json

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


def view_agent_path_targets(path, targets):
  agent_paths = get_paths(path)
  i = 0
  for ag in agent_paths:
    for point in agent_paths[ag]:
      p = point[0] + 256 * point[1]
      if p in targets:
        print("Agent {} visited target {}".format(i, p))
    i += 1

def test_ac(path, ac_dict, targets, clusters, num_agents, sz):
    # num_clusters = len(np.unique(clusters))
    # visited = [False for _ in range(num_clusters)]
    visited_T = [False for _ in range(len(targets))]
    agent_path = get_paths(path)
    target_list_coord = []
    t = 0
    for i in range(len(targets)):
        target = targets[i]
        txy = (target%sz, int(target/sz))
        allowed_ag = ac_dict[target] if target in ac_dict else np.arange(num_agents)
        for ag in allowed_ag:
            if txy in agent_path[ag]:
                # visited[clusters[i]] = True
                target_list_coord.append((txy, target))
                visited_T[i] = True
                break
        t += 1
    if sum(visited_T) != len(targets):
        print("Not all targets were visited!")
        print("vis targets", visited_T)
        print("VISITED coords", target_list_coord)

def calculate_A_star_mat(grids, starts, targets, dests):
  # print("Computing spMat")
  t1 = time.time()
  spMat = cm.getTargetGraph(grids, starts, targets, dests)
  # print("A* cost mat done in time = ", time.time() - t1)
  return spMat


def get_heuristic_ac_dict(grids, starts, targets, dests, clusters, ac_dict, spMat, alpha):
  # print('_______________________________________________________________________________\n\n')
  # print("CBSS Heuristic")
  # print('_______________________________________________________________________________\n\n')

  # time_heuristic = time.time()
  ac_dict_H = heuristics.PathHeuristic(starts, targets, dests, copy.deepcopy(ac_dict), grids,
                                       clusters, spMat, alpha).get_updated_ac_dict()
  # print("time taken by heuristic ac_dict computation = ", time.time() - time_heuristic)
  # print("AC DICT NEW", ac_dict_H)
  return ac_dict_H

def call_CBSS(grids, starts, targets, dests, clusters, ac_dict, configs, spMat):
  res = {}
  print("clusters = ", clusters)

  ac_dict_list = []

  for use_heuristic in [ False, True]:
    print('_______________________________________________________________________________\n\n')
    print("CBSS || Heuristic used =", use_heuristic)
    print('_______________________________________________________________________________\n\n')

    if use_heuristic:
      for alpha in [0.2,0.5,0.8,2,3,4,5,6, 7]:
        print("ALPHA = ", alpha)
        ac_dict_new = get_heuristic_ac_dict(grids, starts, targets, dests, clusters, copy.deepcopy(ac_dict), copy.deepcopy(spMat), alpha)
        ac_dict_list.append(ac_dict_new)
        res_dict = cbss_mcpf.RunCbssMCPF(grids, starts, targets, dests, clusters, ac_dict_new, configs,
                                         copy.deepcopy(spMat))
        print('n_tsp_time \t best_g_value\t num_nodes_transformed_graph')
        print(res_dict['n_tsp_time'], '\t', res_dict['best_g_value'], '\t', res_dict['num_nodes_transformed_graph'])
        test_ac(res_dict['path_set'], ac_dict_new, targets, clusters, len(starts), 32)
        print('_'*100)
    else:
      ac_dict_new = copy.deepcopy(ac_dict)

    t1 = time.time()

    # call CBSS
    # if use_heuristic:
    #   for ac_dict_new in ac_dict_list:
        # print("ALPHA = ",)
        # res_dict = cbss_mcpf.RunCbssMCPF(grids, starts, targets, dests, clusters, ac_dict_new, configs,
        #                                  copy.deepcopy(spMat))
        # print('n_tsp_time \t best_g_value\t num_nodes_transformed_graph')
        # print(res_dict['n_tsp_time'], '\t', res_dict['best_g_value'], '\t', res_dict['num_nodes_transformed_graph'])
    # res_dict1 = cbss_mcpf.RunCbssMCPF(grids, starts, targets, dests, clusters, ac_dict_new, configs, copy.deepcopy(spMat), True)

    if not use_heuristic:
      res_dict = cbss_mcpf.RunCbssMCPF(grids, starts, targets, dests, clusters, ac_dict_new, configs, copy.deepcopy(spMat))

      print("Time taken by CBSS = ", time.time() - t1)
      print('n_tsp_time \t best_g_value\t num_nodes_transformed_graph')
      print(res_dict['n_tsp_time'], '\t', res_dict['best_g_value'], '\t', res_dict['num_nodes_transformed_graph'])

      path = res_dict['path_set']
      max_step = 0
      for agent in path:
        max_step = max(max_step, path[agent][2][-2])
      print("Max step = ", max_step)
      get_paths(path)
      test_ac(path, ac_dict, targets, clusters, len(starts), len(grids))

      res[int(use_heuristic)] = res_dict

  return res

def get_paths(res_path):
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
  # spMat = calculate_A_star_mat(grids, starts, targets_ta, dests)
  print("TEST TARGET ASSIGNMENT\nAC_DICT\t=\t", ac_dict)
  res_dict_ta = cbss_mcpf.RunCbssMCPF(grids, starts, targets, dests, None, ac_dict, configs, spMat=None, test=True)
  print("TEST paths:\n", get_paths(res_dict_ta["path_set"]))
  print("Original paths\n", get_paths(res_dict["path_set"]))
  test_ac(res_dict_ta["path_set"], ac_dict, targets, clusters, len(starts))
  print("COST COMPARISON", res_dict_ta["best_g_value"], res_dict["best_g_value"])
  view_agent_path_targets(res_dict_ta["path_set"], targets)
  view_agent_path_targets(res_dict["path_set"], targets)
  # assert res_dict_ta['path_set'] == res_dict["path_set"]

  # CLUSTER TARGET SELECTION TEST
  cluster_target_selection = res_dict["cluster_target_selection"]
  targets_cts = []
  for c in cluster_target_selection.keys():
    targets_cts.append(cluster_target_selection[c])

  ac_dict_cts = copy.deepcopy(ac_dict)
  for t in ac_dict.keys():
    if t in targets and t not in targets_cts:
      del ac_dict_cts[t]
  # spMat = calculate_A_star_mat(grids, starts, targets_cts, dests)
  res_dict_cts = cbss_mcpf.RunCbssMCPF(grids, starts, targets_cts, dests, None, ac_dict_cts, configs, spMat=None, test=True)
  test_ac(res_dict_cts["path_set"], ac_dict, targets, clusters, len(starts))
  print("TEST paths:\n", get_paths(res_dict_cts["path_set"]))
  print("Original paths\n", get_paths(res_dict["path_set"]))

  print("COST COMPARISON", res_dict_cts["best_g_value"], res_dict["best_g_value"])
  view_agent_path_targets(res_dict_cts["path_set"], targets)
  view_agent_path_targets(res_dict["path_set"], targets)
  # assert res_dict_cts['path_set'] == res_dict_ta["path_set"]


class Results:
    def __init__(self, starts, targets, dests, clusters, grids, ac_dict, eps, spMat, is_heuristic):
        self.starts = starts
        self.targets = targets
        self.dests = dests
        self.clusters = clusters
        self.grids = grids
        self.ac_dict = ac_dict
        self.eps = eps
        self.spMat = spMat
        self.is_heuristic = is_heuristic

    def set_stats(self, total_time, ntsp, cost, agent_paths, target_assignment, cluster_target_selection, num_nodes,
                  max_step, Astar_time, num_conflicts):
        self.total_time = total_time
        self.ntsp = ntsp
        self.cost = cost
        self.agent_paths = agent_paths
        self.target_assignment = target_assignment
        self.cluster_target_selection = cluster_target_selection
        self.num_nodes = num_nodes
        self.max_step = max_step
        self.Astar_time = Astar_time
        self.num_conflicts = num_conflicts

    def print_stats(self):
        return self.total_time

    def get_json(self):
        return str(self)


def call_CBSS_c(starts, dests, targets, ac_dict, grids, clusters, sz):
  configs = dict()
  # configs["problem_str"] = "msmp"
  configs["mtsp_fea_check"] = 0
  configs["mtsp_atLeastOnce"] = 1
  # this determines whether the k-best TSP step will visit each node for at least once or exact once.
  configs["tsp_exe"] = "./pytspbridge/tsp_solver/LKH-2.0.10/LKH"
  configs["time_limit"] = 60 * 10
  configs["eps"] = 0.0
  # load_env = True

  Astar_time = time.time()
  spMat = calculate_A_star_mat(grids, starts, targets, dests)
  Astar_time = time.time() - Astar_time
  spMat_copy = copy.deepcopy(spMat)
  # call_CBSS(grids, starts, targets, dests, clusters, ac_dict, configs, spMat)

  # CBSSc
  total_time = time.time()
  res_dict = cbss_mcpf.RunCbssMCPF(grids, starts, targets, dests, clusters, ac_dict, configs, spMat)
  # create_gif(grids, targets, dests, ac_dict, clusters=clusters, path=res_dict['path_set'], name="test1")
  total_time = time.time() - total_time
  path = res_dict['path_set']
  max_step = 0
  for agent in path:
    max_step = max(max_step, path[agent][2][-2])
  agent_paths = get_paths(path)
  test_ac(path, ac_dict, targets, clusters, len(starts), sz)

  cbss_c_res = Results(starts, targets, dests, clusters, grids, ac_dict, configs["eps"], spMat, is_heuristic=False)
  cbss_c_res.set_stats(total_time, res_dict["n_tsp_time"], res_dict["best_g_value"], agent_paths,
                       res_dict["target_assignment"], res_dict["cluster_target_selection"],
                       res_dict["num_nodes_transformed_graph"], max_step, Astar_time, res_dict["num_conflicts"])

  # Heuristic

  total_time = time.time()
  ac_dict_new = get_heuristic_ac_dict(grids, starts, targets, dests, ac_dict, spMat_copy)
  res_dict = cbss_mcpf.RunCbssMCPF(grids, starts, targets, dests, clusters, ac_dict_new, configs, spMat_copy)
  total_time = time.time() - total_time
  path = res_dict['path_set']
  max_step = 0
  for agent in path:
    max_step = max(max_step, path[agent][2][-2])
  agent_paths = get_paths(path)
  test_ac(path, ac_dict, targets, clusters, len(starts), sz)

  heuristic_res = Results(starts, targets, dests, clusters, grids, ac_dict, configs["eps"], spMat, is_heuristic=True)
  heuristic_res.set_stats(total_time, res_dict["n_tsp_time"], res_dict["best_g_value"], agent_paths,
                       res_dict["target_assignment"], res_dict["cluster_target_selection"],
                       res_dict["num_nodes_transformed_graph"], max_step, Astar_time, res_dict["num_conflicts"])

  return cbss_c_res, heuristic_res



def run_CBSS_MCPF():
  """
  With assignment constraints.
  """

  if DATASET_GIVEN:
    # starts, dests, targets, grids, clusters = get_world(num_agents=10, num_targets=20, num_clusters=10)  # test fail!
    starts, dests, targets, grids, clusters = get_world(num_agents=5, num_targets=15, num_clusters=1)
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
  # for k in targets:
  #   ac_dict[k] = set([ri%2])
  #   ri += 1
  #   if ri >= len(starts)+1:
  #     break
  ri = 0
  for k in dests:
    ac_dict[k] = set([ri])
    ri += 1
  print("Assignment constraints : ", ac_dict)


  configs = dict()
  configs["problem_str"] = "msmp"
  configs["mtsp_fea_check"] = 0
  configs["mtsp_atLeastOnce"] = 1
    # this determines whether the k-best TSP step will visit each node for at least once or exact once.
  configs["tsp_exe"] = "./pytspbridge/tsp_solver/LKH-2.0.10/LKH"
  configs["time_limit"] = 60 * 60 * 24
  configs["eps"] = 0.0
  # load_env = True

  # cluster_cost_mat_test(configs)
  # return

  # if not load_env:
  spMat = calculate_A_star_mat(grids, starts, targets, dests)

  # filename = "latest.npy"
  # if not load_env:
  #   obj = {"starts": starts, "targets": targets, "dests": dests, "clusters": clusters, "spMat": spMat, "grids": grids}
  #   np.save(filename, obj, allow_pickle=True)

  # if load_env:
  #   print("LOADING FROM latest.py ...")
  #   obj = np.load(filename, allow_pickle=True).tolist()
  #   starts, targets, dests, clusters, spMat, grids = obj["starts"], obj["targets"], obj["dests"], obj["clusters"], obj["spMat"], obj["grids"]


  # print("SPMAT______________________\n", spMat)
  call_CBSS(grids, starts, targets, dests, clusters, ac_dict, configs, spMat)


  # res_dict = res[0]




  # if TEST:
  #   print("yay", ac_dict)
  #   run_tests(grids, starts, targets, dests, clusters, ac_dict, configs, res[0])
  #   print("USING HEURISTIC ____________________________________________________")
  #   run_tests(grids, starts, targets, dests, clusters, ac_dict, configs, res[1])




  # create_gif(grids, targets, dests, ac_dict, clusters=None, path=res_dict['path_set'], name='heuristic_ano')
  # print(res_dict)

  # visualize_grid(grids, starts, targets, dests, res_dict['path_set'])
  # create_gif(grids, targets, dests, ac_dict, clusters=None, path=res_dict['path_set'])

  return


if __name__ == '__main__':
  print("begin of main")

  run_CBSS_MCPF()

  print("end of main")

















