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

def run_CBSS_MCPF():
  """
  With assignment constraints.
  """

  if DATASET_GIVEN:
    starts, dests, targets, grids, cluster_target_map = get_world()
  else:
    ny = 10
    nx = 10
    starts = [11, 22, 33, 88, 99]
    targets = [72, 81, 83, 40, 38, 27, 66]
    dests = [46, 69, 19, 28, 37]
    grids = np.zeros((ny, nx))
    grids[5, 3:7] = 1  # obstacles
  print("------run_CBSS_MCPF------")
  # ny = 10
  # nx = 10
  # grids = np.zeros((ny,nx))
  # grids[5,3:7] = 1 # obstacles
  #
  # starts = [11,22,33,88,99]
  # targets = [72,81,83,40,38,27,66]
  # dests = [46,69,19,28,37]

  print("SETUP AT START")
  # visualize_grid(grids, starts, targets, dests)

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
  configs["time_limit"] = 60 * 10
  configs["eps"] = 0.0

  print("Computing spMat")
  t1 = time.time()
  spMat = cm.getTargetGraph(grids, starts, targets, dests)
  print("A* cost mat done in time = ", time.time() - t1)

  # spMat_copy = copy.deepcopy(spMat)
  # print("AC DICT OLD", ac_dict)
  # time_heuristic = time.time()
  # ac_dict = heuristics.PathHeuristic(starts, targets, dests, ac_dict, grids, np.arange(len(targets)), copy.deepcopy(spMat_copy)).get_updated_ac_dict()
  # print("time taken by heuristic = ", time.time() - time_heuristic)
  # print("AC DICT NEW", ac_dict)

  t1 = time.time()
  res_dict = cbss_mcpf.RunCbssMCPF(grids, starts, targets, dests, ac_dict, configs, spMat)
  print("Time taken by CBSS = ", time.time() - t1)

  for key, value in res_dict.items():
    if key in ['best_g_value', 'open_list_size', 'num_low_level_expanded', 'search_success', 'search_time',
               'n_tsp_call', 'n_tsp_time', 'n_roots']:
      print(key, '\t=\t', value)
  path = res_dict['path_set']
  for agent in path:
    print("Agent {}'s path is ".format(agent), [p for p in list(zip(path[agent][0], path[agent][1]))], "at times",
          path[agent][2])
  
  # print(res_dict)

  # visualize_grid(grids, starts, targets, dests, res_dict['path_set'])
  # create_gif(grids, targets, dests, ac_dict, clusters=None, path=res_dict['path_set'])

  return 


if __name__ == '__main__':
  print("begin of main")

  # run_CBSS_MSMP()

  run_CBSS_MCPF()

  print("end of main")
