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


def test_ac(path, ac_dict, targets, num_agents):
  visited = [False for _ in range(len(targets))]
  agent_path = [[] for _ in range(num_agents)]
  i = 0
  for agent in path:
    agent_path[i] = [p for p in list(zip(path[agent][0], path[agent][1]))]
    i += 1
  t = 0
  for target in targets:
    txy = (target%256, int(target/256))
    allowed_ag = ac_dict[target] if target in ac_dict else np.arange(num_agents)
    for ag in allowed_ag:
      if txy in agent_path[ag]:
        visited[t] = True
        break
    t += 1

  assert sum(visited) == len(targets)



def run_CBSS_MCPF():
  """
  With assignment constraints.
  """

  if DATASET_GIVEN:
    starts, dests, targets, grids, cluster_target_map = get_world(num_agents=10, num_targets=20)
  else:
    ny = 10
    nx = 10
    starts = [11, 22, 33, 88, 99]
    targets = [72, 81, 83, 40, 38, 27, 66]
    dests = [46, 69, 19, 28, 37]
    grids = np.zeros((ny, nx))
    grids[5, 3:7] = 1  # obstacles
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

  print("Computing spMat")
  t1 = time.time()
  spMat = cm.getTargetGraph(grids, starts, targets, dests)
  print("A* cost mat done in time = ", time.time() - t1)

  print('_______________________________________________________________________________\n\n')
  print("CBSS Original")
  print('_______________________________________________________________________________\n\n')
  t1 = time.time()
  res_dict = cbss_mcpf.RunCbssMCPF(grids, starts, targets, dests, ac_dict, configs, copy.deepcopy(spMat))
  print("Time taken by CBSS = ", time.time() - t1)

  print('n_tsp_time \t best_g_value\t num_nodes_transformed_graph')
  print(res_dict['n_tsp_time'], '\t', res_dict['best_g_value'], '\t', res_dict['num_nodes_transformed_graph'])
  # for key, value in res_dict.items():
  #   if key in ['search_time', 'n_tsp_time','best_g_value','num_nodes_transformed_graph']:
  #     print(key, '\t=\t', value)
  path = res_dict['path_set']
  max_step = 0
  for agent in path:
    max_step = max(max_step, path[agent][2][-2])
  print("Max step = ", max_step)
  # for agent in path:
  #   print("Agent {}'s path is ".format(agent), [p for p in list(zip(path[agent][0], path[agent][1]))], "at times",
  #         path[agent][2])
  test_ac(path, ac_dict, targets, len(starts))

  print('_______________________________________________________________________________\n\n')
  print("CBSS Heuristic")
  print('_______________________________________________________________________________\n\n')

  print("AC DICT OLD", ac_dict)
  time_heuristic = time.time()
  ac_dict_H = heuristics.PathHeuristic(starts, targets, dests, ac_dict, grids, np.arange(len(targets)), copy.deepcopy(spMat)).get_updated_ac_dict()
  print("time taken by heuristic = ", time.time() - time_heuristic)
  print("AC DICT NEW", ac_dict_H)

  t1 = time.time()
  res_dict = cbss_mcpf.RunCbssMCPF(grids, starts, targets, dests, ac_dict_H, configs, spMat)
  print("Time taken by CBSS = ", time.time() - t1)

  print('n_tsp_time \t best_g_value\t num_nodes_transformed_graph')
  print(res_dict['n_tsp_time'], '\t', res_dict['best_g_value'], '\t', res_dict['num_nodes_transformed_graph'])
  # for key, value in res_dict.items():
  #   if key in ['search_time', 'n_tsp_time','best_g_value','num_nodes_transformed_graph']:
  #     print(key, '\t=\t', value)
  path = res_dict['path_set']
  max_step = 0
  for agent in path:
    max_step = max(max_step, path[agent][2][-2])
  print("Max step = ", max_step)
  # for agent in path:
  #   print("Agent {}'s path is ".format(agent), [p for p in list(zip(path[agent][0], path[agent][1]))], "at times",
  #         path[agent][2])

  test_ac(path, ac_dict, targets, len(starts))
  # print(res_dict)

  # visualize_grid(grids, starts, targets, dests, res_dict['path_set'])
  # create_gif(grids, targets, dests, ac_dict, clusters=None, path=res_dict['path_set'])

  return 


if __name__ == '__main__':
  print("begin of main")

  # run_CBSS_MSMP()

  run_CBSS_MCPF()

  print("end of main")
