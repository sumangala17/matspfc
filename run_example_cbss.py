"""
Author: Zhongqiang (Richard) Ren
All Rights Reserved.
ABOUT: Entrypoint to the code.
Oeffentlich fuer: RSS22
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import context
import time
import numpy as np
import random
import cbss_msmp
import cbss_mcpf

import common as cm
import imageio

import dataset

visited = set()


def get_paths(res_path):
    agent_paths = {}
    for agent in res_path:
        agent_paths[agent] = [p for p in list(zip(res_path[agent][0], res_path[agent][1]))]
    return agent_paths


def test_ac(path, ac_dict, targets, clusters, num_agents, sz):
    # num_clusters = len(np.unique(clusters))
    # visited = [False for _ in range(num_clusters)]
    visited_T = [False for _ in range(len(targets))]
    agent_path = get_paths(path)
    t = 0
    agent_target_assignment = {}
    for i in range(len(targets)):
        target = targets[i]
        txy = (target%sz, int(target/sz))
        allowed_ag = ac_dict[target] if target in ac_dict else np.arange(num_agents)
        for ag in allowed_ag:
            if txy in agent_path[ag]:
                # visited[clusters[i]] = True
                visited_T[i] = True
                if ag in agent_target_assignment.keys():
                    agent_target_assignment[ag].append(target)
                else:
                    agent_target_assignment[ag] = []
                break
    t += 1
    if sum(visited_T) != len(targets):
        print("Agents and assigned targets: ", agent_target_assignment)
        print("Not all Visited!", visited_T)


def create_gif(grids, targets, dests, ac_dict, clusters, path):
    from moviepy.editor import ImageSequenceClip
    n = len(dests)  # number of robots
    max_time = 0
    for i in range(n):
        max_time = max(max_time, path[i][2][-2])

    agent_paths = []
    for i in range(n):
        agent_paths.append(list(zip(path[i][0], path[i][1])))

    filenames = []

    for timestep in range(max_time + 1):
        current_agent_positions = []
        for i in range(n):
            if timestep < len(agent_paths[i]):
                x, y = agent_paths[i][timestep]
            else:
                x, y = agent_paths[i][-1]
            current_agent_positions.append(10*y + x)

        filename = f'{timestep}.png'
        filenames.append(filename)
        visualize_grid(grids, current_agent_positions, targets, dests, ac_dict, clusters, filename)

    # build gif
    clip = ImageSequenceClip([imageio.imread(filename) for filename in filenames], fps=0.002)  # .resize(scale)
    clip.write_gif('loop_mcpf.gif')#, loop=5)



def visualize_grid(grids, starts, targets, dests, ac_dict, clusters, filename=None):
    fig, axs = plt.subplots(10, 10)
    plt.subplots_adjust(wspace=0, hspace=0)
    # plt.figure(figsize=(5, 5))
    # plt.imshow(grids)


    colors = ['red', 'blue', 'yellow', 'green', 'turquoise']

    cluster_colors = ['grey', 'orange', 'pink', 'brown']

    for agent_num in range(len(starts)):
        point = starts[agent_num]
        if point in targets and (point not in ac_dict or agent_num in ac_dict[point]):
            visited.add(point)
        x, y = int(point / 10), point % 10
        circle = patches.Circle((0.5, 0.5), 0.3, linewidth=2, edgecolor=colors[agent_num], facecolor=colors[agent_num])
        axs[9 - x, y].add_patch(circle)

    for i in range(len(targets)):
        point = targets[i]
        x, y = int(point / 10), point % 10
        if point in visited:
            facecolor = 'white'
        else:
            facecolor = 'purple'
        rect = patches.Rectangle((0.25, 0.25), 0.4, height=0.4, linewidth=2,
                                 edgecolor='purple', facecolor=facecolor)
        axs[9 - x, y].add_patch(rect)

    points = np.where(grids == 1)
    # GRIDS (mark obstacles)
    for i in range(len(points[0])):
        x, y = points[0][i], points[1][i]
        rect = patches.Rectangle((0, 0), 1, height=1, linewidth=2, edgecolor='black', facecolor='black')
        axs[9 - x, y].add_patch(rect)

    i = 0
    for point in dests:
        x, y = int(point / 10), point % 10
        triangle = patches.RegularPolygon((0.5, 0.5), 3, radius=0.3, linewidth=2, edgecolor=colors[i],
                                          facecolor=colors[i])
        axs[9 - x, y].add_patch(triangle)
        i += 1


    plt.setp(axs, xticks=[], yticks=[])
    # plt.tight_layout()
    # plt.gca().invert_yaxis()
    # plt.gca().invert_xaxis()
    if not filename:
        plt.show()
    else:
        # save frame
        plt.savefig(filename)
        plt.close()

def run_CBSS_MCPF():
  """
  With assignment constraints.
  """
  print("------run_CBSS_MCPF------")
  # ny = 10
  # nx = 10
  # grids = np.zeros((ny,nx))
  # grids[5,3:7] = 1 # obstacles

  grid_file = '/home/biorobotics/matspfc/datasets/maze-32-32-2.map'
  grid_file_new = '/home/biorobotics/matspfc/datasets/maze-32-32-2-binary.map'

  with open(grid_file) as file:
      for i in range(4):
          next(file)
      newText = file.read().replace('@', '1 ')
      newText = newText.replace('.', '0 ')
      newText = newText

  with open(grid_file_new, 'w') as file:
      file.write(newText)

  grids = np.loadtxt(grid_file_new)

  starts = [79, 613, 372, 555, 755]  # [11,22,33,88,99]
  targets = [854, 191, 417, 810, 528, 141, 95, 50, 607, 377, 74, 653, 741, 843, 650]  # [72,81,83,40,38,27,66]
  dests = [865, 654, 993, 323, 1006]  # [46,69,19,28,37]

  clusters = np.arange(len(targets))

  print("SETUP AT START")
  print("Starts: ", starts)
  print("Dests: ", dests)
  print("Targets: ", targets)
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
  configs["mtsp_fea_check"] = 0  # optional, help speed up K-best TSP module within CBSS for some cases.
  configs["tsp_exe"] = "./pytspbridge/tsp_solver/LKH-2.0.10/LKH"
  configs["time_limit"] = 12
  configs["eps"] = 0.0

  res_dict = cbss_mcpf.RunCbssMCPF(grids, starts, targets, dests, clusters, ac_dict, configs)

  print('n_tsp_time \t best_g_value\t num_nodes_transformed_graph')
  print(res_dict['n_tsp_time'], '\t', res_dict['best_g_value'], '\t', res_dict['num_nodes_transformed_graph'])

  test_ac(res_dict['path_set'], ac_dict, targets, range(len(targets)), len(starts), len(grids))

  # visualize_grid(grids, starts, targets, dests, res_dict['path_set'])
  # create_gif(grids, targets, dests, ac_dict, clusters=None, path=res_dict['path_set'])

  return 


if __name__ == '__main__':
  print("begin of main")

  # run_CBSS_MSMP()

  run_CBSS_MCPF()

  print("end of main")
