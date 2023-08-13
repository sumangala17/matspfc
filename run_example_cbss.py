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
import os
import random
import cbss_msmp
# import cbss_mcpf
import libmcpf.cbss_mcpf as cbss_mcpf

import matplotlib.animation as animation
from PIL import Image
import imageio

import common as cm

visited = set()

def run_CBSS_MSMP():
    """
    fully anonymous case, no assignment constraints.
    """
    print("------run_CBSS_MSMP------")
    ny = 10
    nx = 10
    grids = np.zeros((ny, nx))
    grids[5, 3:7] = 1  # obstacles

    starts = [11, 22, 33, 88, 99]
    targets = [40, 38, 27, 66, 72, 81, 83]
    dests = [19, 28, 37, 46, 69]

    configs = dict()
    configs["problem_str"] = "msmp"
    configs["mtsp_fea_check"] = 1
    configs["mtsp_atLeastOnce"] = 1
    # this determines whether the k-best TSP step will visit each node for at least once or exact once.
    configs["tsp_exe"] = "./pytspbridge/tsp_solver/LKH-2.0.10/LKH"
    configs["time_limit"] = 60
    configs["eps"] = 0.0
    res_dict = cbss_msmp.RunCbssMSMP(grids, starts, targets, dests, configs)

    print(res_dict)

    return


def get_gif():
    from moviepy.editor import ImageSequenceClip
    filenames = ['{}.png'.format(i) for i in range(26)]
    clip = ImageSequenceClip([imageio.imread(filename) for filename in filenames], fps=0.002)  # .resize(scale)
    clip.write_gif('clip2.gif', loop=5)

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

    for timestep in range(max_time):
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
    clip.write_gif('clip2.gif', loop=5)

    # with imageio.get_writer('mygif.gif', mode='I') as writer:
    #     for filename in filenames:
    #         image = imageio.imread(filename)
    #         writer.append_data(image)

    # Remove files
    # for filename in set(filenames):
    #     os.remove(filename)



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
            facecolor = cluster_colors[clusters[i]]
        rect = patches.Rectangle((0.25, 0.25), 0.4, height=0.4, linewidth=2,
                                 edgecolor=cluster_colors[clusters[i]], facecolor=facecolor)
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
    ny = 10
    nx = 10
    grids = np.zeros((ny, nx))
    grids[5, 3:7] = 1  # obstacles

    starts = [11, 22, 33, 88, 99]
    targets = [72, 81, 83, 40, 38, 27, 66]
    dests = [46, 69, 19, 28, 37]

    cluster_target_map = [0, 3, 1, 1, 0, 2, 3]

    print("SETUP AT START")
    # visualize_grid(grids, starts, targets, dests, ac_dict=None, clusters=cluster_target_map)

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
    print("Assignment constraints : ", ac_dict)

    configs = dict()
    configs["problem_str"] = "msmp"
    configs["mtsp_fea_check"] = 1
    configs["mtsp_atLeastOnce"] = 1
    # this determines whether the k-best TSP step will visit each node for at least once or exact once.
    configs["tsp_exe"] = "./pytspbridge/tsp_solver/LKH-2.0.10/LKH"
    configs["time_limit"] = 60
    configs["eps"] = 0.0

    res_dict = cbss_mcpf.RunCbssMCPF(grids, starts, targets, dests, cluster_target_map, ac_dict, configs)

    print(res_dict)

    path = res_dict['path_set']
    for agent in path:
        print("Agent {}'s path is ".format(agent), [p for p in list(zip(path[agent][0], path[agent][1]))], "at times",
              path[agent][2])

    # create_gif(grids, targets, dests, ac_dict, cluster_target_map, res_dict['path_set'])
    # get_gif()

    return


if __name__ == '__main__':
    print("begin of main")

    run_CBSS_MSMP()

    run_CBSS_MCPF()

    print("end of main")
