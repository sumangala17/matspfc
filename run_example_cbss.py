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

from dataset import get_world
from libmcpf import heuristics

visited = set()
DATASET_GIVEN = True

COLORS = [[0,0,1],[0,1,0],[1,0,0],
          [1,1,0],[1,0,1],[0,1,1],
          [0.25,0.75,0.75],[0.75,0.25,0.75],[0.75,0.75,0.25],[0.75,0.75,0.75],]

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
    configs["eps"] = 0.1
    res_dict = cbss_msmp.RunCbssMSMP(grids, starts, targets, dests, configs)

    print(res_dict)

    return


def gym_viz(grids, current_agent_positions, targets, dests, ac_dict, clusters, grid_geoms, grid_xform, grid_list):
    from viz import rendering
    sz = len(grids)
    a = 3
    viewer = rendering.Viewer(1050, 1050)

    render_geoms = []
    render_geoms_xform = []

    for agent in range(len(current_agent_positions)):
        geom = rendering.make_circle(radius=a)
        xform = rendering.Transform()
        geom.set_color(*COLORS[agent%10], alpha=1)
        geom.add_attr(xform)
        render_geoms.append(geom)
        render_geoms_xform.append(xform)

        point = current_agent_positions[agent]
        if point in visited:
            continue
        if point in targets and (point not in ac_dict or agent in ac_dict[point]):
            visited.add(point)
            point_idx = targets.index(point)
            for other_point_idx in range(len(targets)):
                if clusters[point_idx] == clusters[other_point_idx]:   # change color of other points in the visited cluster
                    visited.add(targets[other_point_idx])

    for target in range(len(targets)):
        geom = rendering.make_polygon(v=[(-a,-a),(-a,a),(a,a),(a,-a)])
        xform = rendering.Transform()
        if targets[target] in visited:
            geom.set_color(0.75,0.75,0.75, alpha=0.5)
        else:
            geom.set_color(*COLORS[clusters[target%10]])
        geom.add_attr(xform)
        render_geoms.append(geom)
        render_geoms_xform.append(xform)

    for agent in range(len(dests)):
        geom = rendering.make_polygon(v=[(-a,-a),(0,a),(a,-a)])
        xform = rendering.Transform()
        geom.set_color(*COLORS[agent%10], alpha=1)
        geom.add_attr(xform)
        render_geoms.append(geom)
        render_geoms_xform.append(xform)


    # extend list for grids
    render_geoms.extend(grid_geoms)
    render_geoms_xform.extend(grid_xform)


    # add geoms to viewer
    viewer.geoms = []
    for geom in render_geoms:
        viewer.add_geom(geom)

    results = []
    # update bounds to center around agent
    # cam_range = 1
    # pos = np.zeros(2)
    # viewer.set_bounds(-600, 600, -600, 600)
    # viewer.set_bounds(pos[0] - cam_range, pos[0] + cam_range, pos[1] - cam_range,
    #                            pos[1] + cam_range)
    # update geometry positions
    for i in range(len(current_agent_positions)):
        x, y = current_agent_positions[i] % sz, current_agent_positions[i] // sz
        render_geoms_xform[i].set_translation(x*2*a, y*2*a)
    for i in range(len(targets)):
        x, y = targets[i] % sz, targets[i] // sz
        render_geoms_xform[len(current_agent_positions) + i].set_translation(x*2*a, y*2*a)
    for i in range(len(dests)):
        x, y = dests[i] % sz, dests[i] // sz
        render_geoms_xform[len(current_agent_positions) + len(targets) + i].set_translation(x*2*a, y*2*a)
    k = len(current_agent_positions) + len(targets) + len(dests)
    c = 0
    for i,j in grid_list:
        render_geoms_xform[k + c].set_translation(j * 2*a, i * 2*a)
        c += 1
    # for i in range(len(grids)):
    #     for j in range(len(grids[0])):
    #         if grids[i][j] == 1:
    #             render_geoms_xform[k + c].set_translation(i * 4, j * 4)
    #             c += 1
    # render to display or array
    results.append(viewer.render(return_rgb_array=True))

    return np.squeeze(np.array(results))


def get_gif():
    from moviepy.editor import ImageSequenceClip
    filenames = ['{}.png'.format(i) for i in range(26)]
    clip = ImageSequenceClip([imageio.imread(filename) for filename in filenames], fps=0.002)  # .resize(scale)
    clip.write_gif('clip2.gif', loop=5)


def create_gif(grids, targets, dests, ac_dict, clusters, path):
    from moviepy.editor import ImageSequenceClip
    sz = len(grids)
    n = len(dests)  # number of robots
    max_time = 0
    for i in range(n):
        max_time = max(max_time, path[i][2][-2])

    agent_paths = []
    for i in range(n):
        agent_paths.append(list(zip(path[i][0], path[i][1])))

    # filenames = []
    gif_images = []

    num_obstacles = np.count_nonzero(grids)
    from viz import rendering
    a = 2
    grid_geom = []
    grid_xform = []
    for i in range(num_obstacles):
        geom = rendering.make_polygon(v=[(-a, -a), (-a, a), (a, a), (a, -a)])
        xform = rendering.Transform()
        geom.set_color(0, 0, 0, alpha=1)
        geom.add_attr(xform)
        grid_geom.append(geom)
        grid_xform.append(xform)

    grid_list = []
    for i in range(len(grids)):
        for j in range(len(grids[0])):
            if grids[i][j] == 1:
                grid_list.append((i,j))

    for timestep in range(max_time + 1):
        print("tick tock", timestep)
        current_agent_positions = []
        for i in range(n):
            if timestep < len(agent_paths[i]):
                x, y = agent_paths[i][timestep]
            else:
                x, y = agent_paths[i][-1]
            current_agent_positions.append(sz*y + x)

        # filename = f'{timestep}.png'
        # filenames.append(filename)
        # visualize_grid(grids, current_agent_positions, targets, dests, ac_dict, clusters, filename)
        results = gym_viz(grids, current_agent_positions, targets, dests, ac_dict, clusters, grid_geom, grid_xform, grid_list)
        gif_images.append(results)

    clip = ImageSequenceClip(list(gif_images), fps=0.004)
    clip.write_gif('gym_heuristic.gif')
    # build gif
    # clip = ImageSequenceClip([imageio.imread(filename) for filename in filenames], fps=0.002)  # .resize(scale)
    # clip.write_gif('loop.gif')#, loop=5)

    # with imageio.get_writer('mygif.gif', mode='I') as writer:
    #     for filename in filenames:
    #         image = imageio.imread(filename)
    #         writer.append_data(image)

    # Remove files
    # for filename in set(filenames):
    #     os.remove(filename)



def visualize_grid(grids, starts, targets, dests, ac_dict, clusters, filename=None):
    sz = len(grids)
    print("fine till here")
    fig, axs = plt.subplots(sz, sz)
    print("cool")
    plt.subplots_adjust(wspace=0, hspace=0)
    print("go")
    # plt.figure(figsize=(5, 5))
    # plt.imshow(grids)
    # if not clusters:
    #     clusters = np.arange(len(starts))

    colors = ['red', 'blue', 'yellow', 'green', 'turquoise']

    cluster_colors = ['grey', 'orange', 'pink', 'brown']

    print("draw starts")
    for agent_num in range(len(starts)):
        point = starts[agent_num]
        if point in targets and (point not in ac_dict or agent_num in ac_dict[point]):
            visited.add(point)
        x, y = int(point / sz), point % sz
        circle = patches.Circle((0.5, 0.5), 0.3, linewidth=2, edgecolor=colors[agent_num], facecolor=colors[agent_num])
        axs[sz - 1 - x, y].add_patch(circle)

    print("draw targets")
    for i in range(len(targets)):
        point = targets[i]
        x, y = int(point / sz), point % sz
        if point in visited:
            facecolor = 'white'
        else:
            facecolor = cluster_colors[clusters[i]%4]
        rect = patches.Rectangle((0.25, 0.25), 0.4, height=0.4, linewidth=2,
                                 edgecolor=cluster_colors[clusters[i]%4], facecolor=facecolor)
        axs[sz - 1 - x, y].add_patch(rect)

    points = np.where(grids == 1)
    # GRIDS (mark obstacles)
    print("draw obstacles")
    for i in range(len(points[0])):
        x, y = points[0][i], points[1][i]
        rect = patches.Rectangle((0, 0), 1, height=1, linewidth=2, edgecolor='black', facecolor='black')
        axs[sz - 1 - x, y].add_patch(rect)

    i = 0
    print("draw dests")
    for point in dests:
        x, y = int(point / sz), point % sz
        triangle = patches.RegularPolygon((0.5, 0.5), 3, radius=0.3, linewidth=2, edgecolor=colors[i],
                                          facecolor=colors[i])
        axs[sz - 1 - x, y].add_patch(triangle)
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

        cluster_target_map = np.arange(7)  # [0, 3, 1, 1, 0, 2, 3]

    print("SETUP AT START")
    # visualize_grid(grids, starts, targets, dests, ac_dict=None, clusters=cluster_target_map)

    ac_dict = dict()
    ri = 0
    for k in targets[:len(targets)//2]:
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
    configs["time_limit"] = 60 * 5
    configs["eps"] = 0.0

    spMat = None
    # print("AC DICT OLD", ac_dict)
    # time_heuristic = time.time()
    # ac_dict, spMat = heuristics.PathHeuristic(starts, targets, dests, ac_dict, grids, cluster_target_map).get_updated_ac_dict()
    # print("time taken by heuristic = ", time.time() - time_heuristic)
    # print("AC DICT NEW", ac_dict)

    time_cbss = time.time()
    res_dict = cbss_mcpf.RunCbssMCPF(grids, starts, targets, dests, cluster_target_map, ac_dict, configs, spMat)
    print("Time taken by CBSS = ", time.time() - time_cbss)

    # print("Therefore, total time = ", time_heuristic + time_cbss)

    for key, value in res_dict.items():
        if key in ['best_g_value', 'open_list_size', 'num_low_level_expanded', 'search_success', 'search_time', 'n_tsp_call', 'n_tsp_time', 'n_roots']:
            print(key, '\t=\t', value)

    path = res_dict['path_set']
    for agent in path:
        print("Agent {}'s path is ".format(agent), [p for p in list(zip(path[agent][0], path[agent][1]))], "at times",
              path[agent][2])


    # gym_viz(current_agent_positions=starts, grids=grids, targets=targets, dests=dests, ac_dict=ac_dict, clusters=cluster_target_map)
    create_gif(grids, targets, dests, ac_dict, cluster_target_map, res_dict['path_set'])
    # get_gif()

    return


if __name__ == '__main__':
    print("begin of main")

    # run_CBSS_MSMP()

    run_CBSS_MCPF()

    print("end of main")
