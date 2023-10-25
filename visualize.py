import os

import imageio
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

visited = set()


def create_gif(grids, targets, dests, ac_dict, clusters, path, name):
    from moviepy.editor import ImageSequenceClip
    global visited
    visited = set()
    sz = len(grids)
    n = len(dests)  # number of robots
    max_time = 0
    for i in range(n):
        max_time = max(max_time, path[i][2][-2])

    agent_paths = []
    for i in range(n):
        agent_paths.append(list(zip(path[i][0], path[i][1])))

    filenames = []
    gif_images = []

    num_obstacles = np.count_nonzero(grids)
    import viz.rendering as rendering
    # from .viz import rendering
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

        filename = f'{timestep}.png'
        filenames.append(filename)
        visualize_grid(grids, current_agent_positions, targets, dests, ac_dict, clusters, filename)
        # results = gym_viz(grids, current_agent_positions, targets, dests, ac_dict, clusters, grid_geom, grid_xform, grid_list)
        # gif_images.append(results)

    # clip = ImageSequenceClip(list(gif_images), fps=0.004)
    # clip.write_gif('gym_heuristic1.gif')
    # build gif
    clip = ImageSequenceClip([imageio.imread(filename) for filename in filenames], fps=0.002)  # .resize(scale)
    clip.write_gif('compare_{}.gif'.format(name), loop=5)

    # with imageio.get_writer('mygif.gif', mode='I') as writer:
    #     for filename in filenames:
    #         image = imageio.imread(filename)
    #         writer.append_data(image)

    # Remove files
    for filename in set(filenames):
        os.remove(filename)


def visualize_grid(grids, starts, targets, dests, ac_dict, clusters, filename=None):
    # sz = len(grids)
    sz = max(*grids.shape)
    fig, axs = plt.subplots(sz, sz)
    plt.subplots_adjust(wspace=0, hspace=0)
    # plt.figure(figsize=(5, 5))
    # plt.imshow(grids)
    # if not clusters:
    #     clusters = np.arange(len(starts))

    colors = ['red', 'blue', 'yellow', 'green', 'turquoise', 'black', 'maroon', 'violet', 'ochre', 'grey', 'orange',
              'pink', 'brown', 'magenta', 'grey']

    cluster_colors = ['grey', 'orange', 'pink', 'brown', 'magenta', 'yellow', 'green', 'turquoise', 'blue', 'black']

    # print("draw starts")
    for agent_num in range(len(starts)):
        point = starts[agent_num]
        if point in targets and (point not in ac_dict or agent_num in ac_dict[point]):
            visited.add(point)
        x, y = int(point / sz), point % sz
        circle = patches.Circle((0.5, 0.5), 0.3, linewidth=2, edgecolor=colors[agent_num], facecolor=colors[agent_num])
        axs[sz - 1 - x, y].add_patch(circle)

    # print("draw targets")
    for i in range(len(targets)):
        point = targets[i]
        x, y = int(point / sz), point % sz
        if point in visited:
            facecolor = 'white'
        elif clusters is None:
            facecolor = 'purple'
        else:
            facecolor = cluster_colors[clusters[i]%4]
        rect = patches.Rectangle((0.25, 0.25), 0.4, height=0.4, linewidth=2,
                                 edgecolor=cluster_colors[clusters[i]%4], facecolor=facecolor)
        axs[sz - 1 - x, y].add_patch(rect)

    points = np.where(grids == 1)
    # GRIDS (mark obstacles)
    # print("draw obstacles")
    for i in range(len(points[0])):
        x, y = points[0][i], points[1][i]
        rect = patches.Rectangle((0, 0), 1, height=1, linewidth=2, edgecolor='black', facecolor='black')
        axs[sz - 1 - x, y].add_patch(rect)

    i = 0
    # print("draw dests")
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