import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation
import distinctipy
import time
#import rospkg

matplotlib.rcParams.update({'font.size': 45})

visualize_trail = False

num_agents = 8
colors = np.array(distinctipy.get_colors(num_agents))
colors = np.concatenate((colors, np.ones((colors.shape[0], 1))), 1)

occupancy = np.load('occupancy.npy')
grid_width = occupancy.shape[0]
starts = np.load('starts.npy')
goals = np.load('goals.npy')
grid_path = np.load('grid_path.npy')

fig = plt.figure(figsize=(20, 15))

map_vis = np.zeros((occupancy.shape[0], occupancy.shape[1], 3), dtype=np.int64)
map_vis[:, :, 0] = 255*occupancy.astype(np.int64)

path_idx = 0

imshow_handle = plt.imshow(map_vis)

agent_scatter_handles = []
agent_path_handles = []
start_scatter_handles = []
goal_scatter_handles = []
ax = plt.gca()
ax.set_xlabel('x', labelpad=50)
ax.set_ylabel('y', labelpad=50, rotation=0)
ax.xaxis.set_tick_params(pad=50)
ax.yaxis.set_tick_params(pad=50)
plt.xticks([0, grid_width - 1])
plt.yticks([0, grid_width - 1])
plt.axis([-0.5, grid_width - 0.5, -0.5, grid_width - 0.5])
ax.set_title('Multi-Agent Path Finding with M*', y=1.05)
plt.subplots_adjust(bottom=0.25, top=0.9)
obstacle_handle = plt.scatter([], [], s=350, facecolors='red', marker='s', label='Obstacle')
free_handle = plt.scatter([], [], s=350, facecolors='black', marker='s', label='Free space')
for agent in range(num_agents):
  if agent == 0:
    agent_scatter_handles.append(plt.scatter([], [], s=350, facecolors=colors[agent], linewidths=4, label='Agent'))
    start_scatter_handles.append(plt.scatter([starts[agent, 1]], [starts[agent, 0]], s=350, edgecolors=colors[agent], facecolors=colors[agent], linewidths=2, marker=r'$S$', label='Start'))
    goal_scatter_handles.append(plt.scatter([goals[agent, 1]], [goals[agent, 0]], s=350, edgecolors=colors[agent], facecolors=colors[agent], linewidths=2, marker=r'$G$', label='Goal'))
  else:
    agent_scatter_handles.append(plt.scatter([], [], s=350, facecolors=colors[agent], linewidths=4))
    start_scatter_handles.append(plt.scatter([starts[agent, 1]], [starts[agent, 0]], s=350, edgecolors=colors[agent], facecolors=colors[agent], linewidths=2, marker=r'$S$'))
    goal_scatter_handles.append(plt.scatter([goals[agent, 1]], [goals[agent, 0]], s=350, edgecolors=colors[agent], facecolors=colors[agent], linewidths=2, marker=r'$G$'))

  if visualize_trail:
    if agent == 0:
      agent_path_handles.append(plt.plot(grid_path[:, num_agents + agent], grid_path[:, agent], linewidth=10, color=colors[agent], label='Path')[0])
    else:
      agent_path_handles.append(plt.plot(grid_path[:, num_agents + agent], grid_path[:, agent], linewidth=10, color=colors[agent])[0])

legend_handle = plt.legend(loc='upper right', bbox_to_anchor=(1.65, 1.0))
plt.draw()
# plt.show()

def update_animation(grid_idx):
  global imshow_handle, agent_scatter_handles, obstacle_handle, free_handle, legend_handle, agent_path_handles

  flat_coords = grid_path[grid_idx]
  for agent in range(num_agents):
    point = np.array([flat_coords[agent], flat_coords[num_agents + agent]])

    agent_scatter_handles[agent].set_offsets(np.flip(point).reshape(1, 2))

  return imshow_handle, *agent_scatter_handles, *start_scatter_handles, *goal_scatter_handles, obstacle_handle, free_handle, legend_handle, *agent_path_handles,

anim = animation.FuncAnimation(fig, update_animation, frames=grid_path.shape[0], interval=100, blit=True, repeat=False)
plt.show()
plt.close()
