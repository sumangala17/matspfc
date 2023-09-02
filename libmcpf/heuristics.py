import queue

import numpy as np
import libmcpf.common as cm

class PathHeuristic:
    def __init__(self, starts, targets, dests, ac_dict, grids, clusters, cost_matrix):
        self.N = len(starts)
        self.M = len(targets)

        self.starts = starts
        self.targets = targets
        self.dests = dests
        self.ac_dict = ac_dict
        self.grids = grids
        self.clusters = clusters
        self.cost_matrix = cm.getTargetGraph(grids, starts, targets, dests)

        self.agent_path = [[] for _ in range(self.N)]

    # def init_paths(self):
        # for i in range(self.num_agents):
        #     self.agent_path[i] = [self.starts[i], self.dests[i]]


    def find_delta_cost_after_adding_target(self, agent_id, v):
        # if target in self.ac_dict and agent_id not in self.ac_dict[target]:
        #     return -1
        v_in_spmat = self.N + v
        ag_dest_spmat = self.N + self.M + agent_id

        # empty list, just add target between start and destination
        if not self.agent_path[agent_id]:
            original_cost = self.cost_matrix[agent_id][ag_dest_spmat]
            new_cost = self.cost_matrix[agent_id][v_in_spmat] + self.cost_matrix[v_in_spmat][ag_dest_spmat]
            return new_cost - original_cost, 0

        min_dist = self.cost_matrix[agent_id][v_in_spmat]
        closest_point_on_path = agent_id
        closest_idx = -1

        # find the target in existing agent_path target list closest to the given target
        for i in range(len(self.agent_path[agent_id])):
            target_id = self.agent_path[agent_id][i]
            current_dist = self.cost_matrix[self.N + target_id][v_in_spmat]
            if current_dist < min_dist:
                min_dist = current_dist
                closest_point_on_path = self.N + target_id
                closest_idx = i

        # check if v to destination is closer than v to any target
        current_dist = self.cost_matrix[v_in_spmat][ag_dest_spmat]
        if current_dist < min_dist:
            original_cost = self.cost_matrix[self.N + self.agent_path[agent_id][-1]][ag_dest_spmat]
            new_cost = self.cost_matrix[self.N + self.agent_path[agent_id][-1]][v_in_spmat] + current_dist
            return new_cost - original_cost, -1

        # closest point is start point
        if closest_point_on_path == agent_id:
            original_cost = self.cost_matrix[agent_id][self.N + self.agent_path[agent_id][0]]
            new_cost = min_dist + self.cost_matrix[v_in_spmat][self.N + self.agent_path[agent_id][0]]
            return new_cost - original_cost, 0


        next_to_closest = self.N + self.agent_path[agent_id][closest_idx + 1]
        prev_to_closest = self.N + self.agent_path[agent_id][closest_idx - 1]
        delta_cost_with_next = (self.cost_matrix[closest_point_on_path][v_in_spmat] + self.cost_matrix[v_in_spmat][next_to_closest]
                                - self.cost_matrix[closest_point_on_path][next_to_closest])
        delta_cost_with_prev = (self.cost_matrix[prev_to_closest][v_in_spmat] + self.cost_matrix[v_in_spmat][closest_point_on_path]
                                - self.cost_matrix[prev_to_closest][closest_point_on_path])

        if delta_cost_with_next >= delta_cost_with_prev:
            return delta_cost_with_prev, closest_idx
        else:
            return delta_cost_with_next, closest_idx + 1



    # def get_agent_list_for_target(self, v):
    #     mincost = 0
    #     target_to_agent_path_cost = {}
    #     allowed_agents = np.arange(self.N) if v not in self.ac_dict else self.ac_dict[v]
    #     for i in range(self.N):
    #         if i not in allowed_agents:
    #             continue
    #         cost, idx = self.find_delta_cost_after_adding_target(i, v)
    #         target_to_agent_path_cost[(v, i)] = [cost, p1, p2]

    def work(self):
        # initialize static matrix
        target_agent_mat = np.ones(self.M, self.N) * -1
        for i in range(self.M):
            if self.targets[i] not in self.ac_dict:
                target_agent_mat[self.targets[i]] = 0
                continue
            for j in range(self.N):
                if j in self.ac_dict[self.targets[i]]:
                    target_agent_mat = 0

        coord_dict = {}
        # populate static matrix
        for i in range(self.N):
            for j in range(self.M):
                if target_agent_mat[i][j] == -1:
                    continue
                cost, index = self.find_delta_cost_after_adding_target(i, j)
                target_agent_mat[i][j] = cost
                coord_dict[(i, j)] = index

        visited_targets = set()
        while len(visited_targets) < self.M:
            # find the least cost target-agent pair and insert into agent_path for agent
            agent, target = np.unravel_index(target_agent_mat.argmin(), target_agent_mat.shape)
            self.agent_path[agent].insert(coord_dict[(agent, target)], target)
            visited_targets.add(target)
            # update the target-agent matrix for the given agent whose path was modified
            for j in range(self.M):
                if target_agent_mat[agent][j] == -1:
                    continue
                cost, index = self.find_delta_cost_after_adding_target(i, j)
                target_agent_mat[i][j] = cost
                coord_dict[(i, j)] = index





















