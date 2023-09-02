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
        if not self.agent_path[agent_id]:
            original_cost = self.cost_matrix[agent_id][ag_dest_spmat]
            new_cost = self.cost_matrix[agent_id][v_in_spmat] + self.cost_matrix[v_in_spmat][ag_dest_spmat]
            index = 1
            return new_cost - original_cost, index

        min_dist = self.cost_matrix[agent_id][v_in_spmat]
        closest_point_on_path = agent_id
        closest_idx = -1
        for i in range(len(self.agent_path[agent_id])):
            target_id = self.agent_path[agent_id][i]
            current_dist = self.cost_matrix[self.N + target_id][v_in_spmat]
            if current_dist < min_dist:
                min_dist = current_dist
                closest_point_on_path = self.N + target_id
                closest_idx = i
        current_dist = self.cost_matrix[v_in_spmat][ag_dest_spmat]  # v to destination
        if current_dist < min_dist:
            original_cost = self.cost_matrix[self.N + self.agent_path[agent_id][-1]][ag_dest_spmat]
            new_cost = self.cost_matrix[self.N + self.agent_path[agent_id][-1]][v_in_spmat] + current_dist
            index = -1
            return new_cost - original_cost, index

        if closest_point_on_path == agent_id:  # closest point is start point
            original_cost = self.cost_matrix[agent_id][self.N + self.agent_path[agent_id][0]]
            new_cost = min_dist + self.cost_matrix[v_in_spmat][self.N + self.agent_path[agent_id][0]]
            index = 1
            return new_cost - original_cost, index


        next_to_closest = self.N + self.agent_path[agent_id][closest_idx + 1]
        prev_to_closest = self.N + self.agent_path[agent_id][closest_idx - 1]
        delta_cost_with_next = (self.cost_matrix[closest_point_on_path][v_in_spmat] + self.cost_matrix[v_in_spmat][next_to_closest]
                                - self.cost_matrix[closest_point_on_path][next_to_closest])
        delta_cost_with_prev = (self.cost_matrix[prev_to_closest][v_in_spmat] + self.cost_matrix[v_in_spmat][closest_point_on_path]
                                - self.cost_matrix[prev_to_closest][closest_point_on_path])

        if delta_cost_with_next >= delta_cost_with_prev:
            return delta_cost_with_prev, (prev_to_closest, closest_idx)
        else:
            return delta_cost_with_next, (closest_idx, next_to_closest)
        


    def get_agent_list_for_target(self, v):
        mincost = 0
        allowed_agents = np.arange(self.N) if v not in self.ac_dict else self.ac_dict[v]
        for i in range(self.N):
            cost =

    def work(self):
        for v in self.targets:
            sorted_agents = self.get_agent_list_for_target()





















