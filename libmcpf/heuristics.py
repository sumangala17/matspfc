import copy

import numpy as np
import libmcpf.common as cm

INF_M = 10000000
BIG_M = 10000

class PathHeuristic:
    def __init__(self, starts, targets, dests, ac_dict, grids, clusters):
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

        target_id_list = np.array(self.agent_path[agent_id]) + self.N
        dist_list = self.cost_matrix[target_id_list, v_in_spmat]
        # print("dl", dist_list, "and target id list is", target_id_list)
        # min_dist1 = np.min(dist_list)
        closest_idx_temp = np.argmin(dist_list)
        min_dist1_temp = dist_list[closest_idx_temp]
        if min_dist1_temp < min_dist:
            min_dist = min_dist1_temp
            closest_idx = closest_idx_temp
            closest_point_on_path = target_id_list[closest_idx]
        # print("mindist = ", min_dist, "closest target = ", closest_point_on_path)

        # min_dist = self.cost_matrix[agent_id][v_in_spmat]
        # closest_point_on_path = agent_id
        # closest_idx = -1
        # # find the target in existing agent_path target list closest to the given target
        # for i in range(len(self.agent_path[agent_id])):
        #     target_id = self.agent_path[agent_id][i]
        #     print("we are considering target", self.N + target_id, "which should be present in target id list above")
        #     current_dist = self.cost_matrix[self.N + target_id][v_in_spmat]
        #     if current_dist < min_dist:
        #         min_dist = current_dist
        #         closest_point_on_path = self.N + target_id
        #         closest_idx = i
        #         print("updating loop mindist..", min_dist, closest_point_on_path)
        #
        # print("mindist = ", min_dist, "closest target = ", closest_point_on_path)
        # print("equal?", min_dist==min_dist1, closest_point_on_path==closest_point_on_path1)

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

        if closest_point_on_path >= len(self.agent_path[agent_id]):
            next_to_closest = self.N + self.agent_path[agent_id][-1]
        else:
            next_to_closest = self.N + self.agent_path[agent_id][closest_idx + 1]
        if closest_point_on_path <= 0:
            prev_to_closest = agent_id
        else:
            prev_to_closest = self.N + self.agent_path[agent_id][closest_idx - 1]
        delta_cost_with_next = (self.cost_matrix[closest_point_on_path][v_in_spmat] + self.cost_matrix[v_in_spmat][next_to_closest]
                                - self.cost_matrix[closest_point_on_path][next_to_closest])
        delta_cost_with_prev = (self.cost_matrix[prev_to_closest][v_in_spmat] + self.cost_matrix[v_in_spmat][closest_point_on_path]
                                - self.cost_matrix[prev_to_closest][closest_point_on_path])

        if delta_cost_with_next >= delta_cost_with_prev:
            return delta_cost_with_prev, closest_idx
        else:
            return delta_cost_with_next, closest_idx + 1

    def run_target_assigment_step(self):
        # initialize static matrix
        target_agent_mat = np.ones((self.M, self.N)) * INF_M
        for i in range(self.M):
            if self.targets[i] not in self.ac_dict:
                target_agent_mat[i, :] = np.zeros(self.N)
                continue
            for j in range(self.N):
                if j in self.ac_dict[self.targets[i]]:
                    target_agent_mat[i, j] = 0

        coord_dict = {}
        # populate static matrix
        for t_id in range(self.M):
            for ag_id in range(self.N):
                if target_agent_mat[t_id][ag_id] == INF_M:
                    continue
                cost, index = self.find_delta_cost_after_adding_target(ag_id, t_id)
                target_agent_mat[t_id][ag_id] = cost
                coord_dict[(t_id, ag_id)] = index

        visited_targets = set()
        temp_mat = copy.deepcopy(target_agent_mat)
        # print(temp_mat)
        # i=0
        while len(visited_targets) < self.M:
            # print("VIsited targets:", visited_targets)
            # find the least cost target-agent pair and insert into agent_path for agent
            target, agent = np.unravel_index(temp_mat.argmin(), temp_mat.shape)
            # print(target, "++")
            # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            # print(temp_mat)
            # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            self.agent_path[agent].insert(coord_dict[(target, agent)], target)
            # print("agent path of", agent," is :",  self.agent_path[agent])
            visited_targets.add(target)
            # update the target-agent matrix for the given agent whose path was modified
            for j in range(self.M):
                if temp_mat[j][agent] >= INF_M:
                    continue
                cost, index = self.find_delta_cost_after_adding_target(agent, j)
                temp_mat[j][agent] = cost + BIG_M * (temp_mat[j][agent] >= BIG_M)
                coord_dict[(j, agent)] = index

            temp_mat[target, :] += BIG_M
            # i += 1
            # if i>6:
            #     break

        return temp_mat# - BIG_M

    def run_reduce_nodes_step(self, target_agent_mat):
        valid_num = target_agent_mat[target_agent_mat < INF_M]
        valid_num -= BIG_M
        avg_cost = np.mean(valid_num)
        # std_cost = np.std(valid_num)
        cost_bound = avg_cost #+ 0.2 * std_cost

        target_agent_mat -= BIG_M

        # copy_mat = copy.deepcopy(target_agent_mat)
        bad_pairs = np.argwhere(target_agent_mat >= cost_bound)
        # print(target_agent_mat, target_agent_mat.shape,"Bad pairs", bad_pairs, bad_pairs.shape)

        # for t_id, ag_id in bad_pairs:
        #     if self.targets[t_id] not in self.ac_dict:
        #         self.ac_dict[self.targets[t_id]] = list(np.arange(self.N))
        #     self.ac_dict[self.targets[t_id]].remove(bad_pairs[t_id, ag_id])

        for i in range(len(bad_pairs)):  # target
            # for j in range(len(bad_pairs[0])):      # agent
            t_id, ag_id = bad_pairs[i]
            if self.targets[t_id] in self.ac_dict and ag_id not in self.ac_dict[self.targets[t_id]]:
                continue
            # print(t_id, ag_id, self.targets[t_id],'---', self.ac_dict)
            if self.targets[t_id] not in self.ac_dict:
                self.ac_dict[self.targets[t_id]] = list(np.arange(self.N))
            self.ac_dict[self.targets[t_id]].remove(ag_id)

    def get_updated_ac_dict(self):
        target_agent_mat = self.run_target_assigment_step()
        self.run_reduce_nodes_step(target_agent_mat)
        return self.ac_dict









