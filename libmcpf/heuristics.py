import copy
import time

import numpy as np
import libmcpf.common as cm

INF_M = 10000000
BIG_M = 10000


class PathHeuristic:
    def __init__(self, starts, targets, dests, ac_dict, grids, clusters, spMat, alpha):
        self.N = len(starts)
        self.M = len(targets)

        self.starts = starts
        self.targets = targets
        self.dests = dests
        self.ac_dict = ac_dict
        self.grids = grids
        self.clusters = clusters
        # self.spMat = cm.getTargetGraph(grids, starts, targets, dests)
        self.cost_matrix = copy.deepcopy(spMat)
        self.alpha = alpha

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

        if closest_idx == len(self.agent_path[agent_id]) - 1:
            next_to_closest = self.N + self.M + agent_id  # self.agent_path[agent_id][-1]
        else:
            # print(closest_idx, len(self.agent_path[agent_id]))
            next_to_closest = self.N + self.agent_path[agent_id][closest_idx + 1]
        if closest_idx <= 0:
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


    def init_matrix(self):
        target_agent_mat = np.ones((self.M, self.N)) * INF_M
        for i in range(self.M):
            if self.targets[i] not in self.ac_dict:
                target_agent_mat[i, :] = np.zeros(self.N)
                continue
            for j in range(self.N):
                if j in self.ac_dict[self.targets[i]]:
                    target_agent_mat[i, j] = 0
        return target_agent_mat


    def run_target_assigment_step(self):
        # initialize static matrix
        target_agent_mat = self.init_matrix()

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

        while len(visited_targets) < self.M:
            # find the least cost target-agent pair and insert into agent_path for agent
            target, agent = np.unravel_index(temp_mat.argmin(), temp_mat.shape)
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

        return temp_mat# - BIG_M

    def reassign_targets(self):
        # initialize static matrix
        target_agent_mat = self.init_matrix()

        coord_dict = {}
        # populate static matrix
        for t_id in range(self.M):
            current_ag = -1
            t_index_in_path = -1
            for i in range(len(self.agent_path)):
                if t_id in self.agent_path[i]:
                    current_ag = i
                    t_index_in_path = self.agent_path[i].index(t_id)
                    break
            prev_node = t_index_in_path - 1 if t_index_in_path > 0 else current_ag
            next_node = t_index_in_path + 1 if t_index_in_path < len(self.agent_path[current_ag]) - 1 else self.N + self.M + current_ag
            for ag_id in range(self.N):
                if target_agent_mat[t_id][ag_id] == INF_M:
                    continue
                if ag_id == current_ag:
                    target_agent_mat[t_id][ag_id] = 0
                    coord_dict[(t_id, ag_id)] = t_index_in_path
                    continue
                cost, index = self.find_delta_cost_after_adding_target(ag_id, t_id)
                cost += self.cost_matrix[prev_node][next_node] - (self.cost_matrix[prev_node][self.N + t_id]
                                                                  + self.cost_matrix[self.N + t_id][next_node])
                target_agent_mat[t_id][ag_id] = cost
                coord_dict[(t_id, ag_id)] = index

        visited_targets = set()
        temp_mat = copy.deepcopy(target_agent_mat)

        x = 0

        while len(visited_targets) < self.M:
            # print("NUMBER VISITED", len(visited_targets),':', visited_targets,'\n', temp_mat)
            # find the least cost target-agent pair and insert into agent_path for agent
            target, agent = np.unravel_index(temp_mat.argmin(), temp_mat.shape)
            t_index_in_path = coord_dict[(target, agent)]

            for i in range(len(self.agent_path)):
                if target in self.agent_path[i]:
                    self.agent_path[i].remove(target)
                    break

            self.agent_path[agent].insert(t_index_in_path, target)

            # print("agent path of", agent," is :",  self.agent_path[agent])
            visited_targets.add(target)
            # update the target-agent matrix for the given agent whose path was modified
            for j in range(self.M):
                if temp_mat[j][agent] >= INF_M:
                    continue

                current_ag = -1
                t_index_in_path = -1
                for i in range(len(self.agent_path)):
                    if j in self.agent_path[i]:
                        current_ag = i
                        t_index_in_path = self.agent_path[i].index(j)
                        break
                # print("t index in path = ", t_index_in_path, "current agent", current_ag, "path length", len(self.agent_path[current_ag]))
                prev_node = t_index_in_path - 1 if t_index_in_path > 0 else current_ag
                next_node = t_index_in_path + 1 if t_index_in_path < len(
                    self.agent_path[current_ag]) - 1 else self.N + self.M + current_ag

                cost, index = self.find_delta_cost_after_adding_target(agent, j)
                cost += self.cost_matrix[prev_node][next_node] - (self.cost_matrix[prev_node][self.N + j]
                                                                  + self.cost_matrix[self.N + j][next_node])
                temp_mat[j][agent] = cost + BIG_M * (temp_mat[j][agent] >= BIG_M) * 2
                coord_dict[(j, agent)] = index

            temp_mat[target, :] += BIG_M * 2
            # x+=1
            # print('==============================', x)
            # if x>4:
            #     break

        return temp_mat  # - BIG_M

    def reduce_nodes_target(self, target_agent_mat):
        # print(target_agent_mat)

        alpha = self.alpha
        # print("alpha = ", alpha, "clusters = ", self.clusters)

        valid_num = target_agent_mat[target_agent_mat < INF_M]
        # print(valid_num)
        valid_num -= BIG_M * 2
        avg_cost = np.mean(valid_num)
        med_cost = np.median(valid_num)

        # print("global mean = ", avg_cost)

        target_agent_mat -= BIG_M * 2

        tc_alpha = np.zeros(self.M)
        tc_2 = np.zeros(self.M)
        tc_3 = np.zeros(self.M)
        for t_id in range(self.M):
            row = target_agent_mat[t_id]
            good_num = row[row < INF_M / 2]
            tc_alpha[t_id] = np.mean(good_num)
            tc_2[t_id] = np.median(good_num)
            tc_3[t_id] = np.min(good_num)

        # print("target mean = ", tc_alpha)

        if alpha < 1:
            cost_bound = alpha * tc_alpha + (1 - alpha) * med_cost
        elif alpha == 2:
            cost_bound = avg_cost
        elif alpha == 3:
            cost_bound = med_cost
        elif alpha == 4:
            cost_bound = np.clip(tc_2, a_max=med_cost, a_min=0)
        elif alpha == 5:
            cost_bound = (tc_3 + med_cost)/2
        elif alpha == 6:
            cost_bound = (tc_3 + avg_cost)/2
        elif alpha == 7:
            cost_bound = 100000


        for t_id in range(self.M):
            cb = cost_bound[t_id] if hasattr(cost_bound, "__len__") else cost_bound
            bad_pairs = np.argwhere(target_agent_mat[t_id] >= cb).reshape(-1)
            # print("Bad pairs", bad_pairs.shape)

            for i in range(len(bad_pairs)):  # target
                # for j in range(len(bad_pairs[0])):      # agent
                ag_id = bad_pairs[i]
                if self.targets[t_id] in self.ac_dict and ag_id not in self.ac_dict[self.targets[t_id]]:
                    continue
                # print(t_id, ag_id, self.targets[t_id],'---', self.ac_dict)
                if self.targets[t_id] not in self.ac_dict:
                    self.ac_dict[self.targets[t_id]] = set(np.arange(self.N))
                if len(self.ac_dict[self.targets[t_id]]) > 1:
                    self.ac_dict[self.targets[t_id]].remove(ag_id)


    def reduce_nodes_cluster(self, target_agent_mat):
        # print(target_agent_mat)

        alpha = self.alpha
        # print("alpha = ", alpha, "clusters = ", self.clusters)

        valid_num = target_agent_mat[target_agent_mat < INF_M]
        # print(valid_num)
        valid_num -= BIG_M * 2
        avg_cost = np.median(valid_num)

        # print("global mean = ", avg_cost)

        target_agent_mat -= BIG_M * 2

        cc = np.zeros(np.max(self.clusters) + 1)
        cnt = np.zeros_like(cc)
        for t_id in range(self.M):
            for ag_id in range(self.N):
                if target_agent_mat[t_id][ag_id] < INF_M/2:
                    cid = self.clusters[t_id]
                    cc[cid] += target_agent_mat[t_id][ag_id]
                    cnt[cid] += 1

        cc = np.divide(cc, cnt)

        # print("cluster mean = ", cc)

        cost_bound = alpha * cc + (1 - alpha) * avg_cost

        for t_id in range(self.M):
            cid = self.clusters[t_id]
            bad_pairs = np.argwhere(target_agent_mat[t_id] >= cost_bound[cid]).reshape(-1)
            # print("Bad pairs", bad_pairs, bad_pairs.shape)

            for i in range(len(bad_pairs)):  # target
                # for j in range(len(bad_pairs[0])):      # agent
                ag_id = bad_pairs[i]
                if self.targets[t_id] in self.ac_dict and ag_id not in self.ac_dict[self.targets[t_id]]:
                    continue
                # print(t_id, ag_id, self.targets[t_id],'---', self.ac_dict)
                if self.targets[t_id] not in self.ac_dict:
                    self.ac_dict[self.targets[t_id]] = set(np.arange(self.N))
                if len(self.ac_dict[self.targets[t_id]]) > 1:
                    self.ac_dict[self.targets[t_id]].remove(ag_id)


    def run_reduce_nodes_step(self, target_agent_mat):
        p = 0.2
        valid_num = target_agent_mat[target_agent_mat < INF_M]
        print(valid_num)
        valid_num -= BIG_M*2
        avg_cost = np.mean(valid_num)
        std_cost = np.std(valid_num)
        # cost_bound = (1-p) * avg_cost + p * std_cost
        cost_bound = avg_cost - std_cost

        target_agent_mat -= BIG_M * 2

        # bad_pairs = np.argwhere(target_agent_mat >= cost_bound)
        bad_pairs = np.argwhere(target_agent_mat >= cost_bound)
        print(target_agent_mat, target_agent_mat.shape,"Bad pairs", bad_pairs, bad_pairs.shape)

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
                self.ac_dict[self.targets[t_id]] = set(np.arange(self.N))
            if len(self.ac_dict[self.targets[t_id]]) > 1:
                self.ac_dict[self.targets[t_id]].remove(ag_id)

    def get_updated_ac_dict(self):
        # print("Welcome to Heuristic Park!")
        np.set_printoptions(suppress=True)
        # t1 = time.time()
        target_agent_mat = self.run_target_assigment_step()
        # print(target_agent_mat)
        # for _ in range(5):
        #     target_agent_mat = self.reassign_targets()
        #     print("new iter\n", target_agent_mat)
        # self.run_reduce_nodes_step(target_agent_mat)
        # self.reduce_nodes_cluster(target_agent_mat)
        self.reduce_nodes_target(target_agent_mat)
        # print("Exiting Heuristic park after ", time.time() - t1)
        return self.ac_dict#, self.spMat