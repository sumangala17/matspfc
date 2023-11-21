class Results:
    def __init__(self, ac_dict, eps, spMat, is_heuristic):
        self.ac_dict = ac_dict
        self.eps = eps
        self.spMat = spMat
        self.is_heuristic = is_heuristic

    def set_stats(self, total_time, ntsp, cost, agent_paths, target_assignment, cluster_target_selection, num_nodes,
                  max_step, num_conflicts):
        self.total_time = total_time
        self.ntsp = ntsp
        self.cost = cost
        self.agent_paths = agent_paths
        self.target_assignment = target_assignment
        self.cluster_target_selection = cluster_target_selection
        self.num_nodes = num_nodes
        self.max_step = max_step
        self.num_conflicts = num_conflicts



class Unit:
    def __init__(self, starts, targets, tgid, dests, clusters, grids, Astar_time):
        self.starts = starts
        self.targets = targets
        self.dests = dests
        self.clusters = clusters
        self.grids = grids
        self.Astar_time = Astar_time

        self.res_cbss_c = None
        self.res_hr: Results = None
        self.tg_nodes: list = tgid

    def set_results(self, cbss_c: Results, hr: Results, tgid):
        self.res_cbss_c = cbss_c
        self.res_hr: Results = hr
        self.tg_nodes: list = tgid