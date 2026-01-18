import numpy as np
from RRTTree import RRTTree
import time

'''
class RRTStarPlanner(object):

    def __init__(
        self,
        bb,
        ext_mode,
        max_step_size,
        start,
        goal,
        max_itr=None,
        stop_on_goal=None,
        k=None,
        goal_prob=0.01,
    ):

        # set environment and search tree
        self.bb = bb
        self.tree = RRTTree(self.bb)
        self.start = start
        self.goal = goal

        self.max_itr = max_itr
        self.stop_on_goal = stop_on_goal

        # set search params
        self.ext_mode = ext_mode
        self.goal_prob = goal_prob
        self.k = k

        self.max_step_size = max_step_size


    def plan(self):
        """
        Compute and return the plan. The function should return a numpy array containing the states
        (positions) of the robot.
        """
        t0 = time.time()

        # sensible defaults
        max_itr = 50000 if self.max_itr is None else int(self.max_itr)
        stop_on_goal = True if self.stop_on_goal is None else bool(self.stop_on_goal)

        # init tree with start
        if len(self.tree.vertices) == 0:
            self.tree.add_vertex(self.start)

        # helper: rebuild path from some vertex id to root
        def backtrack_path(goal_vid):
            path = []
            vid = goal_vid
            while True:
                path.append(self.tree.vertices[vid].config)
                if vid == self.tree.get_root_id():
                    break
                vid = self.tree.edges[vid]
            path.reverse()
            return np.array(path)

        # helper: after rewiring a node, update costs down its subtree
        def propagate_costs_from(parent_vid):
            # find children by scanning edges (edges: child -> parent)
            for child_vid, p_vid in list(self.tree.edges.items()):
                if p_vid == parent_vid:
                    p_conf = self.tree.vertices[parent_vid].config
                    c_conf = self.tree.vertices[child_vid].config
                    edge_cost = self.bb.compute_distance(p_conf, c_conf)
                    self.tree.vertices[child_vid].set_cost(self.tree.vertices[parent_vid].cost + edge_cost)
                    propagate_costs_from(child_vid)

        best_goal_vid = None
        best_goal_cost = np.inf

        print(f"DEBUG max_itr is {max_itr}")
        for iii in range(max_itr):
            if iii%200 == 0:
                print(f"current iteration is {iii}")
            # sample (with goal bias)
            x_rand = self.bb.sample_random_config(self.goal_prob, self.goal)

            # nearest
            x_near_id, x_near = self.tree.get_nearest_config(x_rand)

            # extend
            x_new = self.extend(x_near, x_rand)
            if x_new is None:
                continue

            # skip invalid configuration
            if not self.bb.config_validity_checker(x_new):
                continue

            # avoid duplicates (exact equality check)
            if self.tree.get_idx_for_config(x_new) is not None:
                continue

            # choose k
            n = len(self.tree.vertices)
            if n == 0:
                continue
            if self.k is None:
                # classic-ish: k ~ c * log(n)
                k = int(max(1, np.ceil(10 * np.log(n + 1))))
                k = 1
                d = 6
                k = int(np.ceil(np.exp(1+(1/d))*np.log(n+1)))

                k = 3 # only working

            else:
                k = int(self.k)

            # get neighbor set robustly (tree.get_k_nearest_neighbors has argpartition(k) quirks)
            if n == 1:
                nn_ids = [0]
            else:
                k_eff = max(1, min(k, n - 1))
                nn_ids, _ = self.tree.get_k_nearest_neighbors(x_new, k_eff)

            # make sure nearest is included
            if x_near_id not in nn_ids:
                nn_ids.append(x_near_id)

            # ---- choose best parent among neighbors (min cost-to-come + edge) ----
            best_parent = None
            best_cost_to_new = np.inf

            for vid in nn_ids:
                vconf = self.tree.vertices[vid].config
                if not self.bb.edge_validity_checker(vconf, x_new):
                    continue
                cand = self.tree.vertices[vid].cost + self.bb.compute_distance(vconf, x_new)
                if cand < best_cost_to_new:
                    best_cost_to_new = cand
                    best_parent = vid

            if best_parent is None:
                continue  # no valid connection

            # add new vertex + edge from best parent
            new_vid = self.tree.add_vertex(x_new)
            parent_conf = self.tree.vertices[best_parent].config
            self.tree.add_edge(best_parent, new_vid, edge_cost=self.bb.compute_distance(parent_conf, x_new))

            # ---- rewire neighbors through x_new if it improves their cost ----
            for vid in nn_ids:
                if vid == best_parent:
                    continue
                vconf = self.tree.vertices[vid].config
                if not self.bb.edge_validity_checker(x_new, vconf):
                    continue

                new_cost_via_new = self.tree.vertices[new_vid].cost + self.bb.compute_distance(x_new, vconf)
                if new_cost_via_new + 1e-12 < self.tree.vertices[vid].cost:
                    # rewire vid to have parent = new_vid
                    self.tree.edges[vid] = new_vid
                    self.tree.vertices[vid].set_cost(new_cost_via_new)
                    propagate_costs_from(vid)

            # ---- track best goal ----
            goal_vid = self.tree.get_idx_for_config(self.goal)
            if goal_vid is not None:
                goal_cost = self.tree.vertices[goal_vid].cost
                if goal_cost < best_goal_cost:
                    best_goal_cost = goal_cost
                    best_goal_vid = goal_vid

                if stop_on_goal:
                    break

        # return best found path to goal if exists
        if best_goal_vid is None:
            # maybe goal exists but we didn't update (unlikely); check once
            goal_vid = self.tree.get_idx_for_config(self.goal)
            if goal_vid is None:
                return None
            best_goal_vid = goal_vid

        return backtrack_path(best_goal_vid)


    def compute_cost(self, plan):
        """
        Compute the cost of the given path as the sum of distances between consecutive configurations.
        """
        if plan is None or len(plan) < 2:
            return 0.0

        cost = 0.0
        for i in range(len(plan) - 1):
            cost += float(self.bb.compute_distance(plan[i], plan[i + 1]))
        return cost


    def extend(self, x_near, x_rand):
        """
        Extend from x_near toward x_rand using:
          E1: go all the way to x_rand
          E2: move at most max_step_size toward x_rand (in Euclidean config-space)
        """
        if self.ext_mode == "E1":
            return np.array(x_rand, dtype=float)

        if self.ext_mode == "E2":
            x_near = np.array(x_near, dtype=float)
            x_rand = np.array(x_rand, dtype=float)

            delta = x_rand - x_near
            dist = float(np.linalg.norm(delta))

            if dist < 1e-12:
                return None

            if dist <= self.max_step_size:
                return x_rand

            step = (self.max_step_size / dist) * delta
            return x_near + step

        raise ValueError(f"Unknown ext_mode: {self.ext_mode}")
'''

import numpy as np
from RRTTree import RRTTree
import time
from collections import defaultdict


class RRTStarPlanner(object):

    def __init__(
            self,
            bb,
            ext_mode,
            max_step_size,
            start,
            goal,
            max_itr=None,
            stop_on_goal=None,
            k=None,
            goal_prob=0.01,
    ):
        self.bb = bb
        self.tree = RRTTree(self.bb)
        self.start = start
        self.goal = goal
        self.max_itr = max_itr
        self.stop_on_goal = stop_on_goal
        self.ext_mode = ext_mode
        self.goal_prob = goal_prob
        self.k = k
        self.max_step_size = max_step_size

        # --- FIX 1: Add Adjacency List ---
        # Maps parent_vid -> list of child_vids
        # This allows O(1) access instead of O(N) scanning
        self.children = defaultdict(list)

    def plan(self):
        """
        Compute and return the plan.
        """
        # sensible defaults
        max_itr = 50000 if self.max_itr is None else int(self.max_itr)
        stop_on_goal = True if self.stop_on_goal is None else bool(self.stop_on_goal)

        # init tree with start
        if len(self.tree.vertices) == 0:
            self.tree.add_vertex(self.start)

        # helper: rebuild path
        def backtrack_path(goal_vid):
            path = []
            vid = goal_vid
            while True:
                path.append(self.tree.vertices[vid].config)
                if vid == self.tree.get_root_id():
                    break
                vid = self.tree.edges[vid]
            path.reverse()
            return np.array(path)

        # --- FIX 2: Optimized Cost Propagation ---
        def propagate_costs_from(parent_vid):
            # If parent has no children, stop.
            if parent_vid not in self.children:
                return

            parent_cost = self.tree.vertices[parent_vid].cost

            # Iterate ONLY over direct children (O(number of children) vs O(Tree Size))
            for child_vid in self.children[parent_vid]:
                p_conf = self.tree.vertices[parent_vid].config
                c_conf = self.tree.vertices[child_vid].config

                # Recompute distance or store it? Recomputing is safer for now.
                edge_cost = self.bb.compute_distance(p_conf, c_conf)

                new_child_cost = parent_cost + edge_cost
                self.tree.vertices[child_vid].set_cost(new_child_cost)

                # Recurse
                propagate_costs_from(child_vid)

        best_goal_vid = None
        best_goal_cost = np.inf

        print(f"DEBUG: Starting RRT* with max_itr={max_itr}")

        for iii in range(max_itr):
            if iii % 200 == 0:
                print(f"Iteration: {iii}, Tree Size: {len(self.tree.vertices)}")

            # 1. Sample
            x_rand = self.bb.sample_random_config(self.goal_prob, self.goal)

            # 2. Nearest
            x_near_id, x_near = self.tree.get_nearest_config(x_rand)

            # 3. Extend
            x_new = self.extend(x_near, x_rand)
            if x_new is None: continue

            # 4. Validity (The CPU Bottleneck)
            if not self.bb.config_validity_checker(x_new): continue
            if self.tree.get_idx_for_config(x_new) is not None: continue

            # --- FIX 3: Proper K Calculation ---
            n = len(self.tree.vertices)
            if self.k is None:
                # RRT* Formula: k = c * log(n)
                # For 6-DOF, we need a slightly higher constant to ensure connectivity
                dd = 6
                const = np.exp(1+(1/dd))
                k = int(np.ceil(const* np.log(n + 1)))
                # k= 3
                k = max(k, 3)  # Ensure at least 3 neighbors
            else:
                k = int(self.k)

            # 5. Get Neighbors
            k_eff = min(k, n - 1)
            nn_ids, _ = self.tree.get_k_nearest_neighbors(x_new, k_eff)

            # Ensure nearest node is in the list
            if x_near_id not in nn_ids:
                nn_ids.append(x_near_id)

            # 6. Choose Best Parent
            best_parent = None
            min_cost_to_new = np.inf

            valid_neighbors = []  # Cache valid neighbors to avoid re-checking

            for vid in nn_ids:
                vconf = self.tree.vertices[vid].config

                # Check collision (Expensive!)
                if not self.bb.edge_validity_checker(vconf, x_new):
                    continue

                valid_neighbors.append(vid)

                cost = self.tree.vertices[vid].cost + self.bb.compute_distance(vconf, x_new)
                if cost < min_cost_to_new:
                    min_cost_to_new = cost
                    best_parent = vid

            if best_parent is None:
                continue

            # 7. Add to Tree
            new_vid = self.tree.add_vertex(x_new)
            self.tree.add_edge(best_parent, new_vid, min_cost_to_new - self.tree.vertices[best_parent].cost)
            self.tree.vertices[new_vid].set_cost(min_cost_to_new)

            # Update Topology
            self.children[best_parent].append(new_vid)

            # 8. Rewire (RRT* Magic)
            for vid in valid_neighbors:
                if vid == best_parent: continue

                # Cost if we go: Root -> ... -> x_new -> vid
                dist_new_to_neighbor = self.bb.compute_distance(x_new, self.tree.vertices[vid].config)
                new_cost_via_new = min_cost_to_new + dist_new_to_neighbor

                # Check if this path is cheaper
                if new_cost_via_new + 1e-6 < self.tree.vertices[vid].cost:
                    # REWIRE!

                    # A. Identify old parent
                    old_parent = self.tree.edges[vid]

                    # B. Update structural edges in Tree
                    self.tree.edges[vid] = new_vid

                    # C. Update our 'children' map (Topology maintenance)
                    if old_parent in self.children and vid in self.children[old_parent]:
                        self.children[old_parent].remove(vid)
                    self.children[new_vid].append(vid)

                    # D. Update Cost and Propagate
                    self.tree.vertices[vid].set_cost(new_cost_via_new)
                    propagate_costs_from(vid)

            # --- FIX 4: Robust Goal Detection ---
            # Do NOT check for exact equality. Use a distance threshold.
            dist_to_goal = self.bb.compute_distance(x_new, self.goal)

            # If we are close enough (e.g. 0.1 or 0.25 units), we found it.
            if dist_to_goal < 0.1:
                # Check if this specific path to goal is the best so far
                if best_goal_vid is None or min_cost_to_new < best_goal_cost:
                    best_goal_cost = min_cost_to_new
                    best_goal_vid = new_vid
                    print(f"Goal Reached! Cost: {best_goal_cost:.4f}")

                    if stop_on_goal:
                        break

        # Reconstruct path
        if best_goal_vid is None:
            return None

        return backtrack_path(best_goal_vid)

    def extend(self, x_near, x_rand):
        if self.ext_mode == "E1":
            return np.array(x_rand)
        elif self.ext_mode == "E2":
            x_near = np.array(x_near)
            x_rand = np.array(x_rand)
            delta = x_rand - x_near
            dist = np.linalg.norm(delta)

            if dist < 1e-6: return None
            if dist <= self.max_step_size: return x_rand

            return x_near + (self.max_step_size / dist) * delta
        return None

    def compute_cost(self, plan):
        if plan is None or len(plan) < 2: return 0.0
        cost = 0.0
        for i in range(len(plan) - 1):
            cost += float(self.bb.compute_distance(plan[i], plan[i + 1]))
        return cost



# If you are pasting this into the same file, you don't need the import above.

class RRTStarExperiment_old(RRTStarPlanner):
    """
    A subclass of RRTStarPlanner designed for data collection.
    It runs for exactly max_itr and records stats at specific intervals.
    """

    def plan_with_stats(self, report_interval=400):
        # Setup defaults
        max_itr = 2000 if self.max_itr is None else int(self.max_itr)

        # Init tree
        if len(self.tree.vertices) == 0:
            self.tree.add_vertex(self.start)

        # ---------------------------------------------------------
        # Helper: Optimized Cost Propagation (Same as before)
        # ---------------------------------------------------------
        def propagate_costs_from(parent_vid):
            if parent_vid not in self.children: return
            parent_cost = self.tree.vertices[parent_vid].cost
            for child_vid in self.children[parent_vid]:
                p_conf = self.tree.vertices[parent_vid].config
                c_conf = self.tree.vertices[child_vid].config
                new_cost = parent_cost + self.bb.compute_distance(p_conf, c_conf)
                self.tree.vertices[child_vid].set_cost(new_cost)
                propagate_costs_from(child_vid)

        # ---------------------------------------------------------

        best_goal_vid = None
        best_goal_cost = np.inf

        # Data containers for the assignment
        history_success = []  # e.g. [0, 0, 1, 1, 1]
        history_cost = []  # e.g. [None, None, 12.5, 10.2, 9.8]

        for iii in range(1, max_itr + 1):

            # --- RRT* LOGIC START ---
            x_rand = self.bb.sample_random_config(self.goal_prob, self.goal)
            x_near_id, x_near = self.tree.get_nearest_config(x_rand)
            x_new = self.extend(x_near, x_rand)

            if x_new is not None and \
                    self.bb.config_validity_checker(x_new) and \
                    self.tree.get_idx_for_config(x_new) is None:

                # Dynamic K
                n = len(self.tree.vertices)
                # k = int(self.k) if self.k is not None else int(np.ceil(2.0 * np.log(n + 1)))
                dd = 6
                const = np.exp(1+(1/dd))
                k = int(np.ceil(const* np.log(n + 1)))
                # k = max(k, 3)
                # k = 3
                k_eff = min(k, n - 1)

                nn_ids, _ = self.tree.get_k_nearest_neighbors(x_new, k_eff)
                if x_near_id not in nn_ids: nn_ids.append(x_near_id)

                # Choose Parent
                best_parent = None
                min_cost = np.inf
                valid_neighbors = []

                for vid in nn_ids:
                    vconf = self.tree.vertices[vid].config
                    if self.bb.edge_validity_checker(vconf, x_new):
                        valid_neighbors.append(vid)
                        cost = self.tree.vertices[vid].cost + self.bb.compute_distance(vconf, x_new)
                        if cost < min_cost:
                            min_cost = cost
                            best_parent = vid

                if best_parent is not None:
                    new_vid = self.tree.add_vertex(x_new)
                    self.tree.add_edge(best_parent, new_vid, min_cost - self.tree.vertices[best_parent].cost)
                    self.tree.vertices[new_vid].set_cost(min_cost)
                    self.children[best_parent].append(new_vid)

                    # Rewire
                    for vid in valid_neighbors:
                        if vid == best_parent: continue
                        dist = self.bb.compute_distance(x_new, self.tree.vertices[vid].config)
                        new_cost = min_cost + dist
                        if new_cost + 1e-6 < self.tree.vertices[vid].cost:
                            old_parent = self.tree.edges[vid]
                            if old_parent in self.children and vid in self.children[old_parent]:
                                self.children[old_parent].remove(vid)
                            self.tree.edges[vid] = new_vid
                            self.children[new_vid].append(vid)
                            self.tree.vertices[vid].set_cost(new_cost)
                            propagate_costs_from(vid)

                    # Check Goal (Update best found so far)
                    dist_to_goal = self.bb.compute_distance(x_new, self.goal)
                    if dist_to_goal < 0.25:  # Goal Threshold
                        if best_goal_vid is None or min_cost < best_goal_cost:
                            best_goal_cost = min_cost
                            best_goal_vid = new_vid
            # --- RRT* LOGIC END ---

            # --- REPORTING LOGIC ---
            if iii % report_interval == 0:
                # Record status at this snapshot
                if best_goal_vid is not None:
                    history_success.append(1)
                    history_cost.append(float(best_goal_cost))
                else:
                    history_success.append(0)
                    history_cost.append(None)  # Or np.inf if you prefer

        # Return the collected data
        return history_success, history_cost


class RRTStarExperiment_oldV2(RRTStarPlanner):
    """
    A subclass of RRTStarPlanner designed for data collection.
    It runs for exactly max_itr and records stats at specific intervals.
    """

    def plan_with_stats(self, report_interval=400):
        # Setup defaults
        max_itr = 2000 if self.max_itr is None else int(self.max_itr)

        # Init tree
        if len(self.tree.vertices) == 0:
            self.tree.add_vertex(self.start)

        # Helper: Optimized Cost Propagation
        def propagate_costs_from(parent_vid):
            if parent_vid not in self.children: return
            parent_cost = self.tree.vertices[parent_vid].cost
            for child_vid in self.children[parent_vid]:
                p_conf = self.tree.vertices[parent_vid].config
                c_conf = self.tree.vertices[child_vid].config
                new_cost = parent_cost + self.bb.compute_distance(p_conf, c_conf)
                self.tree.vertices[child_vid].set_cost(new_cost)
                propagate_costs_from(child_vid)

        # --- FIX: Track ALL goal nodes, not just the single best snapshot ---
        goal_vids = []

        # Data containers
        history_success = []
        history_cost = []

        for iii in range(1, max_itr + 1):

            # --- RRT* LOGIC START ---
            x_rand = self.bb.sample_random_config(self.goal_prob, self.goal)
            x_near_id, x_near = self.tree.get_nearest_config(x_rand)
            x_new = self.extend(x_near, x_rand)

            if x_new is not None and \
                    self.bb.config_validity_checker(x_new) and \
                    self.tree.get_idx_for_config(x_new) is None:

                # Dynamic K
                n = len(self.tree.vertices)
                dd = 6
                const = np.exp(1 + (1 / dd))
                k = int(np.ceil(const * np.log(n + 1)))
                k_eff = min(k, n - 1)

                nn_ids, _ = self.tree.get_k_nearest_neighbors(x_new, k_eff)
                if x_near_id not in nn_ids: nn_ids.append(x_near_id)

                # Choose Parent
                best_parent = None
                min_cost = np.inf
                valid_neighbors = []

                for vid in nn_ids:
                    vconf = self.tree.vertices[vid].config
                    if self.bb.edge_validity_checker(vconf, x_new):
                        valid_neighbors.append(vid)
                        cost = self.tree.vertices[vid].cost + self.bb.compute_distance(vconf, x_new)
                        if cost < min_cost:
                            min_cost = cost
                            best_parent = vid

                if best_parent is not None:
                    new_vid = self.tree.add_vertex(x_new)
                    self.tree.add_edge(best_parent, new_vid, min_cost - self.tree.vertices[best_parent].cost)
                    self.tree.vertices[new_vid].set_cost(min_cost)
                    self.children[best_parent].append(new_vid)

                    # Rewire
                    for vid in valid_neighbors:
                        if vid == best_parent: continue
                        dist = self.bb.compute_distance(x_new, self.tree.vertices[vid].config)
                        new_cost = min_cost + dist
                        if new_cost + 1e-6 < self.tree.vertices[vid].cost:
                            old_parent = self.tree.edges[vid]
                            if old_parent in self.children and vid in self.children[old_parent]:
                                self.children[old_parent].remove(vid)
                            self.tree.edges[vid] = new_vid
                            self.children[new_vid].append(vid)
                            self.tree.vertices[vid].set_cost(new_cost)
                            propagate_costs_from(vid)

                    # --- FIX: Just add to list, don't calculate min yet ---
                    dist_to_goal = self.bb.compute_distance(x_new, self.goal)
                    if dist_to_goal < 0.25:
                        goal_vids.append(new_vid)

            # --- REPORTING LOGIC ---
            if iii % report_interval == 0:

                # Scan the goal_vids list to find the CURRENT real minimum cost
                # This ensures we capture improvements from rewiring
                current_min_cost = np.inf
                found_any = False

                for vid in goal_vids:
                    # Check the LIVE cost from the tree
                    c = self.tree.vertices[vid].cost
                    if c < current_min_cost:
                        current_min_cost = c
                        found_any = True

                if found_any:
                    history_success.append(1)
                    history_cost.append(float(current_min_cost))
                else:
                    history_success.append(0)
                    history_cost.append(None)

        return history_success, history_cost


class RRTStarExperiment(RRTStarPlanner):
    """
    A subclass of RRTStarPlanner designed for data collection.
    It runs for exactly max_itr and records stats at specific intervals.
    """

    def plan_with_stats(self, report_interval=400):
        # Setup defaults
        max_itr = 2000 if self.max_itr is None else int(self.max_itr)

        # Init tree
        if len(self.tree.vertices) == 0:
            self.tree.add_vertex(self.start)

        # Helper: Optimized Cost Propagation
        def propagate_costs_from(parent_vid):
            if parent_vid not in self.children: return
            parent_cost = self.tree.vertices[parent_vid].cost
            for child_vid in self.children[parent_vid]:
                p_conf = self.tree.vertices[parent_vid].config
                c_conf = self.tree.vertices[child_vid].config
                new_cost = parent_cost + self.bb.compute_distance(p_conf, c_conf)
                self.tree.vertices[child_vid].set_cost(new_cost)
                propagate_costs_from(child_vid)

        # List to track all nodes that are EXACTLY at the goal
        goal_vids = []

        # Data containers
        history_success = []
        history_cost = []

        for iii in range(1, max_itr + 1):

            # --- RRT* LOGIC START ---
            x_rand = self.bb.sample_random_config(self.goal_prob, self.goal)
            x_near_id, x_near = self.tree.get_nearest_config(x_rand)
            x_new = self.extend(x_near, x_rand)

            if x_new is not None and \
                    self.bb.config_validity_checker(x_new) and \
                    self.tree.get_idx_for_config(x_new) is None:

                # Dynamic K
                n = len(self.tree.vertices)
                dd = 6
                const = np.exp(1 + (1 / dd))
                k = int(np.ceil(const * np.log(n + 1)))
                k_eff = min(k, n - 1)

                nn_ids, _ = self.tree.get_k_nearest_neighbors(x_new, k_eff)
                if x_near_id not in nn_ids: nn_ids.append(x_near_id)

                # Choose Parent
                best_parent = None
                min_cost = np.inf
                valid_neighbors = []

                for vid in nn_ids:
                    vconf = self.tree.vertices[vid].config
                    if self.bb.edge_validity_checker(vconf, x_new):
                        valid_neighbors.append(vid)
                        cost = self.tree.vertices[vid].cost + self.bb.compute_distance(vconf, x_new)
                        if cost < min_cost:
                            min_cost = cost
                            best_parent = vid

                if best_parent is not None:
                    # Add x_new to tree
                    new_vid = self.tree.add_vertex(x_new)
                    self.tree.add_edge(best_parent, new_vid, min_cost - self.tree.vertices[best_parent].cost)
                    self.tree.vertices[new_vid].set_cost(min_cost)
                    self.children[best_parent].append(new_vid)

                    # Rewire
                    for vid in valid_neighbors:
                        if vid == best_parent: continue
                        dist = self.bb.compute_distance(x_new, self.tree.vertices[vid].config)
                        new_cost = min_cost + dist
                        if new_cost + 1e-6 < self.tree.vertices[vid].cost:
                            old_parent = self.tree.edges[vid]
                            if old_parent in self.children and vid in self.children[old_parent]:
                                self.children[old_parent].remove(vid)
                            self.tree.edges[vid] = new_vid
                            self.children[new_vid].append(vid)
                            self.tree.vertices[vid].set_cost(new_cost)
                            propagate_costs_from(vid)

                    # --- FIX: Connect to Exact Goal ---
                    dist_to_goal = self.bb.compute_distance(x_new, self.goal)
                    if dist_to_goal < 0.25:  # If close enough...

                        # 1. Add the EXACT goal configuration as a new vertex
                        exact_goal_vid = self.tree.add_vertex(self.goal)

                        # 2. Calculate cost to reach it
                        final_step_cost = dist_to_goal
                        total_goal_cost = min_cost + final_step_cost

                        # 3. Connect x_new -> exact_goal
                        self.tree.add_edge(new_vid, exact_goal_vid, final_step_cost)
                        self.tree.vertices[exact_goal_vid].set_cost(total_goal_cost)

                        # 4. Update Topology (So updates propagate to the goal node too!)
                        self.children[new_vid].append(exact_goal_vid)

                        # 5. Track this specific goal node for stats
                        goal_vids.append(exact_goal_vid)

            # --- REPORTING LOGIC ---
            if iii % report_interval == 0:
                # Scan all known paths to goal for the current best cost
                current_min_cost = np.inf
                found_any = False

                for vid in goal_vids:
                    c = self.tree.vertices[vid].cost
                    if c < current_min_cost:
                        current_min_cost = c
                        found_any = True

                if found_any:
                    history_success.append(1)
                    history_cost.append(float(current_min_cost))
                else:
                    history_success.append(0)
                    history_cost.append(None)

        return history_success, history_cost

class RRTStarExperiment_path_old(RRTStarPlanner):
    """
    A subclass of RRTStarPlanner designed for data collection.
    Runs for exactly max_itr and returns a SINGLE best plan found (not history lists).
    """

    def plan_with_stats(self, report_interval=400):
        # Setup defaults
        max_itr = 2000 if self.max_itr is None else int(self.max_itr)

        # Init tree
        if len(self.tree.vertices) == 0:
            self.tree.add_vertex(self.start)

        # Make sure children structure exists (needed for cost propagation)
        if not hasattr(self, "children") or self.children is None:
            self.children = {vid: [] for vid in self.tree.vertices.keys()}
        else:
            # ensure all existing vids have an entry
            for vid in self.tree.vertices.keys():
                self.children.setdefault(vid, [])

        # Helper: Optimized Cost Propagation
        def propagate_costs_from(parent_vid):
            if parent_vid not in self.children:
                return
            parent_cost = self.tree.vertices[parent_vid].cost
            for child_vid in self.children[parent_vid]:
                p_conf = self.tree.vertices[parent_vid].config
                c_conf = self.tree.vertices[child_vid].config
                new_cost = parent_cost + self.bb.compute_distance(p_conf, c_conf)
                self.tree.vertices[child_vid].set_cost(new_cost)
                propagate_costs_from(child_vid)

        # Track best goal (single best)
        best_goal_vid = None
        best_goal_cost = np.inf

        # Helper: backtrack a plan from a vertex id to the root
        def extract_plan(goal_vid):
            if goal_vid is None:
                return None
            path_configs = []
            cur = goal_vid
            while True:
                path_configs.append(self.tree.vertices[cur].config)
                if cur == self.tree.get_root_id():
                    break
                if cur not in self.tree.edges:
                    # disconnected (shouldn't happen if tree is consistent)
                    return None
                cur = self.tree.edges[cur]
            path_configs.reverse()
            return np.array(path_configs)

        for iii in range(1, max_itr + 1):

            # --- RRT* LOGIC START ---
            x_rand = self.bb.sample_random_config(self.goal_prob, self.goal)
            x_near_id, x_near = self.tree.get_nearest_config(x_rand)
            x_new = self.extend(x_near, x_rand)

            if (
                x_new is not None
                and self.bb.config_validity_checker(x_new)
                and self.tree.get_idx_for_config(x_new) is None
            ):

                # Dynamic K
                n = len(self.tree.vertices)
                dd = 6
                const = np.exp(1 + (1 / dd))
                k = int(np.ceil(const * np.log(n + 1)))
                k = 5
                k_eff = min(k, n - 1)

                nn_ids, _ = self.tree.get_k_nearest_neighbors(x_new, k_eff)
                if x_near_id not in nn_ids:
                    nn_ids.append(x_near_id)

                # Choose Parent
                best_parent = None
                min_cost = np.inf
                valid_neighbors = []

                for vid in nn_ids:
                    vconf = self.tree.vertices[vid].config
                    if self.bb.edge_validity_checker(vconf, x_new):
                        valid_neighbors.append(vid)
                        cost = self.tree.vertices[vid].cost + self.bb.compute_distance(vconf, x_new)
                        if cost < min_cost:
                            min_cost = cost
                            best_parent = vid

                if best_parent is not None:
                    # Add x_new to tree
                    new_vid = self.tree.add_vertex(x_new)
                    self.children.setdefault(new_vid, [])  # ensure children list
                    self.tree.add_edge(best_parent, new_vid, min_cost - self.tree.vertices[best_parent].cost)
                    self.tree.vertices[new_vid].set_cost(min_cost)
                    self.children.setdefault(best_parent, []).append(new_vid)

                    # Rewire
                    for vid in valid_neighbors:
                        if vid == best_parent:
                            continue
                        dist = self.bb.compute_distance(x_new, self.tree.vertices[vid].config)
                        new_cost = min_cost + dist
                        if new_cost + 1e-6 < self.tree.vertices[vid].cost:
                            old_parent = self.tree.edges.get(vid, None)
                            if old_parent is not None and old_parent in self.children and vid in self.children[old_parent]:
                                self.children[old_parent].remove(vid)
                            self.tree.edges[vid] = new_vid
                            self.children[new_vid].append(vid)
                            self.tree.vertices[vid].set_cost(new_cost)
                            propagate_costs_from(vid)

                    # --- Connect to Exact Goal (and keep ONLY best) ---
                    dist_to_goal = self.bb.compute_distance(x_new, self.goal)
                    if dist_to_goal < 0.1:  # close enough

                        # Add exact goal node
                        exact_goal_vid = self.tree.add_vertex(self.goal)
                        self.children.setdefault(exact_goal_vid, [])

                        # Connect new_vid -> exact_goal_vid
                        total_goal_cost = min_cost + dist_to_goal
                        self.tree.add_edge(new_vid, exact_goal_vid, dist_to_goal)
                        self.tree.vertices[exact_goal_vid].set_cost(total_goal_cost)
                        self.children[new_vid].append(exact_goal_vid)

                        # Update best goal if improved
                        if total_goal_cost < best_goal_cost:
                            best_goal_cost = float(total_goal_cost)
                            best_goal_vid = exact_goal_vid

            # (optional) you can still print/report every interval without returning lists
            # if iii % report_interval == 0 and best_goal_vid is not None:
            #     print(f"[{iii}] best_cost={best_goal_cost:.4f}")

        # Return a SINGLE best plan (or None if no solution)
        best_plan = extract_plan(best_goal_vid)
        return best_plan


class RRTStarExperiment_path(RRTStarExperiment):
    """
    Subclass that overrides .plan() to ensure the final path
    connects EXACTLY to the goal configuration.
    """

    # --- RENAME THIS TO plan() TO OVERRIDE THE PARENT CLASS ---
    def plan(self):
        # Setup defaults
        max_itr = 2000 if self.max_itr is None else int(self.max_itr)

        # Init tree
        if len(self.tree.vertices) == 0:
            self.tree.add_vertex(self.start)

        # Ensure children structure exists
        if not hasattr(self, "children") or self.children is None:
            self.children = {vid: [] for vid in self.tree.vertices.keys()}

        # Helper: Optimized Cost Propagation
        def propagate_costs_from(parent_vid):
            if parent_vid not in self.children: return
            parent_cost = self.tree.vertices[parent_vid].cost
            for child_vid in self.children[parent_vid]:
                p_conf = self.tree.vertices[parent_vid].config
                c_conf = self.tree.vertices[child_vid].config
                new_cost = parent_cost + self.bb.compute_distance(p_conf, c_conf)
                self.tree.vertices[child_vid].set_cost(new_cost)
                propagate_costs_from(child_vid)

        # Helper: Extract path from a specific node back to root
        def extract_plan(goal_vid):
            if goal_vid is None: return None
            path_configs = []
            cur = goal_vid

            # 1. Start at the Goal Node
            while True:
                path_configs.append(self.tree.vertices[cur].config)

                # Stop if root
                if cur == self.tree.get_root_id(): break

                # Move to parent
                cur = self.tree.edges.get(cur)
                if cur is None: break  # Safety break

            # 2. Reverse to get Start -> Goal
            return np.array(path_configs[::-1])

        # Track ALL nodes that are connected to the exact goal
        exact_goal_vids = []

        for iii in range(1, max_itr + 1):

            # --- Standard RRT* Expansion ---
            x_rand = self.bb.sample_random_config(self.goal_prob, self.goal)
            x_near_id, x_near = self.tree.get_nearest_config(x_rand)
            x_new = self.extend(x_near, x_rand)

            if (x_new is not None and
                    self.bb.config_validity_checker(x_new) and
                    self.tree.get_idx_for_config(x_new) is None):

                # 1. Find Neighbors
                n = len(self.tree.vertices)
                k = 10  # Increase k slightly to ensure good connections
                k_eff = min(k, n - 1)
                nn_ids, _ = self.tree.get_k_nearest_neighbors(x_new, k_eff)
                if x_near_id not in nn_ids: nn_ids.append(x_near_id)

                # 2. Choose Best Parent
                best_parent = None
                min_cost = np.inf
                valid_neighbors = []

                for vid in nn_ids:
                    vconf = self.tree.vertices[vid].config
                    if self.bb.edge_validity_checker(vconf, x_new):
                        valid_neighbors.append(vid)
                        cost = self.tree.vertices[vid].cost + self.bb.compute_distance(vconf, x_new)
                        if cost < min_cost:
                            min_cost = cost
                            best_parent = vid

                # 3. Add Node and Rewire
                if best_parent is not None:
                    new_vid = self.tree.add_vertex(x_new)
                    self.children.setdefault(new_vid, [])

                    self.tree.add_edge(best_parent, new_vid, min_cost - self.tree.vertices[best_parent].cost)
                    self.tree.vertices[new_vid].set_cost(min_cost)
                    self.children.setdefault(best_parent, []).append(new_vid)

                    # Rewire Neighbors
                    for vid in valid_neighbors:
                        if vid == best_parent: continue
                        dist = self.bb.compute_distance(x_new, self.tree.vertices[vid].config)
                        new_cost = min_cost + dist
                        if new_cost + 1e-6 < self.tree.vertices[vid].cost:
                            old_parent = self.tree.edges.get(vid)
                            if old_parent in self.children and vid in self.children[old_parent]:
                                self.children[old_parent].remove(vid)
                            self.tree.edges[vid] = new_vid
                            self.children[new_vid].append(vid)
                            self.tree.vertices[vid].set_cost(new_cost)
                            propagate_costs_from(vid)

                    # --- CRITICAL: Connect to Exact Goal ---
                    dist_to_goal = self.bb.compute_distance(x_new, self.goal)

                    # If close enough AND valid path to goal
                    if dist_to_goal < 0.1 and self.bb.edge_validity_checker(x_new, self.goal):
                        # A. Add the Goal Node (we might add multiple goal nodes, that's okay)
                        goal_vid = self.tree.add_vertex(self.goal)
                        self.children.setdefault(goal_vid, [])

                        # B. Connect x_new -> Goal
                        total_goal_cost = min_cost + dist_to_goal
                        self.tree.add_edge(new_vid, goal_vid, dist_to_goal)
                        self.tree.vertices[goal_vid].set_cost(total_goal_cost)
                        self.children[new_vid].append(goal_vid)

                        # C. Save this ID to check later
                        exact_goal_vids.append(goal_vid)

        # --- AFTER LOOP: Find Best Path ---
        if not exact_goal_vids:
            return None

        # Find the goal node with the lowest cost
        best_vid = None
        lowest_cost = np.inf

        for vid in exact_goal_vids:
            c = self.tree.vertices[vid].cost
            if c < lowest_cost:
                lowest_cost = c
                best_vid = vid

        return extract_plan(best_vid)








'''

@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

                NEW

@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@



'''
import numpy as np
from RRTTree import RRTTree
import time


class JRRTStarPlanner_old1(object):

    def __init__(
            self,
            bb,
            ext_mode,
            max_step_size,
            start,
            goal,
            max_itr=None,
            stop_on_goal=None,
            k=None,
            goal_prob=0.01,
    ):
        # set environment and search tree
        self.bb = bb
        self.tree = RRTTree(self.bb)
        self.start = start
        self.goal = goal

        # default max iterations if not provided
        self.max_itr = max_itr if max_itr is not None else 2000
        self.stop_on_goal = stop_on_goal if stop_on_goal is not None else False

        # set search params
        self.ext_mode = ext_mode
        self.goal_prob = goal_prob
        self.k = k
        self.max_step_size = max_step_size

        # Logging / Plotting helpers
        self.log_interval = 400
        self.intervals = []  # Will store [400, 800, ...]
        self.success_list = []
        self.cost_list = []

        # Helper structure to track children for cost propagation
        # Key: parent_vid, Value: list of children_vids
        self.children = {}

    def plan(self):
        """
        Compute and return the plan. The function returns a numpy array containing the states (positions) of the robot.
        """
        # 1. Initialize Tree
        self.tree.add_vertex(self.start)
        self.children[self.tree.get_root_id()] = []
        self.tree.vertices[self.tree.get_root_id()].cost = 0.0  # Cost to start is 0

        # Track the best goal index found so far (if any)
        # We might find the goal multiple times, we want the one with min cost.
        goal_vids = []

        for i in range(1, self.max_itr + 1):

            # 2. Sample
            x_rand = self.bb.sample_random_config(self.goal_prob, self.goal)

            # 3. Nearest Neighbor
            x_nearest_id, x_nearest = self.tree.get_nearest_config(x_rand)

            # 4. Steer / Extend
            x_new = self.extend(x_nearest, x_rand)

            # 5. Check Obstacle Free (Vertex Validity & Edge Validity from Nearest)
            if x_new is not None and \
                    self.bb.config_validity_checker(x_new) and \
                    self.bb.edge_validity_checker(x_nearest, x_new):

                # --- RRT* Logic Starts Here ---

                # 6. Near Neighbors (k-Nearest)
                # Calculate dynamic k if self.k is not set (optional), otherwise use fixed k
                current_n = len(self.tree.vertices)
                if self.k is None:
                    # RRT* optimal k factor (approximate)
                    k_rrt = int(np.ceil(np.e * np.log(current_n)))
                    k_eff = min(k_rrt, current_n - 1)
                else:
                    k_eff = min(self.k, current_n - 1)

                # Get k nearest neighbors
                X_near_ids, _ = self.tree.get_k_nearest_neighbors(x_new, k_eff)

                # Ensure x_nearest is in X_near for consideration
                if x_nearest_id not in X_near_ids:
                    X_near_ids.append(x_nearest_id)

                # 7. Choose Best Parent (Connect along minimum-cost path)
                x_min_id = x_nearest_id
                c_min = self.tree.vertices[x_nearest_id].cost + self.bb.compute_distance(x_nearest, x_new)

                # Iterate over neighbors to find the best parent
                valid_X_near_ids = []  # Cache validity to avoid checking again in rewire

                for x_near_id in X_near_ids:
                    x_near = self.tree.vertices[x_near_id].config

                    # Calculate potential cost through this neighbor
                    dist = self.bb.compute_distance(x_near, x_new)
                    c_new = self.tree.vertices[x_near_id].cost + dist

                    # Optimization: Check cost BEFORE collision
                    if c_new < c_min:
                        if self.bb.edge_validity_checker(x_near, x_new):
                            x_min_id = x_near_id
                            c_min = c_new

                    # We only need to know if edge is valid for the *Rewire* step next
                    # But we can defer that check to the rewire loop to strictly follow "check cost first"

                # 8. Add x_new to Tree
                x_new_id = self.tree.add_vertex(x_new)
                self.tree.vertices[x_new_id].set_cost(c_min)
                self.tree.add_edge(x_min_id, x_new_id, c_min - self.tree.vertices[x_min_id].cost)

                # Update children structure
                self.children[x_new_id] = []
                if x_min_id in self.children:
                    self.children[x_min_id].append(x_new_id)
                else:
                    self.children[x_min_id] = [x_new_id]

                # 9. Rewire the Tree
                # Try to route X_near nodes *through* x_new if it's cheaper
                self.rewire(x_new_id, x_new, c_min, X_near_ids)

                # --- Check if we reached the goal ---
                # Exact connection check
                dist_to_goal = self.bb.compute_distance(x_new, self.goal)
                if dist_to_goal < 1e-4:  # Tolerance for "being at goal"
                    goal_vids.append(x_new_id)
                    if self.stop_on_goal:
                        # Log stats one last time before breaking
                        self.log_stats(i, goal_vids)
                        print(f"Goal reached at iteration {i}, stopping.")
                        break

            # --- Logging Interval ---
            if i % self.log_interval == 0:
                self.intervals.append(i)
                self.log_stats(i, goal_vids)

        # Reconstruct Path
        if not goal_vids:
            return None  # Failed to find path

        # Find best goal node
        min_cost = np.inf
        best_goal_id = None
        for vid in goal_vids:
            if self.tree.vertices[vid].cost < min_cost:
                min_cost = self.tree.vertices[vid].cost
                best_goal_id = vid

        # Backtrack
        path = []
        curr_id = best_goal_id
        while curr_id is not None:
            path.append(self.tree.vertices[curr_id].config)
            # Stop if we hit root (root has no parent usually, or parent is None)
            if curr_id == self.tree.get_root_id():
                break
            # Get parent
            curr_id = self.tree.edges[curr_id]

        print(f"self.success_list ={self.success_list }")
        print(f"self.cost_list={self.cost_list}")
        return np.array(path[::-1])  # Return reversed path

    def extend(self, x_near, x_rand):
        """
        Steer from x_near towards x_rand.
        """
        # TODO: HW3 3
        # Use existing logic from RRT or standard steer
        # If ext_mode is specific, handle it. Assuming standard interpolation here.
        dist = self.bb.compute_distance(x_near, x_rand)
        if dist <= self.max_step_size:
            return x_rand
        else:
            # Interpolate
            # Assuming configs are numpy arrays.
            # (x_rand - x_near) / dist * step + x_near
            direction = (x_rand - x_near) / dist
            x_new = x_near + direction * self.max_step_size
            return x_new

    def rewire(self, x_new_id, x_new, c_new, X_near_ids):
        """
        Check if x_new can be a better parent for nodes in X_near_ids.
        """
        for x_near_id in X_near_ids:
            if x_near_id == x_new_id:
                continue  # Skip self

            # 1. Check Cost First (Optimization)
            curr_near_cost = self.tree.vertices[x_near_id].cost
            dist = self.bb.compute_distance(x_new, self.tree.vertices[x_near_id].config)
            potential_new_cost = c_new + dist

            if potential_new_cost < curr_near_cost - 1e-6:  # Epsilon for float stability

                # 2. Check Collision Only if Cost is Better
                if self.bb.edge_validity_checker(x_new, self.tree.vertices[x_near_id].config):

                    # Get old parent to update children list
                    old_parent_id = self.tree.edges[x_near_id]

                    # Remove x_near from old parent's children list
                    if old_parent_id in self.children and x_near_id in self.children[old_parent_id]:
                        self.children[old_parent_id].remove(x_near_id)

                    # Update Tree Structure: Parent of x_near becomes x_new
                    self.tree.edges[x_near_id] = x_new_id

                    # Add x_near to x_new's children list
                    if x_new_id not in self.children:
                        self.children[x_new_id] = []
                    self.children[x_new_id].append(x_near_id)

                    # Update Cost of x_near
                    self.tree.vertices[x_near_id].set_cost(potential_new_cost)

                    # 3. Propagate Cost to all descendants
                    self.propagate_cost(x_near_id)

    def propagate_cost(self, parent_id):
        """
        Recursively update cost for all children of parent_id.
        """
        if parent_id not in self.children:
            return

        parent_cost = self.tree.vertices[parent_id].cost

        for child_id in self.children[parent_id]:
            # Calculate distance between parent and child
            dist = self.bb.compute_distance(
                self.tree.vertices[parent_id].config,
                self.tree.vertices[child_id].config
            )

            new_child_cost = parent_cost + dist
            self.tree.vertices[child_id].set_cost(new_child_cost)

            # Recurse
            self.propagate_cost(child_id)

    def log_stats(self, iteration, goal_vids):
        """
        Helper to log success and cost.
        """
        # Check if we have any solution
        if len(goal_vids) > 0:
            self.success_list.append(1)

            # Find min cost among goal nodes
            min_c = np.inf
            for vid in goal_vids:
                c = self.tree.vertices[vid].cost
                if c < min_c:
                    min_c = c
            self.cost_list.append(min_c)
        else:
            self.success_list.append(0)
            self.cost_list.append(None)  # Or np.inf depending on plotting pref

    def compute_cost(self, plan):
        """
        Calculate total cost of a plan (sum of distances).
        """
        # TODO: HW3 3
        if plan is None or len(plan) < 2:
            return 0.0

        cost = 0.0
        for i in range(len(plan) - 1):
            cost += self.bb.compute_distance(plan[i], plan[i + 1])
        return cost


'''
@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@

SECOND TRY

@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@

'''
import numpy as np
from RRTTree import RRTTree
import time


class JRRTStarPlanner(object):

    def __init__(
            self,
            bb,
            ext_mode,
            max_step_size,
            start,
            goal,
            max_itr=None,
            stop_on_goal=None,
            k=None,
            goal_prob=0.01,
    ):
        self.bb = bb
        self.tree = RRTTree(self.bb)
        self.start = start
        self.goal = goal
        self.max_itr = max_itr if max_itr is not None else 2000
        self.stop_on_goal = stop_on_goal if stop_on_goal is not None else False
        self.ext_mode = ext_mode
        self.goal_prob = goal_prob
        self.k = k
        self.max_step_size = max_step_size

        # Logging
        self.log_interval = 400
        self.intervals = []
        self.success_list = []
        self.cost_list = []

        # Optimization: Map parent_id -> list of children_ids
        self.children = {}

        # TRACKING THE SINGLE GOAL NODE
        self.goal_idx = None  # Will hold the index of the goal vertex once found

    def plan(self):
        # 1. Init Tree
        self.tree.add_vertex(self.start)
        self.children[self.tree.get_root_id()] = []
        self.tree.vertices[self.tree.get_root_id()].cost = 0.0

        for i in range(1, self.max_itr + 1):
            if i % 200 == 0:
                print(f"iter {i}")
            # 2. Sample
            x_rand = self.bb.sample_random_config(self.goal_prob, self.goal)

            # 3. Nearest & Extend
            x_nearest_id, x_nearest = self.tree.get_nearest_config(x_rand)
            x_new = self.extend(x_nearest, x_rand)

            # 4. Validity Check (Obstacle Free)
            if x_new is not None and \
                    self.bb.config_validity_checker(x_new) and \
                    self.bb.edge_validity_checker(x_nearest, x_new):

                # --- RRT* Standard Logic ---

                # Dynamic K
                current_n = len(self.tree.vertices)
                if self.k is None:

                    k_rrt = int(np.ceil(1 * np.log(current_n))) # 4 is calculated as ceil(exp(1+1/d) and d=6)
                    k_rrt = 1
                    k_eff = min(k_rrt, current_n - 1)
                    # k_eff = 1
                else:
                    k_eff = min(self.k, current_n - 1)

                # Get Neighbors
                X_near_ids, _ = self.tree.get_k_nearest_neighbors(x_new, k_eff)
                if x_nearest_id not in X_near_ids:
                    X_near_ids.append(x_nearest_id)

                # Choose Best Parent
                x_min_id = x_nearest_id
                c_min = self.tree.vertices[x_nearest_id].cost + self.bb.compute_distance(x_nearest, x_new)

                for x_near_id in X_near_ids:
                    x_near = self.tree.vertices[x_near_id].config
                    dist = self.bb.compute_distance(x_near, x_new)
                    c_potential = self.tree.vertices[x_near_id].cost + dist

                    if c_potential < c_min:
                        if self.bb.edge_validity_checker(x_near, x_new):
                            x_min_id = x_near_id
                            c_min = c_potential

                # Add x_new
                x_new_id = self.tree.add_vertex(x_new)
                self.tree.vertices[x_new_id].set_cost(c_min)
                self.tree.add_edge(x_min_id, x_new_id, c_min - self.tree.vertices[x_min_id].cost)

                # Update children map
                self.children[x_new_id] = []
                if x_min_id not in self.children: self.children[x_min_id] = []
                self.children[x_min_id].append(x_new_id)

                # Rewire Neighbors
                self.rewire(x_new_id, x_new, c_min, X_near_ids)

                # --- GOAL CONNECTION LOGIC (SINGLE SOLUTION) ---

                # Check distance to exact goal
                dist_to_goal = self.bb.compute_distance(x_new, self.goal)

                # Threshold to attempt connection (e.g., max_step_size or smaller)
                if dist_to_goal <= self.max_step_size:

                    # Calculate cost if we connect via x_new
                    potential_goal_cost = c_min + dist_to_goal

                    # CASE A: Goal is NOT in the tree yet
                    if self.goal_idx is None:
                        # Must check collision for the final step
                        if self.bb.edge_validity_checker(x_new, self.goal):
                            # Add Goal Vertex
                            self.goal_idx = self.tree.add_vertex(self.goal)
                            self.tree.vertices[self.goal_idx].set_cost(potential_goal_cost)
                            self.tree.add_edge(x_new_id, self.goal_idx, dist_to_goal)

                            self.children[self.goal_idx] = []
                            self.children[x_new_id].append(self.goal_idx)

                            if self.stop_on_goal:
                                self.log_stats(i)
                                print(f"Goal reached at {i} cost={potential_goal_cost:.2f}")
                                break

                    # CASE B: Goal IS already in the tree -> Check if we can REWIRE it
                    else:
                        current_goal_cost = self.tree.vertices[self.goal_idx].cost

                        # Only try to rewire if it improves cost
                        if potential_goal_cost < current_goal_cost - 1e-6:
                            if self.bb.edge_validity_checker(x_new, self.goal):
                                # Rewire the Goal Node!
                                self.rewire_single_node(self.goal_idx, x_new_id, potential_goal_cost)
                                # (Cost propagation happens inside rewire_single_node)

            # Logging
            if i % self.log_interval == 0:
                self.intervals.append(i)
                self.log_stats(i)

        # End of Loop - Return best path
        if self.goal_idx is None:
            return None

        print(f"self.success_list ={self.success_list }")
        print(f"self.cost_list={self.cost_list}")
        return self.reconstruct_path(self.goal_idx)

    def rewire(self, x_new_id, x_new, c_new, X_near_ids):
        """Standard RRT* Rewire for neighbors."""
        for x_near_id in X_near_ids:
            if x_near_id == x_new_id: continue

            curr_cost = self.tree.vertices[x_near_id].cost
            dist = self.bb.compute_distance(x_new, self.tree.vertices[x_near_id].config)
            new_cost = c_new + dist

            if new_cost < curr_cost - 1e-6:
                if self.bb.edge_validity_checker(x_new, self.tree.vertices[x_near_id].config):
                    self.rewire_single_node(x_near_id, x_new_id, new_cost)

    def rewire_single_node(self, child_id, new_parent_id, new_child_cost):
        """
        Helper to perform the actual pointer update and cost propagation.
        Used for both standard neighbors and the Goal node.
        """
        # 1. Remove from old parent's children list
        old_parent_id = self.tree.edges[child_id]
        if old_parent_id in self.children and child_id in self.children[old_parent_id]:
            self.children[old_parent_id].remove(child_id)

        # 2. Update Parent Pointer
        self.tree.edges[child_id] = new_parent_id

        # 3. Add to new parent's children list
        if new_parent_id not in self.children: self.children[new_parent_id] = []
        self.children[new_parent_id].append(child_id)

        # 4. Update Cost
        self.tree.vertices[child_id].set_cost(new_child_cost)

        # 5. Propagate
        self.propagate_cost(child_id)

    def propagate_cost(self, parent_id):
        if parent_id not in self.children: return
        parent_cost = self.tree.vertices[parent_id].cost

        for child_id in self.children[parent_id]:
            dist = self.bb.compute_distance(
                self.tree.vertices[parent_id].config,
                self.tree.vertices[child_id].config
            )
            self.tree.vertices[child_id].set_cost(parent_cost + dist)
            self.propagate_cost(child_id)

    def reconstruct_path(self, goal_idx):
        path = []
        curr = goal_idx
        while curr is not None:
            path.append(self.tree.vertices[curr].config)
            if curr == self.tree.get_root_id(): break
            curr = self.tree.edges[curr]
        return np.array(path[::-1])

    def extend(self, x_near, x_rand):
        dist = self.bb.compute_distance(x_near, x_rand)
        if dist <= self.max_step_size:
            return x_rand
        direction = (x_rand - x_near) / dist
        return x_near + direction * self.max_step_size

    def log_stats(self, iteration):
        if self.goal_idx is not None:
            self.success_list.append(1)
            self.cost_list.append(self.tree.vertices[self.goal_idx].cost)
        else:
            self.success_list.append(0)
            self.cost_list.append(None)

    def compute_cost(self, plan):
        if plan is None or len(plan) < 2: return 0.0
        cost = 0.0
        for i in range(len(plan) - 1):
            cost += self.bb.compute_distance(plan[i], plan[i + 1])
        return cost