import itertools
import numpy as np
from shapely.geometry import Point, LineString

import itertools
import numpy as np
from shapely.geometry import Point, LineString  # kept for compatibility (not used in fast path)


class BuildingBlocks2D(object):
    """
    Faster implementation of the same API.
    Key speedups:
      - Vectorized forward kinematics (single + batch).
      - Collision checking without Shapely in inner loops (NumPy segment intersection).
      - Precompute obstacle segments once in __init__.
      - Vectorized inspection point filtering (distance + FOV) + fast occlusion check.
    """

    def __init__(self, env):
        self.env = env
        # define robot properties
        self.links = np.array([80.0, 70.0, 40.0, 40.0], dtype=float)
        self.dim = len(self.links)

        # robot field of fiew (FOV) for inspecting points, from [-np.pi/6, np.pi/6]
        self.ee_fov = np.pi / 3

        # visibility distance for the robot's end-effector. Farther than that, the robot won't see any points.
        self.vis_dist = 60.0

        # cache world bounds
        self._xmin = float(self.env.xlimit[0])
        self._xmax = float(self.env.xlimit[1])
        self._ymin = float(self.env.ylimit[0])
        self._ymax = float(self.env.ylimit[1])

        # ---- Precompute obstacle segments as a flat NumPy array: (M, 2, 2) ----
        # env.obstacles_edges is typically: list[list[LineString]]
        segs = []
        if hasattr(self.env, "obstacles_edges") and self.env.obstacles_edges is not None:
            for obstacle_edges in self.env.obstacles_edges:
                for e in obstacle_edges:
                    # Shapely LineString supports .coords
                    (x1, y1), (x2, y2) = list(e.coords)
                    segs.append([[x1, y1], [x2, y2]])
        self._obs_segs = np.asarray(segs, dtype=float) if len(segs) else np.zeros((0, 2, 2), dtype=float)

        # inspection points as numpy array (ensure contiguous float)
        if hasattr(self.env, "inspection_points") and self.env.inspection_points is not None:
            self._insp_pts = np.asarray(self.env.inspection_points, dtype=float)
        else:
            self._insp_pts = np.zeros((0, 2), dtype=float)

    # -------------------- small utilities --------------------
    @staticmethod
    def _wrap_to_pi(a):
        return (a + np.pi) % (2.0 * np.pi) - np.pi

    @staticmethod
    def _orient(a, b, c):
        # cross((b-a),(c-a)) for arrays (...,2)
        return (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])

    @staticmethod
    def _on_segment(a, b, c, eps=1e-9):
        return (
            (np.minimum(a[..., 0], b[..., 0]) - eps <= c[..., 0]) & (c[..., 0] <= np.maximum(a[..., 0], b[..., 0]) + eps) &
            (np.minimum(a[..., 1], b[..., 1]) - eps <= c[..., 1]) & (c[..., 1] <= np.maximum(a[..., 1], b[..., 1]) + eps)
        )

    def _segments_intersect_any(self, segsA, segsB):
        """
        segsA: (S,2,2)
        segsB: (M,2,2)
        returns True if any segment in A intersects any segment in B (including touching/collinear overlap).
        """
        if segsA.size == 0 or segsB.size == 0:
            return False

        A1 = segsA[:, None, 0, :]   # (S,1,2)
        A2 = segsA[:, None, 1, :]
        B1 = segsB[None, :, 0, :]   # (1,M,2)
        B2 = segsB[None, :, 1, :]

        o1 = self._orient(A1, A2, B1)
        o2 = self._orient(A1, A2, B2)
        o3 = self._orient(B1, B2, A1)
        o4 = self._orient(B1, B2, A2)

        # strict intersection
        inter = (o1 * o2 < 0) & (o3 * o4 < 0)

        # collinear / touching
        eps = 1e-9
        col1 = (np.abs(o1) < eps) & self._on_segment(A1, A2, B1, eps=eps)
        col2 = (np.abs(o2) < eps) & self._on_segment(A1, A2, B2, eps=eps)
        col3 = (np.abs(o3) < eps) & self._on_segment(B1, B2, A1, eps=eps)
        col4 = (np.abs(o4) < eps) & self._on_segment(B1, B2, A2, eps=eps)

        return bool(np.any(inter | col1 | col2 | col3 | col4))

    @staticmethod
    def _robot_segs_from_positions(pos5):
        """
        pos5: (5,2) including base at index 0
        returns (4,2,2) segments between consecutive joints
        """
        return np.stack([pos5[:-1], pos5[1:]], axis=1)

    def _self_collision_4links(self, pos5):
        """
        Fast self-collision for 4-link chain: only non-adjacent pairs can collide.
        pairs among segments [0..3]: (0,2), (0,3), (1,3)
        """
        segs = self._robot_segs_from_positions(pos5)
        # Check only those pairs; each check is tiny.
        pairs = ((0, 2), (0, 3), (1, 3))
        for i, j in pairs:
            if self._segments_intersect_any(segs[i:i + 1], segs[j:j + 1]):
                return True
        return False

    # -------------------- required API methods --------------------
    def compute_distance(self, prev_config, next_config):
        '''
        Compute the euclidean distance betweeen two given configurations.
        @param prev_config Previous configuration.
        @param next_config Next configuration.
        '''
        # Distance in workspace between end-effector positions (fast FK).
        ee1 = self.compute_forward_kinematics(given_config=prev_config)[-1]
        ee2 = self.compute_forward_kinematics(given_config=next_config)[-1]
        return float(np.linalg.norm(ee2 - ee1))

    def sample_random_config(self, goal_prob, goal):
        """
        Sample a random configuration in the configuration space.
        With probability goal_prob, return the goal configuration.
        """

        # goal biasing
        if np.random.rand() < goal_prob:
            return np.array(goal, dtype=float)

        # sample joint angles uniformly in [-pi, pi]
        config = np.random.uniform(low=-np.pi, high=np.pi, size=self.dim)
        return config.astype(float)

    def compute_path_cost(self, path):
        totat_cost = 0.0
        for i in range(len(path) - 1):
            totat_cost += self.compute_distance(path[i], path[i + 1])
        return float(totat_cost)

    def compute_forward_kinematics(self, given_config):
        '''
        Compute the 2D position (x,y) of each one of the links (including end-effector) and return.
        @param given_config Given configuration.
        '''
        q = np.asarray(given_config, dtype=float)
        # cumulative angles + wrap
        angles = self._wrap_to_pi(np.cumsum(q))  # (4,)
        dx = self.links * np.cos(angles)
        dy = self.links * np.sin(angles)
        x = np.cumsum(dx)
        y = np.cumsum(dy)
        return np.column_stack([x, y])  # (4,2)

    def _compute_forward_kinematics_batch(self, Q):
        """
        Q: (N,4)
        returns: (N,4,2)
        """
        Q = np.asarray(Q, dtype=float)
        angles = self._wrap_to_pi(np.cumsum(Q, axis=1))  # (N,4)
        dx = self.links[None, :] * np.cos(angles)
        dy = self.links[None, :] * np.sin(angles)
        x = np.cumsum(dx, axis=1)
        y = np.cumsum(dy, axis=1)
        return np.stack([x, y], axis=2)

    def compute_ee_angle(self, given_config):
        '''
        Compute the 1D orientation of the end-effector w.r.t. world origin (or first joint)
        @param given_config Given configuration.
        '''
        q = np.asarray(given_config, dtype=float)
        return float(self._wrap_to_pi(np.sum(q)))

    def compute_link_angle(self, link_angle, given_angle):
        '''
        Compute the 1D orientation of a link given the previous link and the current joint angle.
        @param link_angle previous link angle.
        @param given_angle Given joint angle.
        '''
        # keep the original behavior but make it branchless & fast
        return float(self._wrap_to_pi(link_angle + given_angle))

    def validate_robot(self, robot_positions):
        '''
        Verify that the given set of links positions does not contain self collisions.
        @param robot_positions Given links positions.
        '''
        pos = np.asarray(robot_positions, dtype=float)

        # Support both inputs:
        # - (4,2): only link endpoints (no base)
        # - (5,2): includes base at row 0
        if pos.shape == (4, 2):
            pos5 = np.vstack([np.zeros((1, 2), dtype=float), pos])
        elif pos.shape == (5, 2):
            pos5 = pos
        else:
            # fallback to original intent: treat as endpoints without base
            pos5 = np.vstack([np.zeros((1, 2), dtype=float), pos.reshape(-1, 2)])

        # self-collision check for 4-link chain
        return (not self._self_collision_4links(pos5))

    def config_validity_checker(self, config):
        '''
        Verify that the config (given or stored) does not contain self collisions or links that are out of the world boundaries.
        Return false if the config is not applicable, and true otherwise.
        @param config The given configuration of the robot.
        '''
        # compute robot links positions
        robot_positions = self.compute_forward_kinematics(given_config=config)  # (4,2)

        # add position of robot placement ([0,0] - position of the first joint)
        pos5 = np.vstack([np.zeros((1, 2), dtype=float), robot_positions])  # (5,2)

        # verify bounds (vectorized)
        if (pos5[:, 0] < self._xmin).any() or (pos5[:, 0] > self._xmax).any() or (pos5[:, 1] < self._ymin).any() or (pos5[:, 1] > self._ymax).any():
            return False

        # self collision (fast)
        if self._self_collision_4links(pos5):
            return False

        # obstacle collision: robot segments vs precomputed obstacle segments
        if self._obs_segs.shape[0] > 0:
            robot_segs = self._robot_segs_from_positions(pos5)  # (4,2,2)
            if self._segments_intersect_any(robot_segs, self._obs_segs):
                return False

        return True

    def edge_validity_checker(self, config1, config2):
        '''
        A function to check if the edge between two configurations is free from collisions. The function will interpolate between the two states to verify
        that the links during motion do not collide with anything.
        @param config1 The source configuration of the robot.
        @param config2 The destination configuration of the robot.
        '''
        required_diff = 0.05
        c1 = np.asarray(config1, dtype=float)
        c2 = np.asarray(config2, dtype=float)

        steps = int(np.linalg.norm(c2 - c1) // required_diff)
        if steps <= 0:
            return True

        # interpolate in joint space
        Q = np.linspace(start=c1, stop=c2, num=steps)  # (N,4)

        # batch FK (fast)
        fkN = self._compute_forward_kinematics_batch(Q)  # (N,4,2)
        posN = np.concatenate([np.zeros((fkN.shape[0], 1, 2), dtype=float), fkN], axis=1)  # (N,5,2)

        # bounds quick reject (vectorized)
        if (posN[:, :, 0] < self._xmin).any() or (posN[:, :, 0] > self._xmax).any() or (posN[:, :, 1] < self._ymin).any() or (posN[:, :, 1] > self._ymax).any():
            return False

        # collision per interpolated config (still very fast: 4 segments vs M)
        if self._obs_segs.shape[0] == 0:
            # only self-collision checks
            for k in range(posN.shape[0]):
                if self._self_collision_4links(posN[k]):
                    return False
            return True

        for k in range(posN.shape[0]):
            pos5 = posN[k]
            if self._self_collision_4links(pos5):
                return False
            robot_segs = self._robot_segs_from_positions(pos5)
            if self._segments_intersect_any(robot_segs, self._obs_segs):
                return False

        return True

    def get_inspected_points(self, config):
        '''
        A function to compute the set of points that are visible to the robot with the given configuration.
        The function will return the set of points that is visible in terms of distance and field of view (FOV) and are not hidden by any obstacle.
        @param config The given configuration of the robot.
        '''
        if self._insp_pts.shape[0] == 0:
            return np.array([])

        # end-effector pose
        ee_pos = self.compute_forward_kinematics(given_config=config)[-1]  # (2,)
        ee_angle = self.compute_ee_angle(given_config=config)

        # vectorized distance + FOV filtering
        rel = self._insp_pts - ee_pos[None, :]              # (M,2)
        d = np.linalg.norm(rel, axis=1)                     # (M,)
        ang = np.arctan2(rel[:, 1], rel[:, 0])              # (M,)
        diff = self._wrap_to_pi(ang - ee_angle)             # (M,)

        mask = (d <= self.vis_dist) & (np.abs(diff) <= (self.ee_fov * 0.5))
        candidates = self._insp_pts[mask]
        if candidates.shape[0] == 0:
            return np.array([])

        # occlusion: segment (ee_pos -> candidate) intersects any obstacle edge
        if self._obs_segs.shape[0] == 0:
            return candidates.copy()

        # chunk to avoid huge (Nc x Mobs) memory
        visible = []
        CHUNK = 256
        for i in range(0, candidates.shape[0], CHUNK):
            chunk = candidates[i:i + CHUNK]  # (K,2)
            rays = np.stack(
                [np.repeat(ee_pos[None, :], chunk.shape[0], axis=0), chunk],
                axis=1
            )  # (K,2,2)

            # If any ray intersects any obstacle segment => hidden
            # We want rays that do NOT intersect obstacles.
            # We'll compute an intersection mask per ray by checking each ray against all obs segments.
            # Do it by batching rays: (K,2,2) vs (M,2,2) => bool any hit, per ray.
            K = rays.shape[0]
            hit_any = np.zeros((K,), dtype=bool)

            # evaluate in one broadcast pass, then reduce over obstacle segments
            A1 = rays[:, None, 0, :]
            A2 = rays[:, None, 1, :]
            B1 = self._obs_segs[None, :, 0, :]
            B2 = self._obs_segs[None, :, 1, :]

            o1 = self._orient(A1, A2, B1)
            o2 = self._orient(A1, A2, B2)
            o3 = self._orient(B1, B2, A1)
            o4 = self._orient(B1, B2, A2)

            inter = (o1 * o2 < 0) & (o3 * o4 < 0)

            eps = 1e-9
            col1 = (np.abs(o1) < eps) & self._on_segment(A1, A2, B1, eps=eps)
            col2 = (np.abs(o2) < eps) & self._on_segment(A1, A2, B2, eps=eps)
            col3 = (np.abs(o3) < eps) & self._on_segment(B1, B2, A1, eps=eps)
            col4 = (np.abs(o4) < eps) & self._on_segment(B1, B2, A2, eps=eps)

            hit = inter | col1 | col2 | col3 | col4
            hit_any = np.any(hit, axis=1)

            visible.append(chunk[~hit_any])

        if len(visible) == 0:
            return np.array([])

        return np.concatenate(visible, axis=0) if len(visible) > 1 else visible[0]

    def compute_angle_of_vector(self, vec):
        '''
        A utility function to compute the angle of the vector from the end-effector to a point.
        @param vec Vector from the end-effector to a point.
        '''
        v = np.asarray(vec, dtype=float)
        return float(np.arctan2(v[1], v[0]))

    def check_if_angle_in_range(self, angle, ee_range):
        '''
        A utility function to check if an inspection point is inside the FOV of the end-effector.
        @param angle The angle beteen the point and the end-effector.
        @param ee_range The FOV of the end-effector.
        '''
        # Keep the signature, but implement a faster/robust check using wrapped difference.
        # ee_range = [center - fov/2, center + fov/2] possibly wrapping aroundz
        ee_center = float(self._wrap_to_pi((ee_range[0] + ee_range[1]) * 0.5))
        diff = float(self._wrap_to_pi(angle - ee_center))
        return (abs(diff) <= (self.ee_fov * 0.5) + 1e-12)

    def compute_union_of_points(self, points1, points2):
        '''
        Compute a union of two sets of inpection points.
        @param points1 list of inspected points.
        @param points2 list of inspected points.
        '''
        # points may be np.array([]) or shape (K,2)
        p1 = np.asarray(points1, dtype=float)
        p2 = np.asarray(points2, dtype=float)

        if p1.size == 0:
            return p2.copy() if p2.size else np.array([])
        if p2.size == 0:
            return p1.copy()

        # Robust union for float coords: round to 1e-6 grid then unique
        P = np.vstack([p1.reshape(-1, 2), p2.reshape(-1, 2)])
        key = np.round(P, 6)  # adjust tolerance if needed
        _, idx = np.unique(key, axis=0, return_index=True)
        return P[np.sort(idx)]

    def compute_coverage(self, inspected_points):
        '''
        Compute the coverage of the map as the portion of points that were already inspected.
        @param inspected_points list of inspected points.
        '''
        if self._insp_pts.shape[0] == 0:
            return 0.0
        return float(len(inspected_points) / len(self._insp_pts))









































'''
OLDER
'''
#
# class BuildingBlocks2D(object):
#
#     def __init__(self, env):
#         self.env = env
#         # define robot properties
#         self.links = np.array([80.0, 70.0, 40.0, 40.0])
#         self.dim = len(self.links)
#
#         # robot field of fiew (FOV) for inspecting points, from [-np.pi/6, np.pi/6]
#         self.ee_fov = np.pi / 3
#
#         # visibility distance for the robot's end-effector. Farther than that, the robot won't see any points.
#         self.vis_dist = 60.0
#
#     def compute_distance(self, prev_config, next_config):
#         '''
#         Compute the euclidean distance betweeen two given configurations.
#         @param prev_config Previous configuration.
#         @param next_config Next configuration.
#         '''
#         x, y = 0, 0
#         prev_angle = 0
#         next_angle = 0
#         for i in range(len(prev_config)):
#             prev_angle = self.compute_link_angle(prev_config[i], prev_angle)
#             next_angle = self.compute_link_angle(next_config[i], next_angle)
#             x += self.links[i] * (np.cos(prev_angle) - np.cos(next_angle))
#             y += self.links[i] * (np.sin(prev_angle) - np.sin(next_angle))
#         return np.sqrt(x ** 2 + y ** 2)
#
#         pass
#
#     def sample_random_config(self, goal_prob, goal):
#         """
#         Sample a random configuration in the configuration space.
#         With probability goal_prob, return the goal configuration.
#         """
#
#         # goal biasing
#         if np.random.rand() < goal_prob:
#             return np.array(goal, dtype=float)
#
#         # sample joint angles uniformly in [-pi, pi]
#         config = np.random.uniform(
#             low=-np.pi,
#             high=np.pi,
#             size=self.dim)
#
#         return config.astype(float)
#
#     def compute_path_cost(self, path):
#         totat_cost = 0
#         for i in range(len(path) - 1):
#             totat_cost += self.compute_distance(path[i], path[i + 1])
#         return totat_cost
#
#     def compute_forward_kinematics(self, given_config):
#         '''
#         Compute the 2D position (x,y) of each one of the links (including end-effector) and return.
#         @param given_config Given configuration.
#         '''
#         x, y = 0, 0
#         angle = 0
#         coords = np.zeros((4,2))
#         for i in range(len(given_config)):
#             angle = self.compute_link_angle(given_config[i], angle)
#             x += self.links[i] * np.cos(angle)
#             y += self.links[i] * np.sin(angle)
#             coords[i][0] = x
#             coords[i][1] = y
#
#         return coords
#     def compute_ee_angle(self, given_config):
#         '''
#         Compute the 1D orientation of the end-effector w.r.t. world origin (or first joint)
#         @param given_config Given configuration.
#         '''
#         ee_angle = given_config[0]
#         for i in range(1, len(given_config)):
#             ee_angle = self.compute_link_angle(ee_angle, given_config[i])
#
#         return ee_angle
#
#     def compute_link_angle(self, link_angle, given_angle):
#         '''
#         Compute the 1D orientation of a link given the previous link and the current joint angle.
#         @param link_angle previous link angle.
#         @param given_angle Given joint angle.
#         '''
#         if link_angle + given_angle > np.pi:
#             return link_angle + given_angle - 2 * np.pi
#         elif link_angle + given_angle < -np.pi:
#             return link_angle + given_angle + 2 * np.pi
#         else:
#             return link_angle + given_angle
#
#     def validate_robot(self, robot_positions):
#         '''
#         Verify that the given set of links positions does not contain self collisions.
#         @param robot_positions Given links positions.
#         '''
#         pts2 = robot_positions.tolist()
#         pts = [[0.0,0.0]]
#         pts.extend(pts2)
#         robot_links = [
#             LineString([pts[i], pts[i + 1]])
#             for i in range(len(pts) - 1)
#         ]
#
#         for i in range(len(robot_links)):
#             for j in range(i + 2, len(robot_links)):  # skip self & adjacent links
#                 if robot_links[i].intersects(robot_links[j]):
#                     return False
#
#         return True
#
#     def config_validity_checker(self, config):
#         '''
#         Verify that the config (given or stored) does not contain self collisions or links that are out of the world boundaries.
#         Return false if the config is not applicable, and true otherwise.
#         @param config The given configuration of the robot.
#         '''
#         # compute robot links positions
#         robot_positions = self.compute_forward_kinematics(given_config=config)
#
#         # add position of robot placement ([0,0] - position of the first joint)
#         robot_positions = np.concatenate([np.zeros((1,2)), robot_positions])
#
#         # verify that the robot do not collide with itself
#         if not self.validate_robot(robot_positions=robot_positions):
#             return False
#
#         # verify that all robot joints (and links) are between world boundaries
#         non_applicable_poses = [(x[0] < self.env.xlimit[0] or x[1] < self.env.ylimit[0] or x[0] > self.env.xlimit[1] or x[1] > self.env.ylimit[1]) for x in robot_positions]
#         if any(non_applicable_poses):
#             return False
#
#         # verify that all robot links do not collide with obstacle edges
#         # for each obstacle, check collision with each of the robot links
#         robot_links = [LineString([Point(x[0],x[1]),Point(y[0],y[1])]) for x,y in zip(robot_positions.tolist()[:-1], robot_positions.tolist()[1:])]
#         for obstacle_edges in self.env.obstacles_edges:
#             for robot_link in robot_links:
#                 obstacle_collisions = [robot_link.crosses(x) for x in obstacle_edges]
#                 if any(obstacle_collisions):
#                     return False
#
#         return True
#
#     def edge_validity_checker(self, config1, config2):
#         '''
#         A function to check if the edge between two configurations is free from collisions. The function will interpolate between the two states to verify
#         that the links during motion do not collide with anything.
#         @param config1 The source configuration of the robot.
#         @param config2 The destination configuration of the robot.
#         '''
#         # interpolate between first config and second config to verify that there is no collision during the motion
#         required_diff = 0.05
#         interpolation_steps = int(np.linalg.norm(config2 - config1) // required_diff)
#         if interpolation_steps > 0:
#             interpolated_configs = np.linspace(start=config1, stop=config2, num=interpolation_steps)
#
#             # compute robot links positions for interpolated configs
#             configs_positions = np.apply_along_axis(self.compute_forward_kinematics, 1, interpolated_configs)
#
#             # compute edges between joints to verify that the motion between two configs does not collide with anything
#             edges_between_positions = []
#             for j in range(self.dim):
#                 for i in range(interpolation_steps - 1):
#                     edges_between_positions.append(LineString(
#                         [Point(configs_positions[i, j, 0], configs_positions[i, j, 1]),
#                          Point(configs_positions[i + 1, j, 0], configs_positions[i + 1, j, 1])]))
#
#             # check collision for each edge between joints and each obstacle
#             for edge_pos in edges_between_positions:
#                 for obstacle_edges in self.env.obstacles_edges:
#                     obstacle_collisions = [edge_pos.crosses(x) for x in obstacle_edges]
#                     if any(obstacle_collisions):
#                         return False
#
#             # add position of robot placement ([0,0] - position of the first joint)
#             configs_positions = np.concatenate([np.zeros((len(configs_positions), 1, 2)), configs_positions], axis=1)
#
#             # verify that the robot do not collide with itself during motion
#             for config_positions in configs_positions:
#                 if not self.validate_robot(config_positions):
#                     return False
#
#             # verify that all robot joints (and links) are between world boundaries
#             if len(np.where(configs_positions[:, :, 0] < self.env.xlimit[0])[0]) > 0 or \
#                     len(np.where(configs_positions[:, :, 1] < self.env.ylimit[0])[0]) > 0 or \
#                     len(np.where(configs_positions[:, :, 0] > self.env.xlimit[1])[0]) > 0 or \
#                     len(np.where(configs_positions[:, :, 1] > self.env.ylimit[1])[0]) > 0:
#                 return False
#
#         return True
#
#     def get_inspected_points(self, config):
#         '''
#         A function to compute the set of points that are visible to the robot with the given configuration.
#         The function will return the set of points that is visible in terms of distance and field of view (FOV) and are not hidden by any obstacle.
#         @param config The given configuration of the robot.
#         '''
#         # get robot end-effector position and orientation for point of view
#         ee_pos = self.compute_forward_kinematics(given_config=config)[-1]
#         ee_angle = self.compute_ee_angle(given_config=config)
#
#         # define angle range for the ee given its position and field of view (FOV)
#         ee_angle_range = np.array([ee_angle - self.ee_fov / 2, ee_angle + self.ee_fov / 2])
#
#         # iterate over all inspection points to find which of them are currently inspected
#         inspected_points = np.array([])
#         for inspection_point in self.env.inspection_points:
#
#             # compute angle of inspection point w.r.t. position of ee
#             relative_inspection_point = inspection_point - ee_pos
#             inspection_point_angle = self.compute_angle_of_vector(vec=relative_inspection_point)
#
#             # check that the point is potentially visible with the distance from the end-effector
#             if np.linalg.norm(relative_inspection_point) <= self.vis_dist:
#
#                 # if the resulted angle is between the angle range of the ee, verify that there are no interfering obstacles
#                 if self.check_if_angle_in_range(angle=inspection_point_angle, ee_range=ee_angle_range):
#
#                     # define the segment between the inspection point and the ee
#                     ee_to_inspection_point = LineString(
#                         [Point(ee_pos[0], ee_pos[1]), Point(inspection_point[0], inspection_point[1])])
#
#                     # check if there are any collisions of the vector with some obstacle edge
#                     inspection_point_hidden = False
#                     for obstacle_edges in self.env.obstacles_edges:
#                         for obstacle_edge in obstacle_edges:
#                             if ee_to_inspection_point.intersects(obstacle_edge):
#                                 inspection_point_hidden = True
#
#                     # if inspection point is not hidden by any obstacle, add it to the visible inspection points
#                     if not inspection_point_hidden:
#                         if len(inspected_points) == 0:
#                             inspected_points = np.array([inspection_point])
#                         else:
#                             inspected_points = np.concatenate([inspected_points, [inspection_point]], axis=0)
#
#         return inspected_points
#
#     def compute_angle_of_vector(self, vec):
#         '''
#         A utility function to compute the angle of the vector from the end-effector to a point.
#         @param vec Vector from the end-effector to a point.
#         '''
#         vec = vec / np.linalg.norm(vec)
#         if vec[1] > 0:
#             return np.arccos(vec[0])
#         else:  # vec[1] <= 0
#             return -np.arccos(vec[0])
#
#     def check_if_angle_in_range(self, angle, ee_range):
#         '''
#         A utility function to check if an inspection point is inside the FOV of the end-effector.
#         @param angle The angle beteen the point and the end-effector.
#         @param ee_range The FOV of the end-effector.
#         '''
#         # ee range is in the expected order
#         if abs((ee_range[1] - self.ee_fov) - ee_range[0]) < 1e-5:
#             if angle < ee_range.min() or angle > ee_range.max():
#                 return False
#         # ee range reached the point in which pi becomes -pi
#         else:
#             if angle > ee_range.min() or angle < ee_range.max():
#                 return False
#
#         return True
#
#     def compute_union_of_points(self, points1, points2):
#         '''
#         Compute a union of two sets of inpection points.
#         @param points1 list of inspected points.
#         @param points2 list of inspected points.
#         '''
#         # TODO: HW3 2.3.2
#         pass
#
#     def compute_coverage(self, inspected_points):
#         '''
#         Compute the coverage of the map as the portion of points that were already inspected.
#         @param inspected_points list of inspected points.
#         '''
#         return len(inspected_points) / len(self.env.inspection_points)
