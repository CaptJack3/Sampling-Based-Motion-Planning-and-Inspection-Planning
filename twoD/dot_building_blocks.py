import itertools
import numpy as np
from shapely.geometry import Point, LineString

class DotBuildingBlocks2D(object):

    def __init__(self, env):
        self.env = env
        # robot field of view (FOV) for inspecting points, from [-np.pi/6, np.pi/6]
        self.ee_fov = np.pi / 3

        # visibility distance for the robot's end-effector. Farther than that, the robot won't see any points.
        self.vis_dist = 60.0

    def compute_distance(self, prev_config, next_config):
        """
        Euclidean distance between two 2D states.
        prev_config, next_config: array-like of shape (2,)
        """
        p = np.asarray(prev_config, dtype=float)
        q = np.asarray(next_config, dtype=float)
        return float(np.linalg.norm(q - p))

    def sample_random_config(self, goal_prob, goal):
        """
        With probability goal_prob return the goal (goal biasing).
        Otherwise, uniformly sample a valid point in the map bounds.
        """
        # Goal biasing
        if np.random.rand() < goal_prob:
            return np.asarray(goal, dtype=float)

        # Uniform sampling within environment bounds
        x_min, x_max = self.env.xlimit
        y_min, y_max = self.env.ylimit

        # Rejection sampling until valid (usually fast in dot maps)
        for _ in range(10_000):
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)
            s = np.array([x, y], dtype=float)
            if self.config_validity_checker(s):
                return s

        # Fallback (should basically never happen unless map is fully blocked)
        return np.asarray(goal, dtype=float)

    def config_validity_checker(self, state):
        return self.env.config_validity_checker(state)

    def edge_validity_checker(self, state1, state2):
        return self.env.edge_validity_checker(state1, state2)


