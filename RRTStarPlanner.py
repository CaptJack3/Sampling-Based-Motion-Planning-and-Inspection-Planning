import numpy as np
from RRTTree import RRTTree
import time


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
        Compute and return the plan. The function should return a numpy array containing the states (positions) of the robot.
        """
        # TODO: HW3 3
        pass

    def compute_cost(self, plan):
        # TODO: HW3 3
        pass

    def extend(self, x_near, x_rand):
        # TODO: HW3 3
        pass
