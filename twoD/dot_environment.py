import os
import json
import numpy as np
from shapely.geometry import Point, LineString, Polygon


class MapDotEnvironment(object):

    def __init__(self, json_file):

        # check if json file exists and load
        json_path = os.path.join(os.getcwd(), json_file)
        if not os.path.isfile(json_path):
            raise ValueError('Json file does not exist!');
        with open(json_path) as f:
            json_dict = json.load(f)

        # obtain boundary limits, start and inspection points
        self.xlimit = [0, json_dict['WIDTH'] - 1]
        self.ylimit = [0, json_dict['HEIGHT'] - 1]
        self.load_obstacles(obstacles=json_dict['OBSTACLES'])


    def load_obstacles(self, obstacles):
        '''
        A function to load and verify scene obstacles.
        @param obstacles A list of lists of obstacles points.
        '''
        # iterate over all obstacles
        self.obstacles, self.obstacles_edges = [], []
        for obstacle in obstacles:
            non_applicable_vertices = [
                x[0] < self.xlimit[0] or x[0] > self.xlimit[1] or x[1] < self.ylimit[0] or x[1] > self.ylimit[1] for x
                in obstacle]
            if any(non_applicable_vertices):
                raise ValueError('An obstacle coincides with the maps boundaries!');

            # make sure that the obstacle is a closed form
            if obstacle[0] != obstacle[-1]:
                obstacle.append(obstacle[0])
                self.obstacles_edges.append(
                    [LineString([Point(x[0], x[1]), Point(y[0], y[1])]) for (x, y) in zip(obstacle[:-1], obstacle[1:])])
            self.obstacles.append(Polygon(obstacle))

    def config_validity_checker(self, state):
        '''
        Verify that the state is in the world boundaries, and is not inside an obstacle.
        Return false if the state is not applicable, and true otherwise.
        @param state The given position of the robot.
        '''

        # make sure robot state is a numpy array
        if not isinstance(state, np.ndarray):
            state = np.array(state)

        # verify that the robot position is between world boundaries
        if state[0] < self.xlimit[0] or state[1] < self.ylimit[0] or state[0] > self.xlimit[1] or state[1] > \
                self.ylimit[1]:
            return False

        # verify that the robot is not positioned inside an obstacle
        for obstacle in self.obstacles:
            if obstacle.intersects(Point(state[0], state[1])):
                return False

        return True

    def edge_validity_checker(self, state1, state2):
        '''
        A function to check if the edge between two states is free from collisions. The function will return False if the edge intersects another obstacle.
        @param state1 The source state of the robot.
        @param state2 The destination state of the robot.
        '''

        # define undirected edge
        given_edge = LineString([state1, state2])

        # verify that the robot does not crossing any obstacle
        for obstacle in self.obstacles:
            if given_edge.intersects(obstacle):
                return False

        return True

    def compute_distance(self, prev_config, next_config):
        '''
        Compute the euclidean distance betweeen two given configurations.
        @param prev_config Previous configuration.
        @param next_config Next configuration.
        '''
        return np.linalg.norm(np.array(next_config) - np.array(prev_config))

