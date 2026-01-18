import itertools
import numpy as np
from shapely.geometry import Point, LineString

class BuildingBlocks2D(object):

    def __init__(self, env):
        self.env = env
        # define robot properties
        self.links = np.array([80.0, 70.0, 40.0, 40.0])
        self.dim = len(self.links)

        # robot field of fiew (FOV) for inspecting points, from [-np.pi/6, np.pi/6]
        self.ee_fov = np.pi / 3

        # visibility distance for the robot's end-effector. Farther than that, the robot won't see any points.
        self.vis_dist = 60.0

    def compute_distance(self, prev_config, next_config):
        '''
        Compute the euclidean distance betweeen two given configurations.
        @param prev_config Previous configuration.
        @param next_config Next configuration.
        '''
        x, y = 0, 0
        prev_angle = 0
        next_angle = 0
        for i in range(len(prev_config)):
            prev_angle = self.compute_link_angle(prev_config[i], prev_angle)
            next_angle = self.compute_link_angle(next_config[i], next_angle)
            x += self.links[i] * (np.cos(prev_angle) - np.cos(next_angle))
            y += self.links[i] * (np.sin(prev_angle) - np.sin(next_angle))
        return np.sqrt(x ** 2 + y ** 2)

    def compute_path_cost(self, path):
        if len(path) == 0:
            return -1
        totat_cost = 0
        for i in range(len(path) - 1):
            totat_cost += self.compute_distance(path[i], path[i + 1])
        return totat_cost

    def compute_forward_kinematics(self, given_config):
        '''
        Compute the 2D position (x,y) of each one of the links (including end-effector) and return.
        @param given_config Given configuration.
        '''
        x, y = 0, 0
        angle = 0
        coords = np.zeros((4,2))
        for i in range(len(given_config)):
            angle = self.compute_link_angle(given_config[i], angle)
            x += self.links[i] * np.cos(angle)
            y += self.links[i] * np.sin(angle)
            coords[i][0] = x
            coords[i][1] = y

        return coords

    def compute_ee_angle(self, given_config):
        '''
        Compute the 1D orientation of the end-effector w.r.t. world origin (or first joint)
        @param given_config Given configuration.
        '''
        ee_angle = given_config[0]
        for i in range(1, len(given_config)):
            ee_angle = self.compute_link_angle(ee_angle, given_config[i])

        return ee_angle

    def compute_link_angle(self, link_angle, given_angle):
        '''
        Compute the 1D orientation of a link given the previous link and the current joint angle.
        @param link_angle previous link angle.
        @param given_angle Given joint angle.
        '''
        if link_angle + given_angle > np.pi:
            return link_angle + given_angle - 2 * np.pi
        elif link_angle + given_angle < -np.pi:
            return link_angle + given_angle + 2 * np.pi
        else:
            return link_angle + given_angle

    def validate_robot(self, robot_positions):
        '''
        Verify that the given set of links positions does not contain self collisions.
        @param robot_positions Given links positions.
        '''
        pts2 = robot_positions.tolist()
        pts = [[0.0,0.0]]
        pts.extend(pts2)
        robot_links = [
            LineString([pts[i], pts[i + 1]])
            for i in range(len(pts) - 1)
        ]

        for i in range(len(robot_links)):
            for j in range(i + 2, len(robot_links)):  # skip self & adjacent links
                if robot_links[i].intersects(robot_links[j]):
                    return False

        return True


    def config_validity_checker(self, config):
        '''
        Verify that the config (given or stored) does not contain self collisions or links that are out of the world boundaries.
        Return false if the config is not applicable, and true otherwise.
        @param config The given configuration of the robot.
        '''
        # compute robot links positions
        robot_positions = self.compute_forward_kinematics(given_config=config)

        # add position of robot placement ([0,0] - position of the first joint)
        robot_positions = np.concatenate([np.zeros((1,2)), robot_positions])

        # verify that the robot do not collide with itself
        if not self.validate_robot(robot_positions=robot_positions):
            return False

        # verify that all robot joints (and links) are between world boundaries
        non_applicable_poses = [(x[0] < self.env.xlimit[0] or x[1] < self.env.ylimit[0] or x[0] > self.env.xlimit[1] or x[1] > self.env.ylimit[1]) for x in robot_positions]
        if any(non_applicable_poses):
            return False

        # verify that all robot links do not collide with obstacle edges
        # for each obstacle, check collision with each of the robot links
        robot_links = [LineString([Point(x[0],x[1]),Point(y[0],y[1])]) for x,y in zip(robot_positions.tolist()[:-1], robot_positions.tolist()[1:])]
        for obstacle_edges in self.env.obstacles_edges:
            for robot_link in robot_links:
                obstacle_collisions = [robot_link.crosses(x) for x in obstacle_edges]
                if any(obstacle_collisions):
                    return False

        return True

    def edge_validity_checker(self, config1, config2):
        '''
        A function to check if the edge between two configurations is free from collisions. The function will interpolate between the two states to verify
        that the links during motion do not collide with anything.
        @param config1 The source configuration of the robot.
        @param config2 The destination configuration of the robot.
        '''
        # interpolate between first config and second config to verify that there is no collision during the motion
        required_diff = 0.05
        interpolation_steps = int(np.linalg.norm(config2 - config1) // required_diff)
        if interpolation_steps > 0:
            interpolated_configs = np.linspace(start=config1, stop=config2, num=interpolation_steps)

            # compute robot links positions for interpolated configs
            configs_positions = np.apply_along_axis(self.compute_forward_kinematics, 1, interpolated_configs)

            # compute edges between joints to verify that the motion between two configs does not collide with anything
            edges_between_positions = []
            for j in range(self.dim):
                for i in range(interpolation_steps - 1):
                    edges_between_positions.append(LineString(
                        [Point(configs_positions[i, j, 0], configs_positions[i, j, 1]),
                         Point(configs_positions[i + 1, j, 0], configs_positions[i + 1, j, 1])]))

            # check collision for each edge between joints and each obstacle
            for edge_pos in edges_between_positions:
                for obstacle_edges in self.env.obstacles_edges:
                    obstacle_collisions = [edge_pos.crosses(x) for x in obstacle_edges]
                    if any(obstacle_collisions):
                        return False

            # add position of robot placement ([0,0] - position of the first joint)
            configs_positions = np.concatenate([np.zeros((len(configs_positions), 1, 2)), configs_positions], axis=1)

            # verify that the robot do not collide with itself during motion
            for config_positions in configs_positions:
                if not self.validate_robot(config_positions):
                    return False

            # verify that all robot joints (and links) are between world boundaries
            if len(np.where(configs_positions[:, :, 0] < self.env.xlimit[0])[0]) > 0 or \
                    len(np.where(configs_positions[:, :, 1] < self.env.ylimit[0])[0]) > 0 or \
                    len(np.where(configs_positions[:, :, 0] > self.env.xlimit[1])[0]) > 0 or \
                    len(np.where(configs_positions[:, :, 1] > self.env.ylimit[1])[0]) > 0:
                return False

        return True
