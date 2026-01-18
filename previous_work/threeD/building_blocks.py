import numpy as np


class BuildingBlocks3D(object):
    '''
    @param resolution determines the resolution of the local planner(how many intermidiate configurations to check)
    @param p_bias determines the probability of the sample function to return the goal configuration
    '''

    def __init__(self, transform, ur_params, env, resolution=0.1):
        self.transform = transform
        self.ur_params = ur_params
        self.env = env
        self.resolution = resolution
        # self.p_bias = p_bias
        self.cost_weights = np.array([0.4, 0.3, 0.2, 0.1, 0.07, 0.05])

        self.single_mechanical_limit = list(self.ur_params.mechanical_limits.values())[-1][-1]

        # pairs of links that can collide during sampling
        self.possible_link_collisions = [['shoulder_link', 'forearm_link'],
                                         ['shoulder_link', 'wrist_1_link'],
                                         ['shoulder_link', 'wrist_2_link'],
                                         ['shoulder_link', 'wrist_3_link'],
                                         ['upper_arm_link', 'wrist_1_link'],
                                         ['upper_arm_link', 'wrist_2_link'],
                                         ['upper_arm_link', 'wrist_3_link'],
                                         ['forearm_link', 'wrist_2_link'],
                                         ['forearm_link', 'wrist_3_link']]

    def sample_random_config(self, goal_prob,  goal_conf) -> np.ndarray:
        """
        sample random configuration
        @param goal_conf - the goal configuration
        :param goal_prob - the probability that goal should be sampled
        """
        if np.random.rand() < goal_prob:
            return np.array(goal_conf, dtype=float)
        config =[]
        for link in self.ur_params.mechanical_limits.keys():
            low, high = self.ur_params.mechanical_limits[link]
            config.append(np.random.uniform(low, high))
        return np.array(config, dtype=float)

    def config_validity_checker(self, conf) -> bool:
        """check for collision in given configuration, arm-arm and arm-obstacle
        return False if in collision
        @param conf - some configuration
        """

        Check_External_collisions = True


        global_sphere_coords = self.transform.conf2sphere_coords(conf)
        # --- 2) INTERNAL COLLISIONS (link–link collisions) ---

        for linkA, linkB in self.possible_link_collisions:


            spheres_A = global_sphere_coords[linkA]
            spheres_B = global_sphere_coords[linkB]

            rA = self.ur_params.sphere_radius[linkA]
            rB = self.ur_params.sphere_radius[linkB]
            rad_sum = rA + rB

            for pA in spheres_A:
                for pB in spheres_B:
                    if np.linalg.norm(pA - pB) < rad_sum:
                        return False  # collision detected


        if Check_External_collisions:
            # --- 3) EXTERNAL COLLISIONS (robot–obstacle collisions) ---
            obstacle_centers = self.env.obstacles
            obstacle_r = self.env.radius

            for link_name in global_sphere_coords:
                r_robot = self.ur_params.sphere_radius[link_name]

                for p_robot in global_sphere_coords[link_name]:
                    for p_obs in obstacle_centers:
                        if np.linalg.norm(p_robot - p_obs) < (r_robot + obstacle_r):
                            return False

        # No collisions found:
        return True


    def edge_validity_checker(self, prev_conf, current_conf) -> bool:
        '''check for collisions between two configurations - return True if trasition is valid
        @param prev_conf - some configuration
        @param current_conf - current configuration
        '''
        prev_conf = np.array(prev_conf, dtype=float) 
        current_conf = np.array(current_conf, dtype=float)
        #
        total_dist = np.linalg.norm(current_conf - prev_conf)
        num_steps = max(3, int(np.ceil(total_dist / self.resolution)) + 1)
        print(f"Number of steps is {num_steps}")
        for alpha in np.linspace(0, 1, num_steps):
            q = prev_conf + alpha * (current_conf - prev_conf)
            if not self.config_validity_checker(q):
                return False
        return True

    def compute_distance(self, conf1, conf2):
        '''
        Returns the Edge cost- the cost of transition from configuration 1 to configuration 2
        @param conf1 - configuration 1
        @param conf2 - configuration 2
        '''
        return np.dot(self.cost_weights, np.power(conf1 - conf2, 2)) ** 0.5
