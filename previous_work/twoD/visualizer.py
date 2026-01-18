import imageio
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt
import matplotlib


class Visualizer:
    def __init__(self, bb):
        self.bb = bb

    def visualize_map(self, config, plan=None, show_map=True):
        '''
        Visualize map with current config of robot and obstacles in the map.
        @param config The requested configuration of the robot.
        @param show_map If to show the map or not.
        '''
        # create empty background
        plt = self.create_map_visualization()

        # add obstacles
        plt = self.visualize_obstacles(plt=plt)

        # add robot
        plt = self.visualize_robot(plt=plt, config=config)

        # add plan if given
        if plan is not None:
            plt = self.visualize_plan(plt=plt, plan=plan, color='navy')

        # show map
        if show_map:
            plt.show()  # replace savefig with show if you want to display map actively
            # plt.savefig('map.png')

        return plt

    def create_map_visualization(self):
        '''
        Prepare the plot of the scene for visualization.
        '''
        # create figure and add background
        plt.figure()
        back_img = np.zeros((self.bb.env.ylimit[1] + 1, self.bb.env.xlimit[1] + 1))
        plt.imshow(back_img, origin='lower', zorder=0)

        return plt

    def visualize_obstacles(self, plt):
        '''
        Draw the scene's obstacles on top of the given frame.
        @param plt Plot of a frame of the plan.
        '''
        # plot obstacles
        for obstacle in self.bb.env.obstacles:
            obstacle_xs, obstacle_ys = zip(*obstacle)
            plt.fill(obstacle_xs, obstacle_ys, "y", zorder=5)

        return plt

    def visualize_plan(self, plt, plan, color):
        '''
        Draw a given plan on top of the given frame.
        @param plt Plot of a frame of the environment.
        @param plan The requested sequence of steps.
        @param color The requested color for the plan.
        '''
        # add plan edges to the plt
        for i in range(0, len(plan) - 1):
            plt.plot([plan[i, 0], plan[i + 1, 0]], [plan[i, 1], plan[i + 1, 1]], color=color, linewidth=1,
                     zorder=20)

        return plt

    def visualize_robot(self, plt, config):
        '''
        Draw the robot on top of the plt.
        @param plt Plot of a frame of the plan.
        @param config The requested configuration of the robot.
        '''
        # get robot joints and end-effector positions.
        robot_positions = self.bb.compute_forward_kinematics(given_config=config)

        # add position of robot placement ([0,0] - position of the first joint)
        robot_positions = np.concatenate([np.zeros((1, 2)), robot_positions])

        # draw the robot
        plt.plot(robot_positions[:, 0], robot_positions[:, 1], 'coral', linewidth=3.0, zorder=10)  # joints
        plt.scatter(robot_positions[:, 0], robot_positions[:, 1], zorder=15)  # joints
        plt.scatter(robot_positions[-1:, 0], robot_positions[-1:, 1], color='cornflowerblue',
                    zorder=15)  # end-effector

        return plt

    def interpolate_plan(self, plan_configs):
        '''
        Interpolate plan of configurations - add steps between each to configs to make visualization smoother.
        @param plan_configs Sequence of configs defining the plan.
        '''
        required_diff = 0.05

        # interpolate configs list
        plan_configs_interpolated = []
        for i in range(len(plan_configs) - 1):
            # number of steps to add from i to i+1
            interpolation_steps = int(np.linalg.norm(plan_configs[i + 1] - plan_configs[i]) // required_diff) + 1
            interpolated_configs = np.linspace(start=plan_configs[i], stop=plan_configs[i + 1], endpoint=False,
                                               num=interpolation_steps)
            plan_configs_interpolated += list(interpolated_configs)

        # add goal vertex
        plan_configs_interpolated.append(plan_configs[-1])

        return plan_configs_interpolated

    def visualize_point_location(self, plt, config, color):
        '''
        Draw a point of start/goal on top of the given frame.
        @param plt Plot of a frame of the plan.
        @param config The requested configuration of the point.
        @param color The requested color for the point.
        '''
        # compute point location in 2D
        point_loc = self.bb.compute_forward_kinematics(given_config=config)[-1]

        # draw the circle
        point_circ = plt.Circle(point_loc, radius=5, color=color, zorder=5)
        plt.gca().add_patch(point_circ)

        return plt

    def visualize_plan_as_gif(self, plan):
        '''
        Visualize the final plan as a GIF and stores it.
        @param plan Sequence of configs defining the plan.
        '''
        # switch backend - possible bugfix if animation fails
        matplotlib.use('TkAgg')

        # interpolate plan and get inspected points
        plan = self.interpolate_plan(plan_configs=plan)

        # visualize each step of the given plan
        plan_images = []
        for i in range(len(plan)):
            # create background, obstacles, start
            plt = self.create_map_visualization()
            plt = self.visualize_obstacles(plt=plt)
            plt = self.visualize_point_location(plt=plt, config=plan[0], color='r')

            # add goal or inspection points
            plt = self.visualize_point_location(plt=plt, config=plan[-1], color='g')

            # add robot with current plan step
            plt = self.visualize_robot(plt=plt, config=plan[i])

            # convert plot to image
            canvas = plt.gca().figure.canvas
            canvas.draw()
            # Use tostring_argb() as suggested by the error, which is compatible with TkAgg.
            # This returns 4 channels (Alpha, R, G, B).
            data = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)
            data = data.reshape(canvas.get_width_height()[::-1] + (4,))
            # Slice the array to keep only the RGB channels, discarding the Alpha channel.
            plan_images.append(data[:, :, 1:])
            plt.close()

        # store gif
        plan_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        imageio.mimsave(f'plan_{plan_time}.gif', plan_images, 'GIF', duration=0.05)