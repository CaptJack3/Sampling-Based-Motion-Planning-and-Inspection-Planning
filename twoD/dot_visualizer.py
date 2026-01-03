import imageio
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt
import matplotlib


class DotVisualizer:
    def __init__(self, bb):
        self.bb = bb

    def visualize_map(self, show_map=False, plan=None, tree_edges=None, expanded_nodes=None, start=None, goal=None):
        '''
        Visualize map with current state of robot and obstacles in the map.
        @param show_map If to show the map or save it.
        @param plan A given plan to draw for the robot.
        @param tree_edges A set of tree edges to draw.
        @param expanded_nodes A set of expanded nodes to draw.
        '''
        # create empty background
        plt = self.create_map_visualization()

        # add obstacles
        plt = self.visualize_obstacles(plt=plt)

        # add plan if given
        if plan is not None:
            plt = self.visualize_plan(plt=plt, plan=plan, color='navy')

        # add tree edges if given
        if tree_edges is not None:
            plt = self.visualize_tree_edges(plt=plt, tree_edges=tree_edges, color='lightgrey')

        # add expanded nodes if given
        if expanded_nodes is not None:
            plt = self.visualize_expanded_nodes(plt=plt, expanded_nodes=expanded_nodes, color='lightgrey')

        if start is not None:
            plt = self.visualize_point_location(plt=plt, state=start, color='r')

        if goal is not None:
            plt = self.visualize_point_location(plt=plt, state=goal, color='g')

        # show map
        if show_map:
            plt.show()
        else:
            plt.savefig('map.png')

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
        @param plt Plot of a frame of the environment.
        '''
        # plot obstacles
        for obstacle in self.bb.env.obstacles:
            obstacle_xs, obstacle_ys = zip(*list(obstacle.exterior.coords))
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
            plt.plot([plan[i, 0], plan[i + 1, 0]], [plan[i, 1], plan[i + 1, 1]], color=color, linewidth=1, zorder=20)

        return plt

    def visualize_tree_edges(self, plt, tree_edges, color):
        '''
        Draw the set of the given tree edges on top of the given frame.
        @param plt Plot of a frame of the environment.
        @param tree_edges The requested set of edges.
        @param color The requested color for the plan.
        '''
        # add plan edges to the plt
        for tree_edge in tree_edges:
            plt.plot([tree_edge[0][0], tree_edge[1][0]], [tree_edge[0][1], tree_edge[1][1]], color=color, zorder=10)

        return plt

    def visualize_expanded_nodes(self, plt, expanded_nodes, color):
        '''
        Draw the set of the given expanded nodes on top of the given frame.
        @param plt Plot of a frame of the environment.
        @param expanded_nodes The requested set of expanded states.
        @param color The requested color for the plan.
        '''
        # add plan edges to the plt
        point_radius = 0.5
        for expanded_node in expanded_nodes:
            point_circ = plt.Circle(expanded_node, radius=point_radius, color=color, zorder=10)
            plt.gca().add_patch(point_circ)

        return plt

    def visualize_point_location(self, plt, state, color):
        '''
        Draw a point of start/goal on top of the given frame.
        @param plt Plot of a frame of the environment.
        @param state The requested state.
        @param color The requested color for the point.
        '''

        # draw the circle
        point_radius = 0.5
        point_circ = plt.Circle(state, radius=point_radius, color=color, zorder=30)
        plt.gca().add_patch(point_circ)

        return plt