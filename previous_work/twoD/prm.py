import time

import numpy as np
import networkx as nx
from scipy.spatial import KDTree
import math
import matplotlib.pyplot as plt
from twoD.building_blocks import BuildingBlocks2D

class PRMController:
    bb : BuildingBlocks2D

    def __init__(self, start, goal, bb, weighted=True):
        self.graph = nx.Graph()
        self.bb = bb 
        self.start = start
        self.goal = goal
        self.coordinates_history = np.concatenate([[start,goal],self.gen_coords(700)])

        self.id = 0
        self.kdtree = KDTree([start, goal])
        # Feel free to add class variables as you wish
        self.weighted=weighted

    def run_PRM(self, num_coords=100, k=5, not_batch=False):
        """
            find a plan to get from current config to destination
            return-the found plan and None if couldn't find
        """
        base_number = self.graph.order()
        how_many_to_add = num_coords - base_number + 2 # start and goal
        if how_many_to_add <= 0 or not_batch:
            base_number = 0
            how_many_to_add = num_coords + 2
            self.clear_graph()
        self.create_graph(base_number, how_many_to_add, k)
        return self.shortest_path()
    
    def create_graph(self, base_number, how_many_to_add, k):
        self.kdtree = KDTree(self.coordinates_history[:base_number + how_many_to_add])
            
        if self.id != base_number:
            print("WARNING! self.id != base_nuber")
            self.id = base_number
        self.add_to_graph(configs=self.coordinates_history[base_number:base_number+how_many_to_add], k=k)

    def clear_graph(self):
        self.graph.clear()
        self.id = 0

    def gen_coords(self, n=5):
        """
        Generate 'n' random collision-free samples called milestones.
        n: number of collision-free configurations to generate
        """
        milestones = np.empty((n, 4))
        i = 0
        low = np.array([0, -np.pi, -np.pi, -np.pi])
        high = np.array([np.pi/2, np.pi, np.pi, np.pi])
        while i < n:
            new_milestone = np.random.uniform(low, high)
            if self.bb.config_validity_checker(new_milestone):
                milestones[i] = new_milestone
                i += 1

        return milestones

    def _inc_id(self):
        self.id += 1

    def add_to_graph(self, configs, k):
        """
            add new configs to the graph.
        """
        for p in configs:
            self.graph.add_node(self.id, point=p)
            weights, neighbors = self.find_nearest_neighbour(p, k)
            for weight, neighbor in zip(weights, neighbors): #type: ignore
                if self.bb.edge_validity_checker(p, self.coordinates_history[neighbor]):
                    self.graph.add_edge(self.id, neighbor, weight = weight if self.weighted else self.bb.compute_distance(p, self.coordinates_history[neighbor]))
 
            self._inc_id()
            

    def find_nearest_neighbour(self, config, k=5):
        """
            Find the k nearest neighbours to config
        """
        weights, nodes = self.kdtree.query(config, k+1)

        return weights[1:], nodes[1:] #type: ignore

    def shortest_path(self):
        """
            Find the shortest path from start to goal using Dijkstra's algorithm (you can use previous implementation from HW1)'
        """
        try:
            node_path = nx.shortest_path(self.graph, 0, 1, weight="weight")
        except nx.NetworkXNoPath:
            return []
        return [self.coordinates_history[node] for node in node_path]
    