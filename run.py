import cProfile, pstats, io
import time
import numpy as np
import os
from datetime import datetime
from twoD.environment import MapEnvironment
from twoD.dot_environment import MapDotEnvironment
from twoD.dot_building_blocks import DotBuildingBlocks2D
from twoD.building_blocks import BuildingBlocks2D
from twoD.dot_visualizer import DotVisualizer
from threeD.environment import Environment
from threeD.kinematics import UR5e_PARAMS, Transform
from threeD.building_blocks import BuildingBlocks3D
from threeD.visualizer import Visualize_UR
from AStarPlanner import AStarPlanner
from RRTMotionPlanner import RRTMotionPlanner
from RRTInspectionPlanner import RRTInspectionPlanner
from RRTStarPlanner import RRTStarPlanner,RRTStarExperiment,RRTStarExperiment_path,JRRTStarPlanner
from twoD.visualizer import Visualizer

# MAP_DETAILS = {"json_file": "twoD/map1.json", "start": np.array([10,10]), "goal": np.array([4, 6])}
MAP_DETAILS = {"json_file": "twoD/map2.json", "start": np.array([360, 150]), "goal": np.array([100, 200])}


def run_dot_2d_astar():
    planning_env = MapDotEnvironment(json_file=MAP_DETAILS["json_file"])
    bb = DotBuildingBlocks2D(planning_env)
    planner = AStarPlanner(bb=bb, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"])

    # execute plan
    plan = planner.plan()
    DotVisualizer(bb).visualize_map(plan=plan, expanded_nodes=planner.expanded_nodes, show_map=True, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"])

def run_dot_2d_rrt():
    planning_env = MapDotEnvironment(json_file=MAP_DETAILS["json_file"])
    bb = DotBuildingBlocks2D(planning_env)
    planner = JRRTStarPlanner(max_step_size=0.3,bb=bb, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"], ext_mode="E2", goal_prob=0.01)

    # execute plan
    plan = planner.plan()
    DotVisualizer(bb).visualize_map(plan=plan, tree_edges=planner.tree.get_edges_as_states(), show_map=True)

def run_dot_2d_rrt_star():
    planning_env = MapDotEnvironment(json_file=MAP_DETAILS["json_file"])
    bb = DotBuildingBlocks2D(planning_env)
    planner = JRRTStarPlanner(max_step_size=1.5,max_itr=20000,bb=bb, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"], ext_mode="E2", goal_prob=0.2, k=None)



    # execute plan
    plan = planner.plan()
    DotVisualizer(bb).visualize_map(plan=plan, tree_edges=planner.tree.get_edges_as_states(), show_map=True)

def run_2d_rrt_star_motion_planning():
    MAP_DETAILS = {
        "json_file": "twoD/map_mp.json",
        "start": np.array([0.78, -0.78, 0.0, 0.0]),
        "goal": np.array([0.3, 0.15, 1.0, 1.1]),}


    # MAP_DETAILS = {
    #     "json_file": "twoD/map_mp.json",
    #     "start": np.array([0.78, -0.78, 0.0, 0.0]),
    #     "goal": np.array([0.78, -0.78, -0.6, 0.6]),}



    planning_env = MapEnvironment(json_file=MAP_DETAILS["json_file"], task="mp")
    bb = BuildingBlocks2D(planning_env)

    planner = JRRTStarPlanner(max_step_size=0.8,max_itr=10000,bb=bb,stop_on_goal=True, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"], ext_mode="E2", goal_prob=0.2, k=None)


    # execute plan
    plan = planner.plan()
    print(f"path is \n {[plan]}")

    print(f"Planner ended now visualization:")
    if plan is not None:
        print(f"Planner ended now visualization:")

        Visualizer(bb).visualize_plan(plan=plan, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"])

def run_2d_rrt_motion_planning():
    MAP_DETAILS = {"json_file": "twoD/map_mp.json", "start": np.array([0.78, -0.78, 0.0, 0.0]), "goal": np.array([0.3, 0.15, 1.0, 1.1])}
    planning_env = MapEnvironment(json_file=MAP_DETAILS["json_file"], task="mp")
    bb = BuildingBlocks2D(planning_env)
    #planner = RRTMotionPlanner(bb=bb, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"], ext_mode="E2", goal_prob=0.01)

    planner = JRRTStarPlanner(max_step_size=1.0,max_itr=5000,bb=bb, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"], ext_mode="E2", goal_prob=0.2, k=None)

    # execute plan
    plan = planner.plan()
    print(f"path is \n {[plan]}")
    # Visualizer(bb).visualize_plan(plan=plan, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"])

def run_2d_rrt_inspection_planning():
    MAP_DETAILS = {"json_file": "twoD/map_ip.json", "start": np.array([0.78, -0.78, 0.0, 0.0]), "goal": np.array([0.3, 0.15, 1.0, 1.1])}
    planning_env = MapEnvironment(json_file=MAP_DETAILS["json_file"], task="ip")
    bb = BuildingBlocks2D(planning_env)
    planner = RRTInspectionPlanner(bb=bb, start=MAP_DETAILS["start"], ext_mode="E2", goal_prob=0.01, coverage=0.5)

    # execute plan
    plan = planner.plan()
    Visualizer(bb).visualize_plan(plan=plan, start=MAP_DETAILS["start"])

def run_3d():
    ur_params = UR5e_PARAMS(inflation_factor=1)
    env = Environment(env_idx=2)
    transform = Transform(ur_params)

    bb = BuildingBlocks3D(transform=transform,
                          ur_params=ur_params,
                          env=env,
                          resolution=0.25 ) #resolution was 0.1

    visualizer = Visualize_UR(ur_params, env=env, transform=transform, bb=bb)

    q_test = np.deg2rad([130, -70, 90, -90, -90, 0])
    print(f"bb.config_validity_checker ={bb.config_validity_checker(q_test)}")  # should be False


    # --------- configurations-------------
    env2_start = np.deg2rad([110, -70, 90, -90, -90, 0 ])
    env2_goal = np.deg2rad([50, -80, 90, -90, -90, 0 ])
    # ---------------------------------------

    rrt_star_planner = JRRTStarPlanner(max_step_size=0.3,
                                      start=env2_start,
                                      goal=env2_goal,
                                      max_itr = 2000,
                                      stop_on_goal=False,
                                      bb=bb,
                                      goal_prob=0.05,
                                      ext_mode="E2")
    start_time = time.perf_counter()
    path = rrt_star_planner.plan()
    end_time = time.perf_counter()
    print(f"Execution time: {end_time-start_time:.4f} seconds")

    if path is not None:

        # create a folder for the experiment
        # Format the time string as desired (YYYY-MM-DD_HH-MM-SS)
        now = datetime.now()
        time_str = now.strftime("%Y-%m-%d_%H-%M-%S")

        # create the folder
        exps_folder_name = os.path.join(os.getcwd(), "exps")
        if not os.path.exists(exps_folder_name):
            os.mkdir(exps_folder_name)
        exp_folder_name = os.path.join(exps_folder_name, "exp_pbias_"+ str(rrt_star_planner.goal_prob) + "_max_step_size_" + str(rrt_star_planner.max_step_size) + "_" + time_str)
        if not os.path.exists(exp_folder_name):
            os.mkdir(exp_folder_name)

        # save the path
        np.save(os.path.join(exp_folder_name, 'path'), path)

        # save the cost of the path and time it took to compute
        with open(os.path.join(exp_folder_name, 'stats'), "w") as file:
            file.write("Path cost: {} \n".format(rrt_star_planner.compute_cost(path)))

        # visualizer.show_path(path)

def run_3d_loop():
    ur_params = UR5e_PARAMS(inflation_factor=1)
    env = Environment(env_idx=2)
    transform = Transform(ur_params)

    bb = BuildingBlocks3D(transform=transform,
                          ur_params=ur_params,
                          env=env,
                          resolution=0.1 )

    visualizer = Visualize_UR(ur_params, env=env, transform=transform, bb=bb)

    q_test = np.deg2rad([130, -70, 90, -90, -90, 0])
    print(f"bb.config_validity_checker ={bb.config_validity_checker(q_test)}")  # should be False


    # --------- configurations-------------
    env2_start = np.deg2rad([110, -70, 90, -90, -90, 0 ])
    env2_goal = np.deg2rad([50, -80, 90, -90, -90, 0 ])
    # ---------------------------------------

    rrt_star_planner = RRTStarPlanner(max_step_size=0.5,
                                      start=env2_start,
                                      goal=env2_goal,
                                      max_itr=4000,
                                      stop_on_goal=True,
                                      bb=bb,
                                      goal_prob=0.05,
                                      ext_mode="E2")

    path = rrt_star_planner.plan()

    if path is not None:

        # create a folder for the experiment
        # Format the time string as desired (YYYY-MM-DD_HH-MM-SS)
        now = datetime.now()
        time_str = now.strftime("%Y-%m-%d_%H-%M-%S")

        # create the folder
        exps_folder_name = os.path.join(os.getcwd(), "exps")
        if not os.path.exists(exps_folder_name):
            os.mkdir(exps_folder_name)
        exp_folder_name = os.path.join(exps_folder_name, "exp_pbias_"+ str(rrt_star_planner.goal_prob) + "_max_step_size_" + str(rrt_star_planner.max_step_size) + "_" + time_str)
        if not os.path.exists(exp_folder_name):
            os.mkdir(exp_folder_name)

        # save the path
        np.save(os.path.join(exp_folder_name, 'path'), path)

        # save the cost of the path and time it took to compute
        with open(os.path.join(exp_folder_name, 'stats'), "w") as file:
            file.write("Path cost: {} \n".format(rrt_star_planner.compute_cost(path)))


def run_profiled():
    pr = cProfile.Profile()
    pr.enable()

    run_3d()   # or run_3d_loop()

    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
    ps.print_stats(30)
    print(s.getvalue())


import json
import os
import numpy as np
from datetime import datetime

# Import your classes (adjust paths as needed)
# from threeD.environment import Environment
# from threeD.kinematics import UR5e_PARAMS, Transform
# from threeD.building_blocks import BuildingBlocks3D
# from RRTStarPlanner import RRTStarExperiment  # The class we just made


def run_3d_experiment():
    # 1. Setup Environment
    ur_params = UR5e_PARAMS(inflation_factor=1)
    env = Environment(env_idx=2)  # Guidance says env_idx=2
    transform = Transform(ur_params)
    bb = BuildingBlocks3D(transform=transform, ur_params=ur_params, env=env, resolution=0.25)

    # 2. Setup Start/Goal (Radians)
    start_conf = np.deg2rad([110, -70, 90, -90, -90, 0])
    goal_conf = np.deg2rad([50, -80, 90, -90, -90, 0])

    # 3. Parameters from Assignment
    max_step_sizes = [0.05, 0.075, 0.1, 0.125, 0.2, 0.25, 0.3, 0.4]
    p_biases = [0.05, 0.2]

    max_step_sizes = [ 0.1, 0.2, 0.3, 0.4]
    max_step_sizes = [0.05, 0.075,  0.125,  0.25]

    p_biases = [0.05, 0.2]
    num_runs = 20
    max_itr = 2000
    report_interval = 400

    # Create results folder
    if not os.path.exists("assignment_results"):
        os.makedirs("assignment_results")

    print(f"Starting Experiment Batch...")

    # --- Nested Loops for Parameters ---
    for p_bias in p_biases:
        for step_size in max_step_sizes:

            print(f"\n--- Testing p_bias={p_bias}, step={step_size} ---")

            # Storage for this case
            all_runs_data = []  # Stores raw [success_list, cost_list] for each run

            # Run 20 times
            for run_idx in range(num_runs):
                print(f"   Run {run_idx + 1}/{num_runs}...", end="", flush=True)

                # Initialize Planner
                planner = RRTStarExperiment(
                    bb=bb,
                    start=start_conf,
                    goal=goal_conf,
                    max_itr=max_itr,
                    ext_mode="E2",
                    max_step_size=step_size,
                    goal_prob=p_bias,
                    stop_on_goal=False,  # Must continue to improve!
                    k=None  # Use auto-k
                )

                # Execute Plan
                success_list, cost_list = planner.plan_with_stats(report_interval=report_interval)

                # Store Run Data
                all_runs_data.append({
                    "run_id": run_idx,
                    "success": success_list,
                    "costs": cost_list
                })

                # Quick status print
                final_cost = cost_list[-1] if cost_list[-1] is not None else "Inf"
                print(f" Done. (Found: {success_list[-1]}, Cost: {final_cost})")

            # --- Aggregation Logic ---
            # We have 20 runs. Each run has lists of length 5 (400, 800, 1200, 1600, 2000).
            num_intervals = max_itr // report_interval
            intervals = [(i + 1) * report_interval for i in range(num_intervals)]

            avg_success_rate = []
            avg_costs_clean = []  # Only average costs where solution was found

            for i in range(num_intervals):
                # 1. Calculate Success Rate at this interval
                success_count = sum(r["success"][i] for r in all_runs_data)
                rate = (success_count / num_runs) * 100.0
                avg_success_rate.append(rate)

                # 2. Calculate Average Cost (only for successful runs)
                valid_costs = [r["costs"][i] for r in all_runs_data if r["costs"][i] is not None]
                if len(valid_costs) > 0:
                    avg_cost = sum(valid_costs) / len(valid_costs)
                else:
                    avg_cost = None  # No solutions yet
                avg_costs_clean.append(avg_cost)

            # --- Save to File ---
            results_payload = {
                "parameters": {
                    "p_bias": p_bias,
                    "max_step_size": step_size,
                    "runs": num_runs
                },
                "stats": {
                    "intervals": intervals,
                    "success_rate_percent": avg_success_rate,
                    "average_cost": avg_costs_clean
                },
                "raw_data": all_runs_data
            }

            # Filename: results_pbias_0.05_step_0.1.json
            fname = f"assignment_results/results_pbias_{p_bias}_step_{step_size}.json"
            with open(fname, "w") as f:
                json.dump(results_payload, f, indent=4)

            print(f"Saved results to {fname}")


import json
import os
import matplotlib.pyplot as plt
import numpy as np

import json
import os
import matplotlib.pyplot as plt
import numpy as np


def plot_assignment_results():
    # 1. Define paths
    # Get the absolute path to ensure we know exactly where we are looking
    current_dir = os.getcwd()
    results_dir = os.path.join(current_dir, "assignment_results")

    print(f"DEBUG: Looking for JSON files in: {results_dir}")

    if not os.path.exists(results_dir):
        print(f"ERROR: The folder '{results_dir}' does not exist.")
        print("Make sure you ran the experiment script first!")
        return

    # Must match the parameters you ran the experiment with
    p_biases = [0.05, 0.2]
    max_step_sizes = [0.05, 0.075, 0.1, 0.125, 0.2, 0.25, 0.3, 0.4]

    cm = plt.get_cmap('tab10')

    # 2. Iterate over p_bias
    for p_bias in p_biases:
        print(f"\n--- Processing p_bias = {p_bias} ---")

        fig_cost, ax_cost = plt.subplots(figsize=(10, 6))
        fig_success, ax_success = plt.subplots(figsize=(10, 6))

        files_found_count = 0

        # 3. Iterate over step_sizes
        for i, step in enumerate(max_step_sizes):
            fname = f"results_pbias_{p_bias}_step_{step}.json"
            fpath = os.path.join(results_dir, fname)

            if not os.path.exists(fpath):
                # Silent skip is okay, but let's print if NO files are found later
                continue

            files_found_count += 1
            print(f"Found: {fname}")

            try:
                with open(fpath, 'r') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Error reading {fname}: {e}")
                continue

            stats = data["stats"]
            intervals = stats["intervals"]
            success_rates = stats["success_rate_percent"]
            avg_costs = stats["average_cost"]

            # Convert None to NaN for plotting
            avg_costs = [np.nan if c is None else c for c in avg_costs]

            color = cm(i % 10)

            # Plot
            ax_cost.plot(intervals, avg_costs, marker='o', label=f"Step={step}", color=color)
            ax_success.plot(intervals, success_rates, marker='s', linestyle='--', label=f"Step={step}", color=color)

        if files_found_count == 0:
            print(f"WARNING: No JSON files found for p_bias={p_bias}. Skipping plots.")
            plt.close(fig_cost)
            plt.close(fig_success)
            continue

        # 4. Save INSIDE the results folder
        ax_cost.set_title(f"Avg Cost vs. Iteration (p_bias={p_bias})")
        ax_cost.set_xlabel("Iteration")
        ax_cost.set_ylabel("Cost")
        ax_cost.legend(title="Step Size")
        ax_cost.grid(True, alpha=0.5)

        cost_path = os.path.join(results_dir, f"plot_cost_pbias_{p_bias}.png")
        fig_cost.savefig(cost_path)
        print(f"--> Saved plot: {cost_path}")

        ax_success.set_title(f"Success Rate vs. Iteration (p_bias={p_bias})")
        ax_success.set_xlabel("Iteration")
        ax_success.set_ylabel("Success (%)")
        ax_success.set_ylim(-5, 105)
        ax_success.legend(title="Step Size")
        ax_success.grid(True, alpha=0.5)

        success_path = os.path.join(results_dir, f"plot_success_pbias_{p_bias}.png")
        fig_success.savefig(success_path)
        print(f"--> Saved plot: {success_path}")

        plt.close(fig_cost)
        plt.close(fig_success)


def run_3d_N_times(N=10):
    # 1. Setup Environment (Same as run_3d)
    ur_params = UR5e_PARAMS(inflation_factor=1)
    env = Environment(env_idx=2)
    transform = Transform(ur_params)

    # Adjust resolution if needed (0.1 or 0.25)
    bb = BuildingBlocks3D(transform=transform,
                          ur_params=ur_params,
                          env=env,
                          resolution=0.25)

    # 2. Setup Start/Goal
    env2_start = np.deg2rad([110, -70, 90, -90, -90, 0])
    env2_goal = np.deg2rad([50, -80, 90, -90, -90, 0])

    # 3. Define Planner Parameters
    step_size = 0.2
    p_bias = 0.10
    max_itr = 4000

    print(f"Starting {N} iterations of 3D RRT*...")

    for i in range(N):
        print(f"\n--- Run {i + 1} / {N} ---")

        # Initialize Planner
        planner = RRTStarExperiment_path(
            max_step_size=step_size,
            start=env2_start,
            goal=env2_goal,
            max_itr=max_itr,
            stop_on_goal=False,  # Set to True if you want to stop at first solution
            bb=bb,
            goal_prob=p_bias,
            ext_mode="E2"
        )

        start_time = time.perf_counter()
        path = planner.plan()
        end_time = time.perf_counter()

        duration = end_time - start_time
        print(f"Execution time: {duration:.4f} seconds")

        if path is not None:
            # Calculate cost
            cost = planner.compute_cost(path)
            print(f"Path found! Cost: {cost:.4f}")

            # --- SAVING LOGIC ---

            # 1. Get Time String
            now = datetime.now()
            time_str = now.strftime("%Y-%m-%d_%H-%M-%S")

            # 2. Create Base "exps" folder if it doesn't exist
            exps_folder_name = os.path.join(os.getcwd(), "exps")
            if not os.path.exists(exps_folder_name):
                os.mkdir(exps_folder_name)

            # 3. Create Specific Experiment Folder
            # Naming format: exp_pbias_{}_step_{}_cost_{}_time_{}
            folder_name_str = (
                f"exp_pbias_{planner.goal_prob}_"
                f"step_{planner.max_step_size}_"
                f"cost_{cost:.2f}_"
                f"{time_str}"
            )

            exp_folder_path = os.path.join(exps_folder_name, folder_name_str)

            if not os.path.exists(exp_folder_path):
                os.mkdir(exp_folder_path)

            # 4. Save the path
            np.save(os.path.join(exp_folder_path, 'path.npy'), path)

            # 5. Save the stats
            with open(os.path.join(exp_folder_path, 'stats.txt'), "w") as file:
                file.write(f"Path cost: {cost} \n")
                file.write(f"Compute time: {duration} seconds \n")
                file.write(f"Iterations: {planner.max_itr} \n")
                file.write(f"Vertices in tree: {len(planner.tree.vertices)} \n")

            print(f"Saved results to: {folder_name_str}")

        else:
            print("No path found for this run.")


def run_my_2d_dot_algorithm():
    print("--- Starting 2D Dot Robot RRT* ---")

    # 1. Define Map Details
    # Make sure this JSON file exists in your 'twoD' folder
    json_file = "twoD/map2.json"

    # Coordinates are in pixels (X, Y)
    # Adjust these if your map requires different start/goals
    start_conf = np.array([360, 150])
    goal_conf = np.array([100, 200])

    # 2. Setup Environment
    planning_env = MapDotEnvironment(json_file=json_file)
    bb = DotBuildingBlocks2D(planning_env)

    # 3. Instantiate YOUR Planner
    # Note: max_step_size is in PIXELS now. 0.5 is too small! Use 15 or 20.
    planner = JRRTStarPlanner(
        bb=bb,
        start=start_conf,
        goal=goal_conf,
        ext_mode="E2",  # E1 usually means standard straight-line extension
        goal_prob=0.10,  # 10% chance to sample goal
        max_step_size=10.0,  # Step size in pixels
        max_itr=1000,  # Iterations
        stop_on_goal=True  # False = Keep optimizing (RRT* behavior)
    )

    # 4. Run Planning
    start_time = time.perf_counter()
    path = planner.plan()
    end_time = time.perf_counter()

    print(f"Execution time: {end_time - start_time:.4f} seconds")

    # 5. Visualization
    if path is not None:
        print(f"Path found! Cost: {planner.compute_cost(path):.2f}")

        # Get edges to visualize the tree structure (Debugging)
        # Your RRTTree likely has a method to get edges, e.g., get_edges_as_states()
        tree_edges = planner.tree.get_edges_as_states()

        # Visualize
        visualizer = DotVisualizer(bb)
        visualizer.visualize_map(
            plan=path,
            tree_edges=tree_edges,
            show_map=True,
            start=start_conf,
            goal=goal_conf
        )
    else:
        print("No path found.")





if __name__ == "__main__":
    start_time = time.perf_counter()
    # run_dot_2d_astar()
    # run_dot_2d_rrt()
    # run_dot_2d_rrt_star()
    # run_2d_rrt_motion_planning()
    # run_2d_rrt_inspection_planning()
    # run_2d_rrt_star_motion_planning()
    # run_profiled()
    # run_dot_2d_rrt()
    # for i in range(5):
    #     run_3d()
    # start_time = time.perf_counter()
    # run_3d_experiment()
    # path = np.load("new_path_yaacov.npy")
    run_2d_rrt_star_motion_planning()
    # run_2d_rrt_motion_planning()
    # plot_assignment_results()
    # run_3d_N_times(N=5)
    end_time = time.perf_counter()
    print(f"Took {end_time-start_time} seconds")
