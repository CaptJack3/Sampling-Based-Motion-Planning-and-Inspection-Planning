
import numpy as np
from environment import Environment
from kinematics import UR5e_PARAMS, Transform
from building_blocks import BuildingBlocks3D
import time
import matplotlib.pyplot as plt

def main():
    inflation_factors = np.linspace(1.0, 1.8, 9)
    # inflation_factors =[1.0,1.8]
    times = []
    is_collision_instances = []
    for inflation_factor in inflation_factors:
        ur_params = UR5e_PARAMS(inflation_factor=inflation_factor)
        env = Environment(env_idx=0)
        transform = Transform(ur_params)
        bb = BuildingBlocks3D(transform=transform, ur_params=ur_params, env=env, resolution=0.1)
        # change the path
        random_samples = np.load('random_samples_100k.npy')

         # TODO: HW2 5.2.5
        # For inflation factor 1.0 → compute ground truth
        if inflation_factor == 1.0:
            ground_truth = []
            start_t = time.time()
            for q in random_samples:
                ground_truth.append(bb.config_validity_checker(q))
            end_t = time.time()

            ground_truth = np.array(ground_truth, dtype=bool)
            times.append(end_t - start_t)
            is_collision_instances.append(0)
            continue

        # For other inflation factors → compute predicted collisions
        predicted = []
        start_t = time.time()
        for q in random_samples:
            predicted.append(bb.config_validity_checker(q))
        end_t = time.time()

        predicted = np.array(predicted, dtype=bool)

        # -----------------------------
        # CORRECT FALSE-POSITIVE CHECK:
        # -----------------------------
        # FP = predicted collision (False) AND ground truth free (True)
        false_pos = np.sum((ground_truth == True) & (predicted == False))

        times.append(end_t - start_t)
        is_collision_instances.append(false_pos)





    fig = plt.figure()
    ax1 = fig.add_subplot()
    ax1.set_xlabel('min radii factor')
    ax2 = ax1.twinx()
    ax1.set_ylabel('time (s)', color='blue')
    ax2.set_ylabel('False Negative Instances', color='red') 
    ax1.scatter(inflation_factors, times, c='blue')
    ax2.scatter(inflation_factors, is_collision_instances, c='red')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax2.tick_params(axis='y', labelcolor='red')
    fig.tight_layout()
    plt.show()




if __name__ == '__main__':
    main()



