import os
import airsim
import numpy as np
import argparse
import random
import time
#import config
import utils
import sys
sys.path.insert(0,'../controllers/')
from classes import Drone, Flock
import pygame
from pygame.locals import *
import math

parser = argparse.ArgumentParser('flocking')

# Flock
parser.add_argument('--flock_number', type=str, default='4', choices=['4', '5', '9', '12', '13', '16', '19', '21'])
parser.add_argument('--drivetrain_type', type=str, default='ForwardOnly', choices=['ForwardOnly', 'MaxDegreeOfFreedom'])
parser.add_argument('--base_dist', type=float, default=4)
parser.add_argument('--spread', type=float, default=1)
parser.add_argument('--frontness', type=float, default=0, help="relative position of leader wrt flock (-1 to +1)")
parser.add_argument('--sideness', type=float, default=0)

# Error and limitation settings
parser.add_argument('--add_observation_error', action='store_true', default=False)
parser.add_argument('--add_actuation_error', action='store_true', default=False)
parser.add_argument('--gaussian_error_mean', type=float, default=0)
parser.add_argument('--gaussian_error_std', type=float, default=0.1)
parser.add_argument('--add_v_threshold', action='store_true', default=False)
parser.add_argument('--v_thres', type=float, default=5)
parser.add_argument('--v_update_freq', type=float, default=50)

# Flocking methods
parser.add_argument('--flocking_method', type=str, default='reynolds', choices=['reynolds', 'sciencerobotics'])

# Reynolds flocking
# Leaders
parser.add_argument('--leader_id', type=int, default=-1)
parser.add_argument('--trajectory', type=str, default='Sinusoidal', choices=['Line', 'Zigzag', 'Sinusoidal', 'Circle', 'Square'])
parser.add_argument('--lookahead', type=float, default=0.5, help='lookahead x distance')
parser.add_argument('--line_len', type=float, default=10000, help='total length (x_direction) of the line, zigzag and sinusoidal trajectory')
parser.add_argument('--zigzag_len', type=float, default=60, help='period length of the zigzag trajectory')
parser.add_argument('--zigzag_width', type=float, default=5, help='one-sided width of the zigzag trajectory')
parser.add_argument('--sine_period_ratio', type=float, default=10, help='period ratio of the sine wave trajectory')
parser.add_argument('--sine_width', type=float, default=5, help='amplitude of the sine wave trajectory')
parser.add_argument('--side_len', type=float, default=30, help='length of the side of the square trajectory')
parser.add_argument('--radius', type=float, default=25, help='radius of the circular trajectory')
parser.add_argument('--v_leader', type=float, default=1)
parser.add_argument('--leader_sep_weight', type=float, default=0.3)
parser.add_argument('--leader_ali_weight', type=float, default=0.3)
parser.add_argument('--leader_coh_weight', type=float, default=0.3)
parser.add_argument('--leader_sep_max_cutoff', type=float, default=3)
parser.add_argument('--leader_ali_radius', type=float, default=200)
parser.add_argument('--leader_coh_radius', type=float, default=200)

# Followers
parser.add_argument('--pos2v_scale', type=float, default=0.5)
parser.add_argument('--sep_weight', type=float, default=1.25)
parser.add_argument('--ali_weight', type=float, default=1.0)
parser.add_argument('--coh_weight', type=float, default=0.75)
parser.add_argument('--sep_max_cutoff', type=float, default=3)
parser.add_argument('--ali_radius', type=float, default=200)
parser.add_argument('--coh_radius', type=float, default=200)

# Misc
parser.add_argument('--single_sim_duration', type=float, default=3.0, help='simulation duration for single experiment in minutes')
parser.add_argument('--log_step', type=int, default=25)
parser.add_argument('--log_level', type=str, default='INFO', choices=['INFO', 'DEBUG', 'WARN'])
parser.add_argument('--optim_path', type=str, default='')
parser.add_argument('--log_dir', type=str, default='logs')
parser.add_argument('--log_time', type=str, default='000')
parser.add_argument('--num_gpu', type=int, default=1)
parser.add_argument('--random_seed', type=int, default=12345)
parser.add_argument('--use_tensorboard', action='store_true', default=False)



def main():

    ti = time.time()

    args = parser.parse_args()
    if args.flocking_method == 'reynolds':
        from reynolds import Controller
    elif args.flocking_method == 'sciencerobotics':
        from sciencerobotics import Controller
    from leader_controller import Leader_Controller

    path = os.path.join(args.optim_path, args.log_dir)
    timestamp = utils.log_args(path, args)
    logger = utils.get_logger(path, timestamp)

    flock = Flock(args)

    utils.log_init_state(logger, flock)

    controller_list = []
    leader_controller_list = []

    for drone in flock.drones:
        if drone.vehicle_name in flock.leader_list:
            controller = Leader_Controller(drone, flock.flock_list, args)
            leader_controller_list.append(controller)
        else:
            controller = Controller(drone, flock.flock_list, args)
        controller_list.append(controller)

    #airsim.wait_key('Press any key to takeoff')
    print("Taking-off")
    flock.take_off()

    #airsim.wait_key('Press any key to go to different altitudes')
    print("Going to different altitudes")
    flock.initial_altitudes()

    #airsim.wait_key('Press any key to start initial motion')
    print("Starting random motion")
    flock.initial_speeds()

    #airsim.wait_key('Press any key to start flocking')
    print("Now flocking")
    count = 0

    first_drone_name = flock.drones[0].vehicle_name
    init_sim_time = flock.client.getMultirotorState(vehicle_name=first_drone_name).timestamp

    while True:
            for controller in controller_list:
                controller.step()
            if count % 1 == 0:
                flock.log_flock_kinematics(logger, count)

            count += 1
            pygame.display.set_mode((1,1))
            pygame.event.pump()
            keys = pygame.key.get_pressed()
            if keys[K_ESCAPE]:
                flock.reset()
                break
            curr_sim_time = flock.client.getMultirotorState(vehicle_name=first_drone_name).timestamp
            if (curr_sim_time-init_sim_time)/1e9/60 > args.single_sim_duration:
                tf = time.time()
                print("Real world time, ", (tf-ti)/60)
                flock.reset()
                break


if __name__ == '__main__':
    main()
