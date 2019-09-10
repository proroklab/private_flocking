import airsim
import numpy as np
import random
import utils
import sys
import json
import os
import warnings
import time
import math


def add_gaussian_noise(mean, std, v):
    error = np.random.normal(mean, std, v.size)
    return v+error


def limit_velocity(v, v_thres):
    v[v > v_thres] = v_thres
    # v[v < -v_thres] = -v_thres
    return v

def list2vec(pos):
    if len(pos) == 3:
        return airsim.Vector3r(pos[0], pos[1], pos[2])
    else:
        print("position list", pos, ": length invalid.")
        return pos

def conv_to_vec_path(list_path):
    vec_path = []
    for pos in list_path:
        vec_path.append(list2vec(pos))
    return vec_path

class Drone(object):
    def __init__(self, client, index, args):
        self.client = client
        self.index = index
        self.vehicle_name = "Drone"+str(index)
        self.client.enableApiControl(True, self.vehicle_name)
        self.client.armDisarm(True, self.vehicle_name)

        self.drivetrain_type = getattr(airsim.DrivetrainType, args.drivetrain_type)

        self.track_err = 0.0

        self.add_observation_error = args.add_observation_error
        self.add_actuation_error = args.add_actuation_error
        self.gaussian_error_mean = args.gaussian_error_mean
        self.gaussian_error_std = args.gaussian_error_std

        self.add_v_threshold = args.add_v_threshold
        self.v_thres = args.v_thres

        self.origin = np.array([0, 0, 0])
        self.init_yaw = 0

    def get_neighbors(self, flock_list):
        neighbors = []
        for drone_name in flock_list:
            if drone_name != self.vehicle_name:
                neighbors.append(drone_name)
        return neighbors

    def get_neighbors_dis(self, flock_list, neighbor_radius):
        neighbors = []
        for drone_name in flock_list:
            self_position = self.get_position()
            drone_position = self.get_neighbor_position(drone_name, flock_list[drone_name])
            distance = np.linalg.norm(self_position-drone_position)
            if distance <= neighbor_radius:
                neighbors.append(drone_name)
        return neighbors

    def get_position(self):
        r = self.client.getMultirotorState(vehicle_name=self.vehicle_name).kinematics_estimated.position
        pos = np.array([r.x_val, r.y_val, r.z_val])
        if self.add_observation_error:
            pos = add_gaussian_noise(self.gaussian_error_mean, self.gaussian_error_mean, pos)
        return pos + self.origin
        # return pos

    def get_neighbor_position(self, neighbor_drone_name, neighbor_origin):
        r = self.client.getMultirotorState(vehicle_name=neighbor_drone_name).kinematics_estimated.position
        pos = np.array([r.x_val, r.y_val, r.z_val])
        if self.add_observation_error:
            pos = add_gaussian_noise(self.gaussian_error_mean, self.gaussian_error_mean, pos)
        return pos + neighbor_origin
        # return pos

    def get_velocity(self):
        r = self.client.getMultirotorState(vehicle_name=self.vehicle_name).kinematics_estimated.linear_velocity
        v = np.array([r.x_val, r.y_val, r.z_val])
        if self.add_observation_error:
            v = add_gaussian_noise(self.gaussian_error_mean, self.gaussian_error_mean, v)
        return v

    def get_neighbor_velocity(self, neighbor_drone_name):
        r = self.client.getMultirotorState(vehicle_name=neighbor_drone_name).kinematics_estimated.linear_velocity
        v = np.array([r.x_val, r.y_val, r.z_val])
        if self.add_observation_error:
            v = add_gaussian_noise(self.gaussian_error_mean, self.gaussian_error_mean, v)
        return v

    def move_by_velocity(self, v, duration=1.0):
        if self.add_actuation_error:
            v = add_gaussian_noise(self.gaussian_error_mean, self.gaussian_error_std, v)
        if self.add_v_threshold:
            v = limit_velocity(v, self.v_thres)
        norm = np.linalg.norm(v)
        if norm <= 0.5:
            msg = self.vehicle_name + ", velocity magnitude smaller than 0.5, yaw control will not affect attitude."
            #warnings.warn(msg)
            #print(msg)
        return self.client.moveByVelocityAsync(v[0], v[1], v[2], duration, drivetrain=self.drivetrain_type,
                                               yaw_mode=airsim.YawMode(False, 0), vehicle_name=self.vehicle_name)


    def move_on_path(self, list_path, velocity):
        vec_path = conv_to_vec_path(list_path)
        return self.client.moveOnPathAsync(vec_path, velocity, drivetrain=self.drivetrain_type,
                                           yaw_mode=airsim.YawMode(False, 0), vehicle_name=self.vehicle_name)


class Flock(object):
    def __init__(self, args):
        self.flock_number = int(args.flock_number)
        self.base_dist = args.base_dist
        self.spread = args.spread
        self.frontness = args.frontness
        self.sideness = args.sideness

        # connect to the AirSim simulator
        client = airsim.MultirotorClient()
        client.confirmConnection()

        self.client = client

        self.flock_list = {}
        self.drones = []

        self.leader_list = []

        self.preset_leader_id = args.leader_id

        for i in range(self.flock_number):
            drone = Drone(client, i, args)
            self.drones.append(drone)

        self.clock_speed = 0.0
        self.load_initial_state()
        self.initial_setup()

        self.data_header_made = False
        self.log_data_list, self.axes, self.sorted_log_data_name_list = utils.get_log_data_list()

    def load_initial_state(self):
        path = '../'
        path = os.path.join(path, 'settings.json')
        if not os.path.exists(path):
            print('settings not available')
            sys.exit(1)
        with open(path) as json_file:
            data = json.load(json_file)
            self.clock_speed = data['ClockSpeed']
            vehicle_data = data['Vehicles']
            for drone in self.drones:
                pos = np.array(
                    [vehicle_data[drone.vehicle_name]['X'],
                     vehicle_data[drone.vehicle_name]['Y'],
                     vehicle_data[drone.vehicle_name]['Z']])
                drone.origin = pos
                drone.init_yaw = vehicle_data[drone.vehicle_name]['Yaw']
                self.flock_list[drone.vehicle_name] = pos

    def initialize_positions(self, num_drones, base_dist, spread ):
        interdrone_d = base_dist*spread
        sqrt2 = math.sqrt(2)
        positions = np.zeros([num_drones,2])
        if num_drones==4:
            positions[0,:] = [0.0,0.0]
            positions[1,:] = [sqrt2*interdrone_d,0.0]
            positions[2,:] = [0.5*(sqrt2*interdrone_d),0.5*(sqrt2*interdrone_d)]
            positions[3,:] = [0.5*(sqrt2*interdrone_d),-0.5*(sqrt2*interdrone_d)]
        elif num_drones==5:
            for i in range(3):
                positions[i,:] = [i*interdrone_d,0.0]
            positions[3,:] = [interdrone_d,interdrone_d]
            positions[4,:] = [interdrone_d,-interdrone_d]
        elif num_drones==9:
            for i in range(3):
                positions[i,:] = [i*(sqrt2*interdrone_d),0.0]
            for i in range(2):
                positions[3+i,:] = [(i+0.5)*(sqrt2*interdrone_d),0.5*(sqrt2*interdrone_d)]
                positions[5+i,:] = [(i+0.5)*(sqrt2*interdrone_d),-0.5*(sqrt2*interdrone_d)]
            positions[7,:] = [(sqrt2*interdrone_d),(sqrt2*interdrone_d)]
            positions[8,:] = [(sqrt2*interdrone_d),-(sqrt2*interdrone_d)]
        elif num_drones==12:
            for i in range(4):
                positions[i,:] = [i*interdrone_d,0.0]
                positions[4+i,:] = [i*interdrone_d,interdrone_d]
            for i in range(2):
                positions[8+i,:] = [(i+1)*interdrone_d,2*interdrone_d]
                positions[10+i,:] = [(i+1)*interdrone_d,-interdrone_d]
        elif num_drones==13:
            for i in range(5):
                positions[i,:] = [i*interdrone_d,0.0]
            for i in range(3):
                positions[5+i,:] = [(i+1)*interdrone_d,interdrone_d]
                positions[8+i,:] = [(i+1)*interdrone_d,-interdrone_d]
            positions[11,:] = [2*interdrone_d,2*interdrone_d]
            positions[12,:] = [2*interdrone_d,-2*interdrone_d]
        elif num_drones==16:
            for i in range(4):
                positions[i,:] = [i*(sqrt2*interdrone_d),0.0]
            for i in range(3):
                positions[4+i,:] = [(i+0.5)*(sqrt2*interdrone_d),0.5*(sqrt2*interdrone_d)]
                positions[7+i,:] = [(i+0.5)*(sqrt2*interdrone_d),-0.5*(sqrt2*interdrone_d)]
            for i in range(2):
                positions[10+i,:] = [(i+1)*(sqrt2*interdrone_d),(sqrt2*interdrone_d)]
                positions[12+i,:] = [(i+1)*(sqrt2*interdrone_d),-(sqrt2*interdrone_d)]
            positions[14,:] = [1.5*(sqrt2*interdrone_d),1.5*(sqrt2*interdrone_d)]
            positions[15,:] = [1.5*(sqrt2*interdrone_d),-1.5*(sqrt2*interdrone_d)]
        elif num_drones==19:
            positions[0,:] = [0.0,0.0]
            positions[1,:] = [-math.sqrt(3)/2*interdrone_d,0.5*interdrone_d]
            positions[2,:] = [-math.sqrt(3)*interdrone_d,interdrone_d]
            positions[3,:] = [math.sqrt(3)/2*interdrone_d,0.5*interdrone_d]
            positions[4,:] = [math.sqrt(3)*interdrone_d,interdrone_d]
            positions[5,:] = [-math.sqrt(3)/2*interdrone_d,-0.5*interdrone_d]
            positions[6,:] = [-math.sqrt(3)*interdrone_d,-interdrone_d]
            positions[7,:] = [math.sqrt(3)/2*interdrone_d,-0.5*interdrone_d]
            positions[8,:] = [math.sqrt(3)*interdrone_d,-interdrone_d]
            positions[9,:] = [0.0,interdrone_d]
            positions[10,:] = [0.0,2*interdrone_d]
            positions[11,:] = [0.0,-interdrone_d]
            positions[12,:] = [0.0,-2*interdrone_d]
            positions[13,:] = [-math.sqrt(3)*interdrone_d,0.0]
            positions[14,:] = [math.sqrt(3)*interdrone_d,0.0]
            positions[15,:] = [-math.sqrt(3)/2*interdrone_d,1.5*interdrone_d]
            positions[16,:] = [math.sqrt(3)/2*interdrone_d,1.5*interdrone_d]
            positions[17,:] = [-math.sqrt(3)/2*interdrone_d,-1.5*interdrone_d]
            positions[18,:] = [math.sqrt(3)/2*interdrone_d,-1.5*interdrone_d]
        elif num_drones==21:
            for i in range(5):
                positions[i,:] = [i*interdrone_d,0.0]
                positions[5+i,:] = [i*interdrone_d,interdrone_d]
                positions[10+i,:] = [i*interdrone_d,2*interdrone_d]
            for i in range(3):
                positions[15+i,:] = [(i+1)*interdrone_d,3*interdrone_d]
                positions[18+i,:] = [(i+1)*interdrone_d,-interdrone_d]
        else:
            print("Error: invalid number of drones for initialization.")
        return positions

    def initial_teleport(self, pos ):
        for drone in self.drones:
            pose = self.client.simGetVehiclePose(vehicle_name=drone.vehicle_name)
            pose.position.x_val += (pos[drone.index][0] - drone.origin[0])
            pose.position.y_val += (pos[drone.index][1] - drone.origin[1])
            self.client.simSetVehiclePose(pose, True, drone.vehicle_name)
        time.sleep(1)


    def get_leader_id(self, pos, front, side ):
        minx = miny = 1000.0
        maxx = maxy = -1000.0
        for i in range(pos.shape[0]):
            if pos[i,0] > maxx:
                maxx = pos[i,0]
            if pos[i,0] < minx:
                minx = pos[i,0]
            if pos[i,1] > maxy:
                maxy = pos[i,1]
            if pos[i,1] < miny:
                miny = pos[i,1]
        cx = minx+(maxx-minx)/2
        cy = miny+(maxy-miny)/2
        if (maxx-minx)<0 or (maxy-miny)<0:
            print("Warning in the get_leader_id function, unexpected formation size")
        r=max((maxx-minx),(maxy-miny))/2
        target = np.array([side,front])
        if np.linalg.norm(target) > 1.0:
            target = target/np.linalg.norm(target)
        target = r*target + [cx,cy]
        leader_id = 0
        min_distance = 1000.0
        for i in range(pos.shape[0]):
            if np.linalg.norm(pos[i,:]-target) < min_distance:
                min_distance = np.linalg.norm(pos[i,:]-target)
                leader_id = i
            print(np.linalg.norm(pos[i,:]-target))
        return leader_id

    def initial_setup(self):
        positions = self.initialize_positions(self.flock_number, self.base_dist, self.spread)
        self.initial_teleport(positions)
        leader_id = self.get_leader_id(positions, self.frontness, self.sideness)
        if self.preset_leader_id != -1:
            leader_id = self.preset_leader_id
        self.leader_list.append('Drone'+str(leader_id))

    def take_off(self):
        for vehicle_name in self.flock_list:
            f = self.client.takeoffAsync(vehicle_name=vehicle_name)
        f.join()

    def initial_altitudes(self):
        dur = 3.0
        for drone in self.drones:
            rand_vz = random.uniform(-4, -2)
            f = drone.move_by_velocity(np.array([0.0, 0.0, rand_vz]), duration=dur)
        f.join()
        for drone in self.drones:
            f = drone.move_by_velocity(np.array([0.0, 0.0, 0.0]))
        f.join()

    def initial_speeds(self):
        for drone in self.drones:
            v = np.random.rand(3) - np.array([0.5, 0.5, 0.5])
            f = drone.move_by_velocity(v, duration=3.0)
        f.join()

    def reset(self):
        for drone in self.drones:
            self.client.armDisarm(False, drone.vehicle_name)
        self.client.reset()
        for drone in self.drones:
            self.client.enableApiControl(False, drone.vehicle_name)

    def make_log_data_header(self, sorted_log_data_name_list, axes):
        msg = "iteration, os_timestamp"
        for drone in self.drones:
            est_ts_made = False
            for data_name in sorted_log_data_name_list:
                if 'est' in data_name and not est_ts_made:
                    msg += (", timestamp_"+str(drone.index))
                    est_ts_made = True
                for axis in axes:
                    if axis == 'w' and "orient" not in data_name:
                            continue
                    msg += ("," + drone.vehicle_name + "_" + data_name + "_" + axis)
        msg += ", Leader_track_error"
        return msg

    def log_flock_kinematics(self, logger, iteration):
        if not self.data_header_made:
            logger.info(self.make_log_data_header(self.sorted_log_data_name_list, self.axes))
            self.data_header_made = True

        os_timestamp = 0
        est_timestamp = {}
        sim_kinematics = {}
        est_kinematics = {}

        os_timestamp = utils.get_unix_timestamp()
        msg = '{},{}'.format(iteration, os_timestamp)

        # get simGroundTruthKinematics and multiRotorState.kinematics
        for drone in self.drones:
            sim_kinematics[drone.vehicle_name] = self.client.simGetGroundTruthKinematics(vehicle_name=drone.vehicle_name)
            multi_rotor_state = self.client.getMultirotorState(vehicle_name=drone.vehicle_name)
            est_timestamp[drone.vehicle_name] = multi_rotor_state.timestamp
            est_kinematics[drone.vehicle_name] = multi_rotor_state.kinematics_estimated

        # sort the data
        for drone in self.drones:
            est_ts_made = False
            for data_type in self.sorted_log_data_name_list:
                func = self.log_data_list[data_type]
                if 'sim' in data_type:
                    sim_data = getattr(sim_kinematics[drone.vehicle_name], func)
                    for axis in self.axes:
                        if axis == 'w' and "orient" not in data_type:
                            continue
                        msg += ','
                        data = getattr(sim_data, axis+'_val')
                        if 'pos' in data_type:
                            msg += str(data + drone.origin[self.axes.index(axis)-1])
                        else:
                            msg += str(data)
                else:
                    est_data = getattr(est_kinematics[drone.vehicle_name], func)
                    if not est_ts_made:
                        msg += ','
                        msg += str(est_timestamp[drone.vehicle_name])
                        est_ts_made = True
                    for axis in self.axes:
                        if axis == 'w' and "orient" not in data_type:
                            continue
                        msg += ','
                        msg += str(getattr(est_data, axis+'_val'))
        for drone_name in self.leader_list:
            drone_index = int(drone_name.replace("Drone", ""))
            msg += ','
            msg += str(self.drones[drone_index].track_err)
        logger.info(msg)


