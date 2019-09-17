from __future__ import print_function
import os
import csv
import json
import shutil
import math
import numpy as np
import random
import logging
import time
from datetime import datetime
import torch
import torch.utils.data
import scipy.io as sio


class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def get_logger(log_dir, log_time, file_type='csv'):
    path = os.path.join(log_dir, log_time)
    makedirs(path)
    log_format = '%(message)s'
    # logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    #                     format=log_format, datefmt='%m/%d %I:%M:%S %p')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(path, 'log.' + file_type))
    fh.setFormatter(logging.Formatter(log_format))
    logger.addHandler(fh)
    return logger

def log_args(log_dir, args):
    timestamp = args.log_time
    if timestamp == "000":
        timestamp = str(get_unix_timestamp())
    path = os.path.join(log_dir, timestamp)
    makedirs(path)
    log_format = '%(message)s'
    # logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    #                     format=log_format, datefmt='%m/%d %I:%M:%S %p')
    args_logger = logging.getLogger()
    args_logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(path, 'args.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    args_logger.addHandler(fh)
    args_logger.info("args = %s", args)
    args_logger.removeHandler(fh)
    return timestamp

def create_exp_dir(path, scripts_to_save=None):
    makedirs(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'backup_code'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'backup_code', os.path.basename(script))
            shutil.copyfile(script, dst_file)

def get_time():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def get_unix_timestamp():
    return int(time.mktime(datetime.now().timetuple()))

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save(model, model_path):
    torch.save(model.state_dict(), model_path)

def load(model, model_path):
    model.load_state_dict(torch.load(model_path))

def remove_file(path):
    if os.path.exists(path):
        os.remove(path)

def backup_file(path):
    root, ext = os.path.splitext(path)
    new_path = "{}.backup_{}{}".format(root, get_time(), ext)
    os.rename(path, new_path)

def log_init_state(logger, flock):
    header = "init_timestamp, clock_speed, number_of_drones, number_of_leaders"
    for drone_name in flock.leader_list:
        header += ', {}'.format("leader_id_" + str(flock.leader_list.index(drone_name)))
    for drone in flock.drones:
        header += ',{},{},{},{}'.format(drone.vehicle_name+"_init_pos_x", drone.vehicle_name+"_init_pos_y", drone.vehicle_name+"_init_pos_z", drone.vehicle_name+"_init_yaw")
    logger.info(header)
    first_drone_name = flock.drones[0].vehicle_name
    first_ts = int(flock.client.getMultirotorState(vehicle_name=first_drone_name).timestamp/1e9)
    init_state = "{},{},{},{}".format(first_ts, flock.clock_speed, flock.flock_number, len(flock.leader_list))
    for i in range(len(flock.leader_list)):
        init_state += ("," + flock.leader_list[i].replace("Drone", ""))
    for drone in flock.drones:
        init_state += ("," + str(drone.origin[0]) + "," + str(drone.origin[1]) + "," + str(drone.origin[2]) + "," + str(drone.init_yaw))
    logger.info(init_state)

def get_log_data_list():
    log_data_list = {  # 'sim_angular_acc': 'angular_acceleration',
                'sim_linear_acc': 'linear_acceleration',
                # 'sim_angular_vel': 'angular_velocity',
                'sim_linear_vel': 'linear_velocity',
                'sim_orient': 'orientation',
                'sim_pos': 'position',
                # 'est_angular_acc': 'angular_acceleration',
                'est_linear_acc': 'linear_acceleration',
                # 'est_angular_vel': 'angular_velocity',
                'est_linear_vel': 'linear_velocity',
                'est_orient': 'orientation',
                'est_pos': 'position',
                }

    axes = ['w', 'x', 'y', 'z']

    sim_data_name_list = []
    est_data_name_list = []
    for key in sorted(log_data_list):
        if "sim" in key:
            sim_data_name_list.append(key)
        else:
            est_data_name_list.append(key)

    sorted_log_data_name_list = sim_data_name_list + est_data_name_list
    return log_data_list, axes, sorted_log_data_name_list

def get_data(data_path, observ_window, downsampling, multi_slice):
    list_of_train_set = []
    input_shape=None
    for x in sorted(os.listdir(data_path + '/train/')):
        if '.mat' in x:
            train_data_contents = sio.loadmat(data_path + '/train/' + x)
            train_timestamp = np.reshape(np.array(train_data_contents['timestamp']), (-1, 1))
            data_point = math.floor(train_timestamp[-1] / observ_window)
            input_all = train_data_contents['input']
            metrics_all = train_data_contents['metrics']
            lower_i = 0
            upper_i = np.argmax(train_timestamp > observ_window)

            if not multi_slice:
                data_point = 1
                ub = np.argmax(train_timestamp + observ_window > train_timestamp[-1])
                lower_i = random.randint(0, ub-1)
                upper_i = np.argmax(train_timestamp - train_timestamp[lower_i] > observ_window)

            for i in range(data_point):
                input_slice = input_all[lower_i:upper_i]
                ali_slice = np.array(metrics_all[0][lower_i:upper_i])
                ali_slice = ali_slice[np.logical_not(np.isnan(ali_slice))]
                speed_slice = np.array(metrics_all[1][lower_i:upper_i])
                spacing_slice = np.array(metrics_all[2][lower_i:upper_i])
                spacing_var_slice = np.array(metrics_all[3][lower_i:upper_i])
                track_err_slice = np.array(metrics_all[4][lower_i:upper_i])

                metrics = {
                    'ali_avg': np.mean(ali_slice),
                    'ali_var': np.var(ali_slice),
                    'flock_speed_avg': np.mean(speed_slice),
                    'flock_speed_var': np.var(speed_slice),
                    'avg_flock_spacing_avg': np.mean(spacing_slice),
                    "avg_flock_spacing_var": np.var(spacing_slice),
                    "var_flock_spacing_avg": np.mean(spacing_var_slice),
                    "ldr_track_err_avg": np.mean(track_err_slice),
                    "ldr_track_err_var": np.var(track_err_slice)
                }

                # average alignment (maximise)
                obj1 = -metrics['ali_avg']
                # variance of alignment (minimise)
                obj2 = metrics['ali_var']
                # average flock speed (close to certain value or range of values?)
                FLOCK_SPEED_MIN = 0.5
                FLOCK_SPEED_MAX = 2.0
                if FLOCK_SPEED_MIN < metrics['flock_speed_avg'] < FLOCK_SPEED_MAX:
                    obj3 = 0.0
                else:
                    obj3 = min(math.fabs(metrics['flock_speed_avg'] - FLOCK_SPEED_MIN),
                               math.fabs(metrics['flock_speed_avg'] - FLOCK_SPEED_MAX))
                # variance of flock speed (minimise)
                obj4 = metrics['flock_speed_var']
                # average (over time) of the average (over drones) flock spacing (close to certrain value)
                FLOCK_SPACING_MIN = 1.0
                FLOCK_SPACING_MAX = 3.0
                if FLOCK_SPACING_MIN < metrics['avg_flock_spacing_avg'] < FLOCK_SPACING_MAX:
                    obj5 = 0.0
                else:
                    obj5 = min(math.fabs(metrics['avg_flock_spacing_avg'] - FLOCK_SPACING_MIN),
                               math.fabs(metrics['avg_flock_spacing_avg'] - FLOCK_SPACING_MAX))
                # variance (over time) of the average (over drones) flock spacing (minimise)
                obj6 = metrics['avg_flock_spacing_var']
                # average (over time) of the variance (over drones) flock spacing (minimise)
                obj7 = metrics['var_flock_spacing_avg']
                # average leader traj tracking distance error (minimise)
                obj8 = metrics['ldr_track_err_avg']
                # variance of leader traj tracking distance error (minimise)
                obj9 = metrics['ldr_track_err_var']

                single_obj = obj1 + obj2 + obj3 + obj4 + obj5 + obj6 + obj7 + obj8 + obj9

                lower_i = upper_i
                if i < data_point - 1:
                    upper_i = np.argmax(train_timestamp > (i + 2) * observ_window)
                train_input = input_slice[::downsampling]
                if input_shape is None:
                    input_shape = train_input.shape
                else:
                    if input_shape[0] < train_input.shape[0]:
                        train_input = train_input[:input_shape[0]]
                    elif input_shape[0] > train_input.shape[0]:
                        repeat = train_input[-1].reshape(1, train_input.shape[1], -1)
                        for i in range(input_shape[0] - train_input.shape[0]):
                            train_input = np.concatenate((train_input, repeat), axis=0)

                offset = np.mean(train_input[0], axis=0)
                calib_pos = np.array(train_input - offset)
                train_input = np.reshape(calib_pos, (1, calib_pos.shape[0], calib_pos.shape[1], -1))
                train_input = torch.from_numpy(train_input).float()

                leader_id = int(train_data_contents['target'][0][0])
                train_target = torch.from_numpy(np.array([leader_id]))

                if single_obj > 1000:
                    continue

                list_of_train_set.append(torch.utils.data.TensorDataset(train_input, train_target))

    train_data = torch.utils.data.ConcatDataset(list_of_train_set)

    list_of_test_set = []

    for x in sorted(os.listdir(data_path + '/test/')):
        if '.mat' in x:
            test_data_contents = sio.loadmat(data_path + '/test/' + x)
            test_timestamp = np.reshape(np.array(test_data_contents['timestamp']), (-1, 1))
            data_point = math.floor(test_timestamp[-1] / observ_window)
            input_all = test_data_contents['input']
            metrics_all = test_data_contents['metrics']
            lower_i = 0
            upper_i = np.argmax(test_timestamp > observ_window)

            if not multi_slice:
                data_point = 1
                ub = np.argmax(test_timestamp + observ_window > test_timestamp[-1])
                lower_i = random.randint(0, ub-1)
                upper_i = np.argmax(test_timestamp - test_timestamp[lower_i] > observ_window)

            for i in range(data_point):
                input_slice = input_all[lower_i:upper_i]
                ali_slice = np.array(metrics_all[0][lower_i:upper_i])
                ali_slice = ali_slice[np.logical_not(np.isnan(ali_slice))]
                speed_slice = np.array(metrics_all[1][lower_i:upper_i])
                spacing_slice = np.array(metrics_all[2][lower_i:upper_i])
                spacing_var_slice = np.array(metrics_all[3][lower_i:upper_i])
                track_err_slice = np.array(metrics_all[4][lower_i:upper_i])

                metrics = {
                    'ali_avg': np.mean(ali_slice),
                    'ali_var': np.var(ali_slice),
                    'flock_speed_avg': np.mean(speed_slice),
                    'flock_speed_var': np.var(speed_slice),
                    'avg_flock_spacing_avg': np.mean(spacing_slice),
                    "avg_flock_spacing_var": np.var(spacing_slice),
                    "var_flock_spacing_avg": np.mean(spacing_var_slice),
                    "ldr_track_err_avg": np.mean(track_err_slice),
                    "ldr_track_err_var": np.var(track_err_slice)
                }

                # average alignment (maximise)
                obj1 = -metrics['ali_avg']
                # variance of alignment (minimise)
                obj2 = metrics['ali_var']
                # average flock speed (close to certain value or range of values?)
                FLOCK_SPEED_MIN = 0.5
                FLOCK_SPEED_MAX = 2.0
                if FLOCK_SPEED_MIN < metrics['flock_speed_avg'] < FLOCK_SPEED_MAX:
                    obj3 = 0.0
                else:
                    obj3 = min(math.fabs(metrics['flock_speed_avg'] - FLOCK_SPEED_MIN),
                               math.fabs(metrics['flock_speed_avg'] - FLOCK_SPEED_MAX))
                # variance of flock speed (minimise)
                obj4 = metrics['flock_speed_var']
                # average (over time) of the average (over drones) flock spacing (close to certest value)
                FLOCK_SPACING_MIN = 1.0
                FLOCK_SPACING_MAX = 3.0
                if FLOCK_SPACING_MIN < metrics['avg_flock_spacing_avg'] < FLOCK_SPACING_MAX:
                    obj5 = 0.0
                else:
                    obj5 = min(math.fabs(metrics['avg_flock_spacing_avg'] - FLOCK_SPACING_MIN),
                               math.fabs(metrics['avg_flock_spacing_avg'] - FLOCK_SPACING_MAX))
                # variance (over time) of the average (over drones) flock spacing (minimise)
                obj6 = metrics['avg_flock_spacing_var']
                # average (over time) of the variance (over drones) flock spacing (minimise)
                obj7 = metrics['var_flock_spacing_avg']
                # average leader traj tracking distance error (minimise)
                obj8 = metrics['ldr_track_err_avg']
                # variance of leader traj tracking distance error (minimise)
                obj9 = metrics['ldr_track_err_var']

                single_obj = obj1 + obj2 + obj3 + obj4 + obj5 + obj6 + obj7 + obj8 + obj9

                lower_i = upper_i
                if i < data_point - 1:
                    upper_i = np.argmax(test_timestamp > (i + 2) * observ_window)
                test_input = input_slice[::downsampling]
                if input_shape is None:
                    input_shape = test_input.shape
                else:
                    if input_shape[0] < test_input.shape[0]:
                        test_input = test_input[:input_shape[0]]
                    elif input_shape[0] > test_input.shape[0]:
                        repeat = test_input[-1].reshape(1, test_input.shape[1], -1)
                        for i in range(input_shape[0] - test_input.shape[0]):
                            test_input = np.concatenate((test_input, repeat), axis=0)

                offset = np.mean(test_input[0], axis=0)
                calib_pos = np.array(test_input - offset)
                test_input = np.reshape(calib_pos, (1, calib_pos.shape[0], calib_pos.shape[1], -1))
                test_input = torch.from_numpy(test_input).float()

                leader_id = int(test_data_contents['target'][0][0])
                test_target = torch.from_numpy(np.array([leader_id]))

                if single_obj > 1000:
                    continue

                list_of_test_set.append(torch.utils.data.TensorDataset(test_input, test_target))

    test_data = torch.utils.data.ConcatDataset(list_of_test_set)

    return train_data, test_data, input_shape

def get_metrics(optimization_id, experiment_id):
    path = os.path.join('../exp_logs', optimization_id, 'logs')
    print("Scanning folder '{}'".format(path))
    for x in sorted(os.walk(path)):
        if os.path.join(path, str(experiment_id)) == x[0]:
            print("Now parsing experiment", x[0])
            with open(x[0] + "/log.csv") as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                n_rows = -2  # account for the two headers
                for row in csv_reader:
                    n_rows += 1
                    n_iterations = row[0]
                    if n_rows == 0:
                        first_ts = int(row[0])
                        clock_s = float(row[1])
                        n_drones = int(row[2])
                        n_leaders = int(row[3])
                        leader_ids = np.zeros((n_leaders))
                        for l in range(n_leaders):
                            leader_ids[l] = int(row[4 + l])
                        initial_positions = np.zeros((n_drones, 4))
                        for d in range(n_drones):
                            initial_positions[d, 0] = float(row[4 + n_leaders + 4 * d])
                            initial_positions[d, 1] = float(row[5 + n_leaders + 4 * d])
                            initial_positions[d, 2] = float(row[6 + n_leaders + 4 * d])
                            initial_positions[d, 3] = float(row[7 + n_leaders + 4 * d])
            n_iterations = int(n_iterations) + 1
            print("Found", n_rows, "entries in the .csv, corresponding to", n_iterations,
                  "simulation steps with clock speed", clock_s, "for", n_drones, "drones (first timestep:", first_ts,
                  ")")
            print("Initial positions:")
            print(initial_positions)
            print("Leaders:")
            print(leader_ids)

            sim_linear_vel = np.zeros((n_iterations, n_drones, 3))
            sim_pos = np.zeros((n_iterations, n_drones, 3))
            #
            drones_timestamp = np.zeros((n_iterations, n_drones))
            #
            ldr_traj_track_err = np.zeros((n_iterations))

            with open(x[0] + "/log.csv") as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                index = 0
                for row in csv_reader:
                    index += 1
                    if index > 3:
                        for d in range(n_drones):
                            drones_timestamp[int(row[0])][d] = float(row[15 + d * 27]) / 1e9 - first_ts
                            for c in range(3):
                                sim_linear_vel[int(row[0])][d][c] = float(row[5 + c + d * 27])
                                sim_pos[int(row[0])][d][c] = float(row[12 + c + d * 27])
                        ldr_traj_track_err[int(row[0])] = float(row[2 + n_drones * 27])

            # compute metrics
            # velocity correlation (alignment)
            sim_linear_vel_norm = np.linalg.norm(sim_linear_vel, axis=2)
            ali = 0
            EPSILON = 1e-3
            for i in range(n_drones):
                for j in range(n_drones):
                    if j != i:
                        d = np.einsum('ij,ij->i', sim_linear_vel[:, i, :], sim_linear_vel[:, j, :])
                        ali += (d / (sim_linear_vel_norm[:, i] + EPSILON) / (sim_linear_vel_norm[:, j] + EPSILON))
            ali /= (n_drones * (n_drones - 1))

            # flocking speed
            cof_v = np.mean(sim_linear_vel, axis=1)
            avg_sim_flock_linear_speed = np.linalg.norm(cof_v, axis=-1)

            # spacing
            whole_flock_spacing = []
            for i in range(n_drones):
                sim_flck_neighbor_pos = np.delete(sim_pos, [i], 1)
                drone_neighbor_pos_diff = sim_flck_neighbor_pos - np.reshape(sim_pos[:, i, :],
                                                                             (sim_pos[:, i, :].shape[0], 1, -1))
                drone_neighbor_dis = np.linalg.norm(drone_neighbor_pos_diff, axis=-1)
                drone_spacing = np.amin(drone_neighbor_dis, axis=-1)
                whole_flock_spacing.append(drone_spacing)
            whole_flock_spacing = np.stack(whole_flock_spacing, axis=-1)
            avg_flock_spacing = np.mean(whole_flock_spacing, axis=-1)
            var_flock_spacing = np.var(whole_flock_spacing, axis=-1)

            # trajectory tracking error (leader to trajectory)
            traj_track_err = ldr_traj_track_err

            ts = np.reshape(drones_timestamp[:, 0], (-1, 1))

            cutoff_index = np.argmax(ts > 60)

            ali_filter = ali[cutoff_index:]
            ali_filter = ali_filter[np.logical_not(np.isnan(ali_filter))]

            perf_metrics = {
                "ali_avg": np.mean(ali_filter),
                "ali_var": np.var(ali_filter),
                "flock_speed_avg": np.mean(avg_sim_flock_linear_speed[cutoff_index:]),
                "flock_speed_var": np.var(avg_sim_flock_linear_speed[cutoff_index:]),
                "avg_flock_spacing_avg": np.mean(avg_flock_spacing[cutoff_index:]),
                "avg_flock_spacing_var": np.var(avg_flock_spacing[cutoff_index:]),
                "var_flock_spacing_avg": np.mean(var_flock_spacing[cutoff_index:]),
                "ldr_track_err_avg": np.mean(traj_track_err[cutoff_index:]),
                "ldr_track_err_var": np.var(traj_track_err[cutoff_index:])
            }

            with open(x[0] + '/metrics.json', 'w') as json_file:
                json.dump(perf_metrics, json_file)

            return perf_metrics

def parse_sim_data(optim_path, timestamp):

    with open(optim_path + '/logs/' + timestamp + "/log.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        n_rows = -2  # account for the two headers
        for row in csv_reader:
            n_rows += 1
            n_iterations = row[0]
            if n_rows == 0:
                first_ts = int(row[0])
                n_drones = int(row[2])
                n_leaders = int(row[3])
                leader_ids = np.zeros((n_leaders))
                for l in range(n_leaders):
                    leader_ids[l] = int(row[4 + l])
    n_iterations = int(n_iterations) + 1

    sim_pos = np.zeros((n_iterations, n_drones, 3))
    drones_timestamp = np.zeros((n_iterations, n_drones))

    with open(optim_path + '/logs/' + timestamp + "/log.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        index = 0
        for row in csv_reader:
            index += 1
            if index > 3:
                for d in range(n_drones):
                    drones_timestamp[int(row[0])][d] = float(row[15 + d * 27]) / 1e9 - first_ts
                    for c in range(3):
                        sim_pos[int(row[0])][d][c] = float(row[12 + c + d * 27])

    ts = np.reshape(drones_timestamp[:, 0], (-1, 1))

    cutoff_index = np.argmax(ts > 60)

    ts = ts[cutoff_index:]

    ts = (ts-ts[0]).tolist()

    input = sim_pos[cutoff_index:].tolist()

    target = int(leader_ids[0])

    data_pair = {
        'input': input,
        'target': target,
        'timestamp': ts
    }

    makedirs(os.path.join(optim_path, 'data'))
    sio.savemat(os.path.join(optim_path, 'data', timestamp + '.mat'), data_pair)

def load_sim_data(optim_path, ts, input_shape, stride=1, observ_window=5, downsampling=1, multi_slice=True):
    X = []
    y = []
    data_contents = sio.loadmat(os.path.join(optim_path, 'data', ts + '.mat'))
    timestamp = np.reshape(np.array(data_contents['timestamp']), (-1, 1))
    input_all = data_contents['input']

    ub = np.argmax(timestamp + observ_window > timestamp[-1])
    lower_i = 0
    if not multi_slice:
        lower_i = random.randint(0, ub - 1)

    while lower_i < ub:
        upper_i = np.argmax(timestamp - timestamp[lower_i] > observ_window)
        input_slice = input_all[lower_i:upper_i]
        lower_i = lower_i + int(stride * (upper_i - lower_i))
        if not multi_slice:
            lower_i = ub
        input = input_slice[::downsampling]
        if input_shape is None:
            input_shape = input.shape
        else:
            if input_shape[0] < input.shape[0]:
                input = input[:input_shape[0]]
            elif input_shape[0] > input.shape[0]:
                repeat = input[-1].reshape(1, input.shape[1], -1)
                for i in range(input_shape[0] - input.shape[0]):
                    input = np.concatenate((input, repeat), axis=0)

        offset = np.mean(input[0], axis=0)
        calib_pos = np.array(input - offset)
        input = np.reshape(calib_pos, (calib_pos.shape[0], calib_pos.shape[1], -1)).tolist()

        leader_id = int(data_contents['target'][0][0])
        X.append(input)
        y.append(leader_id)

    return X, y

def load_single_test_data(optim_path, ts, input_shape):
    X, y = load_sim_data(optim_path, ts, input_shape)
    arrayX = np.array(X)
    # print(arrayX.shape) # that is, for 15'' experiments: (2, 11, 9, 3)
    # arrayX[:,:,:,2].fill(0.0) # remove Z component (double check with Summer)
    return torch.from_numpy(arrayX).float(), torch.from_numpy(np.array(y)).long()

def load_disc_update_data(optim_path, ts_list, input_shape):
    X = []
    y = []
    for ts in ts_list:
        Xi, yi = load_sim_data(optim_path, ts, input_shape)
        X = X + Xi
        y = y + yi
    train_data = torch.utils.data.TensorDataset(torch.from_numpy(np.array(X)).float(),
                                                torch.from_numpy(np.array(y)).long())
    return train_data
