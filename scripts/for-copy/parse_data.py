import os
import sys
import json
import numpy as np
import scipy.io as sio
sys.path.insert(0,'../../../')
import utils
import math

def main():
    ts_len = 10
    old_data = []

    for x in sorted(os.walk("../logs/")):
        input = [0] * 25
        objs = [0] * 14
        predict = []
        if len(x[0]) == (len('../logs/') + ts_len):
            with open(x[0] + "/args.txt", 'r') as txt_file:
                lines = txt_file.readline().replace(')\n', '')
                list = [i for i in lines.split(',')]
                for item in list:
                    data = [i for i in item.split('=')]
                    if 'spread' in data[0]:
                        input[0] = float(data[1])
                    elif 'frontness' in data[0]:
                        input[1] = float(data[1])
                    elif 'sideness' in data[0]:
                        input[2] = float(data[1])
                    elif 'lookahead' in data[0]:
                        input[3] = float(data[1])
                    elif 'zigzag_len' in data[0]:
                        input[4] = float(data[1])
                    elif 'zigzag_width' in data[0]:
                        input[5] = float(data[1])
                    elif 'sine_period_ratio' in data[0]:
                        input[6] = float(data[1])
                    elif 'sine_width' in data[0]:
                        input[7] = float(data[1])
                    elif 'v_leader' in data[0]:
                        input[8] = float(data[1])
                    elif 'leader_sep_weight' in data[0]:
                        input[9] = float(data[1])
                    elif 'leader_ali_weight' in data[0]:
                        input[10] = float(data[1])
                    elif 'leader_coh_weight' in data[0]:
                        input[11] = float(data[1])
                    elif 'leader_sep_max_cutoff' in data[0]:
                        input[12] = float(data[1])
                    elif 'leader_ali_radius' in data[0]:
                        input[13] = float(data[1])
                    elif 'leader_coh_radius' in data[0]:
                        input[14] = float(data[1])
                    elif 'pos2v_scale' in data[0]:
                        input[15] = float(data[1])
                    elif 'sep_weight' in data[0]:
                        input[16] = float(data[1])
                    elif 'ali_weight' in data[0]:
                        input[17] = float(data[1])
                    elif 'coh_weight' in data[0]:
                        input[18] = float(data[1])
                    elif 'sep_max_cutoff' in data[0]:
                        input[19] = float(data[1])
                    elif 'ali_radius' in data[0]:
                        input[20] = float(data[1])
                    elif 'coh_radius' in data[0]:
                        input[21] = float(data[1])
                    elif 'flock_number' in data[0]:
                        input[22] = int(data[1].replace('\'', ''))
                    elif 'drivetrain_type' in data[0]:
                        if data[1].replace('\'', '')=='ForwardOnly':
                            input[23] = 0
                        else:
                            input[23] = 1
                    elif 'trajectory' in data[0]:
                        if data[1].replace('\'', '')=='Line':
                            input[24] = 0
                            traj = 'line/'
                        elif data[1].replace('\'', '')=='Zigzag':
                            input[24] = 1
                            traj = 'zigzag/'
                        elif data[1].replace('\'', '')=='Sinusoidal':
                            input[24] = 2
                            traj = 'sinusoidal/'
                        elif data[1].replace('\'', '')=='Circle':
                            input[24] = 3
                            traj = 'circle/'
                        elif data[1].replace('\'', '')=='Square':
                            input[24] = 4
                            traj = 'square/'

            if not os.path.exists(x[0] + "/metrics.json"):
                continue

            with open(x[0] + "/metrics.json") as json_file:
                metrics = json.load(json_file)
                objs[0] = float(metrics['ali_avg'])
                objs[1] = float(metrics['ali_var'])
                objs[2] = float(metrics['flock_speed_avg'])
                objs[3] = float(metrics['flock_speed_var'])
                objs[4] = float(metrics['avg_flock_spacing_avg'])
                objs[5] = float(metrics['avg_flock_spacing_var'])
                objs[6] = float(metrics['var_flock_spacing_avg'])
                objs[7] = float(metrics['ldr_track_err_avg'])
                objs[8] = float(metrics['ldr_track_err_var'])

            with open(x[0] + "/privacy.json") as json_file:
                privacy_data = json.load(json_file)
                objs[9] = float(privacy_data['optim_obj'])
                objs[10] = float(privacy_data['privacy_score'])
                objs[11] = float(privacy_data['privacy_obj'])
                objs[12] = float(privacy_data['ga_obj'])
                objs[13] = float(privacy_data['pretrain_privacy_score'])
                predict.append(privacy_data['predict_correctness'])
                predict.append(privacy_data['predict_confidence'])
                predict.append(privacy_data['pretrain_predict_correctness'])
                predict.append(privacy_data['pretrain_predict_confidence'])

            data_pair = {
                'input': input,
                'objs': objs,
                'discrim_predict': predict
            }

            if data_pair not in old_data:
                utils.makedirs('../data_json')
                with open('../data_json/'+ x[0].replace('../logs/', '') + '.json', 'w') as json_file:
                    json.dump(data_pair, json_file)
                old_data.append(data_pair)

if __name__ == '__main__':
    main()
