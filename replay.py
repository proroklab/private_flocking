import os
import sys
import time
import argparse
from datetime import datetime

parser = argparse.ArgumentParser('replay')
parser.add_argument('--log_dir', type=str, default='../exp_logs')
parser.add_argument('--optim_id', type=str, default='000')
parser.add_argument('--exp_name', type=str, default='000')
parser.add_argument('--single_sim_duration', type=float, default=10.0)

def main():
    args = parser.parse_args()

    if args.exp_name == '000' or args.optim_id == '000':
        print('No experiment name specified')
        sys.exit(1)

    x = [0] * 26

    with open(os.path.join(args.log_dir, args.optim_id, 'logs', args.exp_name, 'args.txt'), 'r') as txt_file:
        lines = txt_file.readline().replace(')\n', '')
        list = [i for i in lines.split(',')]
        for item in list:
            data = [i for i in item.split('=')]
            if 'spread' in data[0]:
                x[0] = float(data[1])
            elif 'frontness' in data[0]:
                x[1] = float(data[1])
            elif 'sideness' in data[0]:
                x[2] = float(data[1])
            elif 'lookahead' in data[0]:
                x[3] = float(data[1])
            elif 'zigzag_len' in data[0]:
                x[4] = float(data[1])
            elif 'zigzag_width' in data[0]:
                x[5] = float(data[1])
            elif 'sine_period_ratio' in data[0]:
                x[6] = float(data[1])
            elif 'sine_width' in data[0]:
                x[7] = float(data[1])
            elif 'v_leader' in data[0]:
                x[8] = float(data[1])
            elif 'leader_sep_weight' in data[0]:
                x[9] = float(data[1])
            elif 'leader_ali_weight' in data[0]:
                x[10] = float(data[1])
            elif 'leader_coh_weight' in data[0]:
                x[11] = float(data[1])
            elif 'leader_sep_max_cutoff' in data[0]:
                x[12] = float(data[1])
            elif 'leader_ali_radius' in data[0]:
                x[13] = float(data[1])
            elif 'leader_coh_radius' in data[0]:
                x[14] = float(data[1])
            elif 'pos2v_scale' in data[0]:
                x[15] = float(data[1])
            elif 'sep_weight' in data[0]:
                x[16] = float(data[1])
            elif 'ali_weight' in data[0]:
                x[17] = float(data[1])
            elif 'coh_weight' in data[0]:
                x[18] = float(data[1])
            elif 'sep_max_cutoff' in data[0]:
                x[19] = float(data[1])
            elif 'ali_radius' in data[0]:
                x[20] = float(data[1])
            elif 'coh_radius' in data[0]:
                x[21] = float(data[1])
            elif 'flock_number' in data[0]:
                x[22] = int(data[1].replace('\'', ''))
            elif 'drivetrain_type' in data[0]:
                if data[1].replace('\'', '')=='ForwardOnly':
                    x[23] = 0
                else:
                    x[23] = 1
            elif 'trajectory' in data[0]:
                if data[1].replace('\'', '')=='Line':
                    x[24] = 0
                    traj = 'line/'
                elif data[1].replace('\'', '')=='Zigzag':
                    x[24] = 1
                    traj = 'zigzag/'
                elif data[1].replace('\'', '')=='Sinusoidal':
                    x[24] = 2
                    traj = 'sinusoidal/'
                elif data[1].replace('\'', '')=='Circle':
                    x[24] = 3
                    traj = 'circle/'
                elif data[1].replace('\'', '')=='Square':
                    x[24] = 4
                    traj = 'square/'
            elif 'leader_id' in data[0]:
                x[25] = int(data[1])

    drivetrain_list = ['ForwardOnly', 'MaxDegreeOfFreedom']
    traj_list = ['Line', 'Zigzag', 'Sinusoidal', 'Circle', 'Square']
    drivetrain = drivetrain_list[int(x[23])]

    traj = traj_list[int(x[24])]

    timestamp = str(int(time.mktime(datetime.now().timetuple())))
    os.system('mkdir ../exp_logs/replay')

    os.system('python3.5 ../experiment.py --spread {} --frontness {} --sideness {} --lookahead {} \
    --zigzag_len {} --zigzag_width {} --sine_period_ratio {} --sine_width {} --v_leader {} \
    --leader_sep_weight {} --leader_ali_weight {} --leader_coh_weight {} --leader_sep_max_cutoff {} \
    --leader_ali_radius {} --leader_coh_radius {} --pos2v_scale {} --sep_weight {} --ali_weight {} \
    --coh_weight {} --sep_max_cutoff {} --ali_radius {} --coh_radius {} --flock_number {} --drivetrain_type {} \
    --trajectory {} --leader_id {} --log_time {} --optim_path ../exp_logs/replay --single_sim_duration {}'.format(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12],
                             x[13], x[14], x[15], x[16], x[17], x[18], x[19], x[20], x[21],
                             x[22], drivetrain, traj, x[25], timestamp, args.single_sim_duration))
    

if __name__ == '__main__':
    main()