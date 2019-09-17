import os
import time
from datetime import datetime
import json
import math
import utils
from config import args

TS_LIST = []

class Flocking:

    def __init__(self, path, ts, model):
        self.flock_number = args.num_drones
        self.trajectory = args.trajectory
        self.model = model
        self.optim_path = path
        self.optim_id = ts
        self.count = 0
        self.best = None
        self.ts = ts

    def fitness(self, x):
        # simulation
        drivetrain_list = ['ForwardOnly', 'MaxDegreeOfFreedom']
        drivetrain = drivetrain_list[int(x[16])]

        timestamp = str(int(time.mktime(datetime.now().timetuple())))

        os.system('python3.5 ../experiment.py --spread {} --frontness {} --sideness {} \
        --zigzag_len 60 --zigzag_width 5 --sine_period_ratio 10 --sine_width 5 --v_leader {} \
        --leader_sep_weight {} --leader_ali_weight {} --leader_coh_weight {} --leader_sep_max_cutoff {} \
        --leader_ali_radius {} --leader_coh_radius {} --sep_weight {} --ali_weight {} \
        --coh_weight {} --sep_max_cutoff {} --ali_radius {} --coh_radius {} --flock_number {} --drivetrain_type {} \
        --trajectory {} --log_time {} --optim_path {}'.format(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12],
                                 x[13], x[14], x[15],
                                 self.flock_number, drivetrain, self.trajectory, timestamp, self.optim_path))

        metrics = utils.get_metrics(self.optim_id, timestamp)

        # average alignment (maximise)
        obj1 = -metrics['ali_avg']
        # variance of alignment (minimise)
        obj2 = metrics['ali_var']*10
        # average flock speed (close to certain value or range of values?)
        FLOCK_SPEED_MIN = 2.0
        if FLOCK_SPEED_MIN < metrics['flock_speed_avg']:
            obj3 = 0.0
        elif metrics['flock_speed_avg'] < 0.1: 
            obj3 = 4.0
        else:
            obj3 =  math.fabs(metrics['flock_speed_avg']-FLOCK_SPEED_MIN)
        # variance of flock speed (minimise)
        obj4 = metrics['flock_speed_var']
        # average (over time) of the average (over drones) flock spacing (close to certrain value)
        FLOCK_SPACING_MIN = 1.0
        FLOCK_SPACING_MAX = 3.0
        if FLOCK_SPACING_MIN < metrics['avg_flock_spacing_avg'] < FLOCK_SPACING_MAX:
            obj5 = 0.0
        else:
            obj5 = min(math.fabs(metrics['avg_flock_spacing_avg']-FLOCK_SPACING_MIN), math.fabs(metrics['avg_flock_spacing_avg']-FLOCK_SPACING_MAX))
        # variance (over time) of the average (over drones) flock spacing (minimise)
        obj6 = metrics['avg_flock_spacing_var']
        # average (over time) of the variance (over drones) flock spacing (minimise)
        obj7 = metrics['var_flock_spacing_avg']*0.001
        # average leader traj tracking distance error (minimise)
        obj8 = metrics['ldr_track_err_avg']*2
        # variance of leader traj tracking distance error (minimise)
        obj9 = metrics['ldr_track_err_var']*10

        # flocking metric
        optim_obj = obj1 + obj2 + obj3 + obj4 + obj5 + obj6 + obj7 + obj8 + obj9
        optim_obj = optim_obj/2 # make "good flocking to be within performance of 1"
        
        # online training examples
        if optim_obj < 500: #1000
            TS_LIST.append(timestamp) # only "good" flocking
        #TS_LIST.append(timestamp) # any generated solution (should converge to good flocking)

        # online model to predict the leader of the current solution
        utils.parse_sim_data(self.optim_path, timestamp)
        utils.load(self.model, os.path.join(self.optim_path, 'weights.pt'))
        privacy_score, predict_correctness, predict_confidence = self.model.test_loss(timestamp)

        # VERIFY HOW WE ARE PERFORMING WRT THE ORIGINAL DISCRIMINATOR
        model_copy = self.model
        utils.load(model_copy, '../pretrain-weights.pt')
        pretrain_privacy_score, pretrain_predict_correctness, pretrain_predict_confidence = model_copy.test_loss(timestamp)

        # privacy metric
        privacy_obj = 1/(privacy_score + 1) # 2/(privacy_score + 1)

        # combining flocking and privacy in a single fitness value
        if optim_obj > 1.0:
            single_obj = optim_obj
        else: 
            #alpha = 1.0 #pure flocking
            alpha = 0.5 #private flocking
            single_obj = alpha * optim_obj + (1.0-alpha) * privacy_obj

        # JSON Logging
        privacy_data = {
            'optim_obj': optim_obj,
            'privacy_score': privacy_score,
            'privacy_obj': privacy_obj,
            'ga_obj': single_obj,
            'predict_correctness': predict_correctness,
            'predict_confidence': predict_confidence,
            'pretrain_privacy_score': pretrain_privacy_score,
            'pretrain_predict_correctness': pretrain_predict_correctness,
            'pretrain_predict_confidence': pretrain_predict_confidence
        }
        with open(os.path.join(self.optim_path, 'logs', timestamp, "privacy.json"), 'w') as json_file:
            json.dump(privacy_data, json_file)

        self.count += 1
        print(self.count)
        return [single_obj]

    def get_nobj(self):
        return 1

    def get_nic(self):
        return 0

    def get_nix(self):
        return 1

    def get_name(self):
        return "Flocking problem."

    def get_bounds(self):
        # spread float 1-2
        # frontness float [-1, 1]
        # sideness float [-1 ,1]
            # zigzag_len float 4-30
            # ziazag_width float 0.2-2
            # sine_period_ratio 0.5-5
            # sine_width float 0.2s-2
        # v_leader float 0.5-1.5
        # leader_sep_weight float 0-1
        # leader_ali_weight float 0-1
        # leader_coh_weight float 0-2
        # leader_sep_max_cutoff float 1-5
        # leader_ali_radius 1-20
        # leader_coh_radius 1-20
        # sep_weight float 0.5-1.5
        # ali_weight float 0.5-1.5
        # coh_weight float 0.5-1.5
        # sep_max_cutoff float 1-5
        # ali_radius 1-20
        # coh_radius 1-20

        # drivetrain_type index 0-ForwardOnly, 1-MaxDegreeOfFreedom

        # with zz and sin par
        # lb = [1, -1, -1, 4, 0.2, 0.5, 0.2, 0.5, 0, 0, 0, 1, 1, 1, 0.5, 0.5, 0.5, 1, 1, 1, 0]
        # ub = [2, 1, 1, 30, 2, 5, 2, 1.5, 1, 1, 2, 5, 20, 20, 1.5, 1.5, 1.5, 5, 20, 20, 1]

        # without zz and sin par
        lb = [1, -1, -1, 0.5, 0, 0, 0, 1, 1, 1, 0.5, 0.5, 0.5, 1, 1, 1, 0]
        ub = [2, 1, 1, 1.5, 1, 1, 2, 5, 20, 20, 1.5, 1.5, 1.5, 5, 20, 20, 1]
        return lb, ub