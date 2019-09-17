import random
import os


def main():
    for i in range(10000):

        pretrain_log_path = './'

        frontness = random.uniform(-1, 1)
        sideness = random.uniform(-1, 1)
        zigzag_len = 45 + random.uniform(-15, 15)
        zigzag_width = 5 + random.uniform(-2.5, 2.5)
        sine_period_ratio = 7.5 + random.uniform(-2.5, 2.5)
        sine_width = 5 + random.uniform(-2.5, 2.5)
        v_leader = 0.75 + random.uniform(-0.25, 0.25)
        leader_sep_weight = 0.3 + random.uniform(-0.1, 0.1)
        leader_ali_weight = 0.7 + random.uniform(-0.1, 0.1)
        leader_coh_weight = 0.9 + random.uniform(-0.3, 0.3)
        leader_sep_max_cutoff = 3 + random.uniform(-1, 1)
        leader_ali_radius = 10 + random.uniform(-3, 3)
        leader_coh_radius = 18 + random.uniform(-3, 3)
        sep_weight = 1.2 + random.uniform(-0.1, 0.1)
        ali_weight = 1.0  + random.uniform(-0.1, 0.1)
        coh_weight = 0.75 + random.uniform(-0.1, 0.1)
        sep_max_cutoff = 3+ random.uniform(-1, 1)
        ali_radius = 13 + random.uniform(-3, 3)
        coh_radius = 20 + random.uniform(-3, 3)

        #################################################
        ########### Mix traj or mono-traj ###############
        #################################################
        traj_int = random.randint(0, 2)
        traj_list = ['Line', 'Sinusoidal', 'Zigzag']
        #traj_list = ['Sinusoidal']
        traj = traj_list[traj_int]
        #################################################
        #################################################
        #################################################

        leader_id = random.randint(0, 8)

        os.system('python3.5 ../experiment.py --spread 1 --frontness {} --sideness {} \
                   --zigzag_len {} --zigzag_width {} --sine_period_ratio {} --sine_width {} --v_leader {} \
                   --leader_sep_weight {} --leader_ali_weight {} --leader_coh_weight {} --leader_sep_max_cutoff {} \
                   --leader_ali_radius {} --leader_coh_radius {} --sep_weight {} --ali_weight {} \
                   --coh_weight {} --sep_max_cutoff {} --ali_radius {} --coh_radius {} --flock_number 9 --drivetrain_type ForwardOnly \
                   --trajectory {} --leader_id {} --optim_path {}'.format(frontness, sideness, zigzag_len, zigzag_width,
                                                                          sine_period_ratio, sine_width,
                                                                          v_leader, leader_sep_weight,
                                                                          leader_ali_weight, leader_coh_weight,
                                                                          leader_sep_max_cutoff, leader_ali_radius,
                                                                          leader_coh_radius, sep_weight, ali_weight,
                                                                          coh_weight, sep_max_cutoff, ali_radius,
                                                                          coh_radius, traj, leader_id, pretrain_log_path))


if __name__ == '__main__':
    main()
