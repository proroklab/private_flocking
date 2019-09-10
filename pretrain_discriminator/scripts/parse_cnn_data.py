import os
import csv
import numpy as np
import scipy.io as sio

def main():
    ts_len = 10

    for x in sorted(os.walk("../logs/")):
        if len(x[0]) == (len('../logs/') + ts_len):
            with open(x[0] + "/log.csv") as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                n_rows = -2  # account for the two headers
                for row in csv_reader:
                    n_rows += 1
                    n_iterations = row[0]
                    if n_rows == 0:
                        first_ts = int(row[0])
                        # clock_s = float(row[1])
                        n_drones = int(row[2])
                        n_leaders = int(row[3])
                        leader_ids = np.zeros((n_leaders))
                        for l in range(n_leaders):
                            leader_ids[l] = int(row[4 + l])
            n_iterations = int(n_iterations) + 1

            sim_linear_vel = np.zeros((n_iterations, n_drones, 3))
            sim_pos = np.zeros((n_iterations, n_drones, 3))
            drones_timestamp = np.zeros((n_iterations, n_drones))
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
            avg_sim_flock_linear_speed = np.mean(sim_linear_vel_norm, axis=1)

            # trajectory tracking error (flock centroid to trajectory)
            traj_track_err = ldr_traj_track_err

            ts = np.reshape(drones_timestamp[:, 0], (-1, 1))

            cutoff_index = np.argmax(ts > 60)

            ts = ts[cutoff_index:]

            ts = (ts-ts[0]).tolist()

            input = sim_pos[cutoff_index:].tolist()

            # ali_filter = ali[cutoff_index:]
            # ali_filter = ali_filter[np.logical_not(np.isnan(ali_filter))]

            metrics = [ali[cutoff_index:].tolist(),
                       avg_sim_flock_linear_speed[cutoff_index:].tolist(),
                       avg_flock_spacing[cutoff_index:].tolist(),
                       var_flock_spacing[cutoff_index:].tolist(),
                       traj_track_err[cutoff_index:].tolist()]

            target = int(leader_ids[0])

            data_pair = {
                'input': input,
                'target': target,
                'timestamp': ts,
                'metrics': metrics
            }

            if not os.path.exists('../data/'):
                os.makedirs('../data/')
            sio.savemat('../data/' + x[0].replace('../logs/', '') + '.mat', data_pair)


if __name__ == '__main__':
    main()
