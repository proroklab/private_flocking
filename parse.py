import os
import csv
import json
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_id")
parser.add_argument("--optimization_id")
parser.add_argument("--downsampling", type=int, default=1)
args = parser.parse_args()

path = os.path.join('../exp_logs', args.optimization_id, 'logs')
print("Scanning folder '{}'".format(path))
for x in sorted(os.walk(path)):
	if os.path.join(path, str(args.experiment_id))==x[0]:
		print("Now parsing experiment", x[0])
		with open(x[0]+"/log.csv") as csv_file:
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
						leader_ids[l] = int(row[4+l])
					initial_positions = np.zeros((n_drones,4))
					for d in range(n_drones):
						initial_positions[d,0] = float(row[4+n_leaders+4*d])
						initial_positions[d,1] = float(row[5+n_leaders+4*d])
						initial_positions[d,2] = float(row[6+n_leaders+4*d])
						initial_positions[d,3] = float(row[7+n_leaders+4*d])
		n_iterations = int(n_iterations)+1
		print("Found", n_rows, "entries in the .csv, corresponding to", n_iterations, "simulation steps with clock speed", clock_s , "for", n_drones, "drones (first timestep:", first_ts, ")")
		print("Initial positions:")
		print(initial_positions)
		print("Leaders:")
		print(leader_ids)

		os_timestemp = np.zeros((n_iterations))
		#
		sim_linear_acc = np.zeros((n_iterations, n_drones, 3))
		sim_linear_vel = np.zeros((n_iterations, n_drones, 3))
		sim_orient = np.zeros((n_iterations, n_drones, 4))
		sim_pos = np.zeros((n_iterations, n_drones, 3))
		#
		drones_timestamp = np.zeros((n_iterations, n_drones))
		#
		est_linear_acc = np.zeros((n_iterations, n_drones, 3))
		est_linear_vel = np.zeros((n_iterations, n_drones, 3))
		est_orient = np.zeros((n_iterations, n_drones, 4))
		est_pos = np.zeros((n_iterations, n_drones, 3))
		#
		ldr_traj_track_err = np.zeros((n_iterations))

		with open(x[0]+"/log.csv") as csv_file:
			csv_reader = csv.reader(csv_file, delimiter=',')
			index = 0
			for row in csv_reader:
				index +=1
				if index > 3:
					os_timestemp[int(row[0])] = int(row[1]) - first_ts
					for d in range(n_drones):
						drones_timestamp[int(row[0])][d] = float(row[15+d*27])/1e9 - first_ts
						for c in range(3):
							sim_linear_acc[int(row[0])][d][c] = float(row[2+c+d*27])
							sim_linear_vel[int(row[0])][d][c] = float(row[5+c+d*27])
							sim_orient[int(row[0])][d][c] = float(row[8+c+d*27])
							sim_pos[int(row[0])][d][c] = float(row[12+c+d*27])
							est_linear_acc[int(row[0])][d][c] = float(row[16+c+d*27])
							est_linear_vel[int(row[0])][d][c] = float(row[19+c+d*27])
							est_orient[int(row[0])][d][c] = float(row[22+c+d*27])
							est_pos[int(row[0])][d][c] = float(row[26+c+d*27])
						sim_orient[int(row[0])][d][3] = float(row[11+c+d*27])
						est_orient[int(row[0])][d][3] = float(row[25+c+d*27])
					ldr_traj_track_err[int(row[0])] = float(row[2+n_drones*27])

		# write to file for latex

		for d in range(n_drones):
			to_be_written = sim_pos[:,d,:]
			ts = np.reshape(drones_timestamp[:,d], (-1, 1))
			to_be_written = np.concatenate((to_be_written, ts), axis=-1)
			np.savetxt(x[0]+"/latex/pgfplotsdata/sim_pos_"+str(d)+".csv", to_be_written[::args.downsampling], delimiter=',') #downsample if needed
			if d in leader_ids:
				np.savetxt(x[0]+"/latex/pgfplotsdata/sim_pos_leader.csv", to_be_written[::args.downsampling], delimiter=',') #downsample if needed
			to_be_written = sim_linear_vel[:,d,:]
			to_be_written = np.concatenate((to_be_written, ts), axis=-1)
			np.savetxt(x[0]+"/latex/pgfplotsdata/sim_vel_"+str(d)+".csv", to_be_written[::args.downsampling], delimiter=',') #downsample if needed
			if d in leader_ids:
				np.savetxt(x[0]+"/latex/pgfplotsdata/sim_vel_leader.csv", to_be_written[::args.downsampling], delimiter=',') #downsample if needed

		# compute metrics

		# velocity correlation (alignment)
		sim_linear_vel_norm = np.linalg.norm(sim_linear_vel, axis=2)
		ali = 0
		EPSILON = 1e-3
		for i in range(n_drones):
			for j in range(n_drones):
				if j != i:
					d = np.einsum('ij,ij->i', sim_linear_vel[:,i,:], sim_linear_vel[:, j, :])
					ali += (d/(sim_linear_vel_norm[:,i] + EPSILON)/(sim_linear_vel_norm[:,j]+EPSILON))
		ali /= (n_drones*(n_drones-1))
		# flocking speed
		cof_v = np.mean(sim_linear_vel, axis=1)
		avg_sim_flock_linear_speed = np.linalg.norm(cof_v, axis=-1)
		# extension (cohesion)
		avg_sim_linear_vel = np.mean(sim_linear_vel, axis=1)
		avg_sim_pos = np.mean(sim_pos, axis=1)
		pos_diff = sim_pos - np.reshape(avg_sim_pos, (avg_sim_pos.shape[0], 1, -1))
		dis = np.linalg.norm(pos_diff, axis=-1)
		ext = np.mean(dis, axis=1)
		# position distance (cohesion)
		avg_sim_fllwr_pos = np.mean(np.delete(sim_pos, leader_ids.astype(int), 1), axis=1)
		# relative position of leader wrt followers (front/back)
		ldr_avg_fllwr_pos_diff = sim_pos[:, leader_ids.astype(int)] - np.reshape(avg_sim_fllwr_pos, (avg_sim_fllwr_pos.shape[0], 1, -1))
		pos_vel_dot = np.einsum('kij,kij->ki', ldr_avg_fllwr_pos_diff, sim_linear_vel[:, leader_ids.astype(int)])
		ldr_avg_fllwr_dis = np.linalg.norm(ldr_avg_fllwr_pos_diff, axis=-1)
		ldr_pos_wrt_fllwr = ldr_avg_fllwr_dis * np.sign(pos_vel_dot)
		# leader-nearest follower distance / followers' average spacing
		flock_spacing = []
		for i in range(n_drones):
			if i in leader_ids.astype(int):
				sim_fllwr_neighbor_pos = np.delete(sim_pos, leader_ids.astype(int), 1)
			else:
				sim_fllwr_neighbor_pos = np.delete(sim_pos, leader_ids.astype(int).tolist() + [i], 1)
			drone_neighbor_pos_diff = sim_fllwr_neighbor_pos - np.reshape(sim_pos[:,i,:], (sim_pos[:,i,:].shape[0], 1, -1))
			drone_neighbor_dis = np.linalg.norm(drone_neighbor_pos_diff, axis=-1)
			drone_spacing = np.amin(drone_neighbor_dis, axis=-1)
			flock_spacing.append(drone_spacing)
		flock_spacing = np.stack(flock_spacing, axis=-1)
		avg_fllwr_spacing = np.mean(np.delete(flock_spacing, leader_ids.astype(int), 1), axis=-1)
		ratio = flock_spacing[:, leader_ids.astype(int)]/np.reshape(avg_fllwr_spacing,(-1, 1))
		# spacing
		whole_flock_spacing = []
		for i in range(n_drones):
			sim_flck_neighbor_pos = np.delete(sim_pos, [i], 1)
			drone_neighbor_pos_diff = sim_flck_neighbor_pos - np.reshape(sim_pos[:,i,:], (sim_pos[:,i,:].shape[0], 1, -1))
			drone_neighbor_dis = np.linalg.norm(drone_neighbor_pos_diff, axis=-1)
			drone_spacing = np.amin(drone_neighbor_dis, axis=-1)
			whole_flock_spacing.append(drone_spacing)
		whole_flock_spacing = np.stack(whole_flock_spacing, axis=-1)
		avg_flock_spacing = np.mean(whole_flock_spacing, axis=-1)
		var_flock_spacing = np.var(whole_flock_spacing, axis=-1)
		# trajectory tracking error (leader to trajectory)
		traj_track_err = ldr_traj_track_err

		ts = np.reshape(drones_timestamp[:, 0], (-1, 1))
		ali_metric = np.concatenate((ts, np.reshape(ali, (-1, 1))), axis=-1)
		flock_speed_metric = np.concatenate((ts, np.reshape(avg_sim_flock_linear_speed, (-1, 1))), axis=-1)
		ext_metric = np.concatenate((ts, np.reshape(ext, (-1, 1))), axis=-1)
		ldr_pos_wrt_fllwr_metric = np.concatenate((ts, np.reshape(ldr_pos_wrt_fllwr, (-1, 1))), axis=-1)
		spacing_ratio_metric = np.concatenate((ts, np.reshape(ratio, (-1, 1))), axis=-1)
		traj_track_metric = np.concatenate((ts, np.reshape(traj_track_err, (-1, 1))), axis=-1)

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

		np.savetxt(x[0]+"/latex/pgfplotsdata/metric_1.csv", ali_metric[::args.downsampling], delimiter=',')  # downsample if needed
		np.savetxt(x[0]+"/latex/pgfplotsdata/metric_2.csv", flock_speed_metric[::args.downsampling], delimiter=',')  # downsample if needed
		np.savetxt(x[0]+"/latex/pgfplotsdata/metric_3.csv", ext_metric[::args.downsampling], delimiter=',')  # downsample if needed
		np.savetxt(x[0]+"/latex/pgfplotsdata/metric_4.csv", ldr_pos_wrt_fllwr_metric[::args.downsampling], delimiter=',')  # downsample if needed
		np.savetxt(x[0]+"/latex/pgfplotsdata/metric_5.csv", spacing_ratio_metric[::args.downsampling], delimiter=',')  # downsample if needed
		np.savetxt(x[0]+"/latex/pgfplotsdata/metric_6.csv", traj_track_metric[::args.downsampling], delimiter=',')  # downsample if needed
