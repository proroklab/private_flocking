import os
import sys
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

metrics = np.array((11,1))
parameters = np.array((25,1))
labels = []
num_labels = []
pretrain_labels = []
pretrain_num_labels = []

for x in sorted(os.walk("../data_json/")):
	flag = True
	timestep = np.asarray(sorted(x[2]))
	timestep = [int(i[0:10]) for i in timestep]
	for y in sorted(x[2]):
		with open("../data_json/"+y, "r") as read_file:
			data = json.load(read_file)
			#
			temp = np.transpose(np.asarray(data["objs"]))
			temp = np.expand_dims(temp, axis=1)
			if flag == True:
				metrics = temp 
			else:
				metrics = np.hstack((metrics, temp)) 
			#
			temp = np.transpose(np.asarray(data["input"]))
			temp = np.expand_dims(temp, axis=1)
			if flag == True:
				parameters = temp
			else:
				parameters = np.hstack((parameters, temp))
			flag = False
			#
			labels.append(np.sum(np.asarray(data["discrim_predict"][0])))
			num_labels.append((np.asarray(data["discrim_predict"][0])).size)
			pretrain_labels.append(np.sum(np.asarray(data["discrim_predict"][2])))
			pretrain_num_labels.append((np.asarray(data["discrim_predict"][2])).size)
print(metrics.shape)
print(parameters.shape)
print("**************************************************")
print("**************************************************")
print("**************************************************")
labels = np.array(labels)
print(labels.shape)
print(labels)
num_labels = np.array(num_labels)
#print(num_labels.shape)
#print(num_labels)
print("**************************************************")
print("**************************************************")
print("**************************************************")
pretrain_labels = np.array(pretrain_labels)
print(pretrain_labels.shape)
print(pretrain_labels)
pretrain_num_labels = np.array(pretrain_num_labels)
#print(pretrain_num_labels.shape)
#print(pretrain_num_labels)
print("**************************************************")
print("**************************************************")
print("**************************************************")


generation_size=10
generation_counter = 0
gen_labels = 0
gen_correct_labels = 0
gen_good_flock_labels = 0
gen_good_flock_correct_labels = 0
for i in range(labels.shape[0]):
	if i%generation_size==0:
		if i==0:
			pass
		elif gen_labels==0:
			print("Should never happen")
		elif gen_good_flock_labels==0:
			print("No good lables in this generation", generation_counter)
			print("overall (bad) flock perf", gen_correct_labels/gen_labels, gen_correct_labels, "over", gen_correct_labels)
		else:
			print("gen", generation_counter)
			print("good flock perf", gen_good_flock_correct_labels/gen_good_flock_labels, gen_good_flock_correct_labels, "over", gen_good_flock_labels)
			print("overall perf", gen_correct_labels/gen_labels, gen_correct_labels, "over", gen_labels)
			print("bad flock perf", (gen_correct_labels-gen_good_flock_correct_labels)/(gen_labels-gen_good_flock_labels), gen_correct_labels-gen_good_flock_correct_labels, "over", gen_labels-gen_good_flock_labels)
		generation_counter += 1
		gen_labels = 0
		gen_correct_labels = 0
		gen_good_flock_labels = 0
		gen_good_flock_correct_labels = 0
	if metrics[9][i] < 1.0:
		gen_good_flock_labels += num_labels[i]
		gen_good_flock_correct_labels += labels[i]
	gen_labels += num_labels[i]
	gen_correct_labels += labels[i]
	#print("gen", generation_counter, "flock", metrics[9][i], "correct", labels[i], "out of", num_labels[i], "i.e.", labels[i]/num_labels[i])
print("**************************************************")
print("**************************************************")
print("**************************************************")


## Scatter Plots
###########################

#for j in range(int(metrics.shape[1]//10)):
for j in range(metrics.shape[1]):
	if j < 10:
		pass
	elif j < 100:
		if j%10!=0:
			continue
	else:
		if j%50!=0:
			continue
	# co-optimized privacy
	ax = plt.subplot(1, 2, 1)
	for k in range(j):
		plt.scatter(metrics[9,10*k:10*(k+1)], metrics[11,10*k:10*(k+1)])
	plt.xlim(0, 5)
	plt.ylim(0, 1)
	plt.xlabel('flock')
	plt.ylabel('privacy')
	# original privacy
	ax = plt.subplot(1, 2, 2)
	for k in range(j):
		plt.scatter(metrics[9,10*k:10*(k+1)], 2/(metrics[13,10*k:10*(k+1)]+1))
	plt.xlim(0, 5)
	plt.ylim(0, 1)
	plt.xlabel('flock')
	plt.ylabel('privacy')
	if j < 10:
		print(j)
		plt.show()
	elif j < 100:
			if j%10==0:
				print(j)
				plt.show()
	else:
		if j%50==0:
			print(j)
			plt.show()

#sys.exit()

## Parameters Distributions
###########################
for i in range(25):
	ax = plt.subplot(5, 5, i+1)
	ax.set_xlim([np.min(parameters[i,:]), np.max(parameters[i,:])])
	ax.set_ylim([0, metrics.shape[1]])
	if i == 0:
		y_label = "init spread"
	elif i == 1:
		y_label = "frontness"
	elif i == 2:
		y_label = "sideness"
	elif i == 3:
		y_label = "look ahead (fix)"
	elif i == 4:
		y_label = "zz T (fix)"
	elif i == 5:
		y_label = "zz A (fix)"
	elif i == 6:
		y_label = "sin T (fix)"
	elif i == 7:
		y_label = "sin A (fix)"
	elif i == 8:
		y_label = "lead v"
	elif i == 9:
		y_label = "l sep_w"
	elif i == 10:
		y_label = "l al_w"
	elif i == 11:
		y_label = "l coh_w"
	elif i == 12:
		y_label = "l sep c-o"
	elif i == 13:
		y_label = "l al_r"
	elif i == 14:
		y_label = "l coh_r"
	elif i == 15:
		y_label = "p2v (fix)"
	elif i == 16:
		y_label = "sep_w"
	elif i == 17:
		y_label = "al_w"
	elif i == 18:
		y_label = "coh_w"
	elif i == 19:
		y_label = "sep c-o"
	elif i == 20:
		y_label = "al_r"
	elif i == 21:
		y_label = "coh_r"
	elif i == 22:
		y_label = "num (fix)"
	elif i == 23:
		y_label = "drivetrain"
	elif i == 24:
		y_label = "traj (fix)"
	plt.ylabel(y_label)
	plt.hist(parameters[i,:], color = 'blue', edgecolor = 'black', bins = int(50))
plt.show()



## Parameters Evolution
#######################
for i in range(25):
	ax = plt.subplot(5, 5, i+1)
	if i == 0:
		y_label = "init spread"
	elif i == 1:
		y_label = "frontness"
	elif i == 2:
		y_label = "sideness"
	elif i == 3:
		y_label = "look ahead (fix)"
	elif i == 4:
		y_label = "zz T (fix)"
	elif i == 5:
		y_label = "zz A (fix)"
	elif i == 6:
		y_label = "sin T (fix)"
	elif i == 7:
		y_label = "sin A (fix)"
	elif i == 8:
		y_label = "lead v"
	elif i == 9:
		y_label = "l sep_w"
	elif i == 10:
		y_label = "l al_w"
	elif i == 11:
		y_label = "l coh_w"
	elif i == 12:
		y_label = "l sep c-o"
	elif i == 13:
		y_label = "l al_r"
	elif i == 14:
		y_label = "l coh_r"
	elif i == 15:
		y_label = "p2v (fix)"
	elif i == 16:
		y_label = "sep_w"
	elif i == 17:
		y_label = "al_w"
	elif i == 18:
		y_label = "coh_w"
	elif i == 19:
		y_label = "sep c-o"
	elif i == 20:
		y_label = "al_r"
	elif i == 21:
		y_label = "coh_r"
	elif i == 22:
		y_label = "num (fix)"
	elif i == 23:
		y_label = "drivetrain"
	elif i == 24:
		y_label = "traj (fix)"
	plt.ylabel(y_label)
	plt.plot(parameters[i,:])
plt.show()






## Metrics Distributions
#########################
for i in range(13):
	filtered = metrics[i,:]
	if i == 0:
		y_label = "align"
	elif i == 1:
		y_label = "var"
	elif i == 2:
		y_label = "speed"
		filtered = np.extract(filtered < 5, filtered)
	elif i == 3:
		y_label = "var"
		filtered = np.extract(filtered < 2, filtered)
	elif i == 4:
		y_label = "spacing"
		filtered = np.extract(filtered < 15, filtered)
	elif i == 5:
		y_label = "var"
		filtered = np.extract(filtered < 1, filtered)
	elif i == 6:
		y_label = "spacing var"
		filtered = np.extract(filtered < 20, filtered)
	elif i == 7:
		y_label = "track err"
		filtered = np.extract(filtered < 2, filtered)
	elif i == 8:
		y_label = "var"
		filtered = np.extract(filtered < 0.5, filtered)
	elif i == 9:
		y_label = "flock metric"
		filtered = np.extract(filtered < 100, filtered)
	elif i == 10:
		y_label = "pri. score"
		filtered = np.extract(filtered < 20, filtered)
	elif i == 11:
		y_label = "pri. obj"
		filtered = np.extract(filtered > -20, filtered)
	elif i == 12:
		y_label = "ga fitness"
		filtered = np.extract(filtered < 200, filtered)
	ax = plt.subplot(13, 1, i+1)
	ax.set_xlim([np.min(filtered), np.max(filtered)])
	plt.ylabel(y_label)
	plt.hist(filtered, color = 'blue', edgecolor = 'black', bins = int(50))
plt.show()



print(metrics[10,:])
print(metrics[13,:])

## Metrics Evolution
#####################
for i in range(20):
	n=0
	ax = plt.subplot(10, 2, i+1)
	if i == 0:
		y_label = "align"
		n=0
		ax.set_ylim([0, 1])
	elif i == 2:
		y_label = "var"
		n=1
		ax.set_ylim([0, 0.1])
	elif i == 4:
		y_label = "speed"
		n=2
		ax.set_ylim([0, 5])
	elif i == 6:
		y_label = "var"
		n=3
		ax.set_ylim([0, 10])
	elif i == 8:
		y_label = "spacing"
		n=4
		ax.set_ylim([0, 25])
	elif i == 10:
		y_label = "var"
		n=5
		ax.set_ylim([0, 50])
	elif i == 12:
		y_label = "spacing var"
		n=6
		ax.set_ylim([0, 150])
	elif i == 14:
		y_label = "track err"
		n=7
		ax.set_ylim([0, 5])
	elif i == 16:
		y_label = "var"
		n=8
		ax.set_ylim([0, 5])
	elif i == 1:
		y_label = "flock metric"
		n=9
		ax.set_ylim([0, 1500])
	elif i == 3:
		y_label = "best flock metric"
		ax.set_ylim([0, 100])
	elif i == 5:
		y_label = "pri. score"
		n=10
		#ax.set_ylim([0, 1000])
	elif i == 7:
		y_label = "best pri. score"
		#ax.set_ylim([-10, 100])
	elif i == 9:
		y_label = "pri. obj"
		n=11
		#ax.set_ylim([-1000, 0])
	elif i == 11:
		y_label = "best pri. obj"
		#ax.set_ylim([-1000, 0])
	elif i == 13:
		y_label = "ga fitness"
		n=12
		ax.set_ylim([-50, 100])
	elif i == 15:
		y_label = "best ga fitness"
		ax.set_ylim([-50, 100])
	elif i == 17:
		y_label = "correct labels"
	elif i == 18:
		y_label = "pt pri. score"
		n=13
	elif i == 19:
		y_label = "pt correct labels"
	plt.ylabel(y_label)
	#
	#
	if i == 3:
		min_metrics = np.zeros((1,metrics.shape[1]))
		for y in range(metrics.shape[1]):
			if y == 0:
				min_metrics[0,y] = metrics[9,y]
			else:
				#print flocking champions
				if metrics[9,y] < min_metrics[0,y-1]:
					print("flocking champion")
					print(y)
					print(timestep[y])
				min_metrics[0,y] = min(min_metrics[0,y-1],metrics[9,y])
		plt.plot(min_metrics[0,:])
	elif i == 7:
		max_metrics = np.zeros((1,metrics.shape[1]))
		for y in range(metrics.shape[1]):
			if y == 0:
				max_metrics[0,y] = metrics[10,y]
			else:
				max_metrics[0,y] = max(max_metrics[0,y-1],metrics[10,y])
		plt.plot(max_metrics[0,:])
	elif i == 11:
		min_metrics = np.zeros((1,metrics.shape[1]))
		for y in range(metrics.shape[1]):
			if y == 0:
				min_metrics[0,y] = metrics[11,y]
			else:
				min_metrics[0,y] = min(min_metrics[0,y-1],metrics[11,y])
		plt.plot(min_metrics[0,:])
	elif i == 15:
		min_metrics = np.zeros((1,metrics.shape[1]))
		for y in range(metrics.shape[1]):
			if y == 0:
				min_metrics[0,y] = metrics[12,y]
			else:
				#print fitness champions
				# if metrics[12,y] < min_metrics[0,y-1]:
				#	print("fitness champion")
				# 	print(y)
				# 	print(timestep[y])
				min_metrics[0,y] = min(min_metrics[0,y-1],metrics[12,y])
		plt.plot(min_metrics[0,:])
	elif i == 17:
		plt.plot(labels[:])
	elif i == 19:
		plt.plot(pretrain_labels[:])
	else: 
		plt.plot(metrics[n,:])
plt.show()






samples = int(np.max(labels)+1)

counters = np.zeros((samples))
for i in range(labels.shape[0]):
	counters[int(labels[i])] += 1
	if labels[i] > 0 and labels[i] < 5:
		print(i)
		print(labels[i])
		print(timestep[i])
		print()
print(counters)

plt.ylabel(y_label)
plt.hist(labels, color = 'blue', edgecolor = 'black', bins = int(samples))
plt.show()




















        # spread float 1-2
        # frontness float [-1, 1]
        # sideness float [-1 ,1]
        # zigzag_len float 4-30
        # ziazag_width float 0.2-2
        # sine_period_ratio 0.5-5
        # sine_width float 0.2-2
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




       # spread float 1-5
       # frontness float [-1, 1]
       # sideness float [-1 ,1]
       # lookahead float 0.3-1
       # zigzag_len float 4-60
       # ziazag_width float 0.2-2
       # sine_period_ratio 0.5-5
       # sine_width float 0.1-2
       # v_leader float 0.5-2
       # leader_sep_weight float 0-1
       # leader_ali_weight float 0-1
       # leader_coh_weight float 0-2
       # leader_sep_max_cutoff float 1-10
       # leader_ali_radius 10-200
       # leader_coh_radius 10-200
       # pos2v_scale float 0.2-2
       # sep_weight float 0.5-1.5
       # ali_weight float 0.5-1.5
       # coh_weight float 0.5-1.5
       # sep_max_cutoff float 1-10
       # ali_radius 10-200
       # coh_radius 10-200
       # flock_number int 4, 5, 6, 12, 13, 16
       # drivetrain_type index 0-ForwardOnly, 1-MaxDegreeOfFreedom
       # trajectory index 0-Line, 1-Zigzag, 2-Sinusoidal, 3-Circle, 4-Square

       # alignment     0-1
       # flock speed   0-inf
       # extension     0-inf
       # leader position wrt flock     -inf to +inf
       # leader-nn/average follower-nn     0-inf
       # leader traj tracking distance error   0-inf
       # extra time used due to flocking      0-inf
