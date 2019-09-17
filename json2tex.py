import sys
import json
import numpy as np

metrics = np.array((13,1))
parameters = np.array((25,1))
labels = []

flag = True
with open("../exp_logs/"+sys.argv[1]+"/data_json/"+sys.argv[2]+".json", "r") as read_file:
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
labels = np.array(labels)
print(parameters.shape)
print(metrics.shape)
print(labels.shape)
print(parameters)
print(metrics)
print(labels)

f= open("../exp_logs/"+sys.argv[1]+"/logs/"+sys.argv[2]+"/latex/pgfplotsdata/metrics.tex","w+")
sys.stdout = f
print("{\scriptsize al.", "{:.2f}".format(float(metrics[0])), ",", "{:.2f}".format(float(metrics[1])), \
	" - sp.", "{:.2f}".format(float(metrics[2])), ",", "{:.2f}".format(float(metrics[3])), \
	" - spac.", "{:.2f}".format(float(metrics[4])), ",", "{:.2f}".format(float(metrics[5])), ",", "{:.2f}".format(float(metrics[6])), \
	" - track", "{:.2f}".format(float(metrics[7])), ",", "{:.2f}".format(float(metrics[8])), "}\\\[-0.5em]")
print("{\scriptsize flock", "{:.2f}".format(float(metrics[9])), \
	" - privacy", "{:.2f}".format(float(metrics[11])), ",", "{:.2f}".format(float(2/(metrics[13]+1))), ",", int(labels), \
	" - fitness", "{:.2f}".format(float(metrics[12])), \
	" - id", sys.argv[2], "}" )
f.close()
