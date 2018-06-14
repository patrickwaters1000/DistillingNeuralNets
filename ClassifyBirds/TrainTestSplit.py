import numpy as np
import os, re, sys

animals = open("animals","r").read().split("\n")[:5]
print(animals)
#  Load features
model_names=["mobilenet","inception"]
labels = []
features = {x:[] for x in model_names} 
for i,animal in enumerate(animals):
	folder = "Features/"+animal

	for model_name in model_names:
		new_features = np.load("{}/{}.npy".format(folder,model_name))
		features[model_name] += list(new_features)
	new_labels=[[(1 if i==j else 0) for j in range(len(animals))] for row in range(len(new_features))]
	labels += new_labels

#  Change lists to numpy arrays
labels = np.array(labels)
for model_name in model_names:
	features[model_name] = np.array(features[model_name])

#  Split data into train and test sets
n = labels.shape[0]
perm = np.random.permutation(n)
s1,s2 = 2*n//5, 4*n//5
index_sets= {"train1": perm[:s1], "train2":perm[s1:s2], "test":perm[s2:]}

if not os.path.exists("Data"):
	os.makedirs("Data")
for case in index_sets:
	indices = index_sets[case]
	y = labels[indices]
	np.save("Data/y_{}.npy".format(case),y)
	for model_name in model_names:
		x=features[model_name][indices]
		np.save("Data/x_{}_{}.npy".format(case,model_name),x)

