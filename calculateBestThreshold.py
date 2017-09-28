import sys
import numpy as np

minDist = 0

rates_read = open("tpr_fpr_alignedLfw_list.txt", "r")

# tp, fp, tn, fn

points = []
for line in rates_read:
	lineSplit = line.split()
	points.append((float(lineSplit[0]), float(lineSplit[1]), float(lineSplit[2]), float(lineSplit[3])))

accuracies = []

for point in points:
	accuracies.append((point[0] + point[2])/(point[0] + point[1] + point[2] + point[3]))

# print(str(accuracies))
print("max accuracy: " + str(max(accuracies)))
print(str(np.argmax(accuracies)/400) + ", " + str(accuracies[np.argmax(accuracies)]))