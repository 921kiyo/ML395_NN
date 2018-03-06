# Function to load an test the model for Q5

# Loading
import pickle
import os
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt

def get_labels():
    path = "/vol/bitbucket/395ML_NN_Data"
    path_to_labels = os.path.join(path, 'datasets','FER2013','labels_public.txt')

    labs = open(path_to_labels).readlines()

    labels = {}
    for j in range(1,len(labs)):
        parts = labs[j].partition(',')
        name = parts[0]
        num = int(parts[2][0])
        labels[name] = num

    return labels


# vals = np.fromiter(labels.values(), dtype= np.int64)
#
# plt.hist(vals)
