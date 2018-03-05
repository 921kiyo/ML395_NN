import numpy as np
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization

count = 0

def get_vals(lr):
    global count
    print(count)
    if count == 0:
        count += 1
        return 0.22
    elif count == 1:
        count += 1
        return 0.394
    elif count == 2:
        count += 1
        return 0.393
    count += 1
    return 0.4


bo = BayesianOptimization(lambda lr: get_vals(lr), {"lr":(1e-6,1e-3)})
bo.explore({"lr":(1e-6,1e-3)})
bo.maximize(init_points=2, n_iter=25, kappa=10,acq="ucb") #, acq="ucb"