import numpy as np
from bayes_opt import BayesianOptimization
from src.fcnet import FullyConnectedNet
from src.utils.solver import Solver
from src.utils.data_utils import get_FeR2013_data
import matplotlib.pyplot as plt
import json


"""
TODO: Use a Solver instance to train a TwoLayerNet that achieves at least 50%
accuracy on the validation set.
"""
###########################################################################
#                           BEGIN OF YOUR CODE                            #
###########################################################################
data = get_FeR2013_data()


def plot_bo(f, bo):
    xs = [x["x"] for x in bo.res["all"]["params"]]
    ys = bo.res["all"]["values"]

    mean, sigma = bo.gp.predict(np.arange(len(f)).reshape(-1, 1), return_std=True)

    plt.figure(figsize=(16, 9))
    plt.plot(f)
    plt.plot(np.arange(len(f)), mean)
    plt.fill_between(np.arange(len(f)), mean + sigma, mean - sigma, alpha=0.1)
    plt.scatter(bo.X.flatten(), bo.Y, c="red", s=50, zorder=10)
    plt.xlim(0, len(f))
    plt.ylim(f.min() - 0.1 * (f.max() - f.min()), f.max() + 0.1 * (f.max() - f.min()))
    plt.show()




#for lr in [0.0001,0.0005,0.001,0.005,0.01,0.05,0.1]:
def get_vals(lr):
    #128
    model = FullyConnectedNet(hidden_dims=[200], input_dim=48 * 48 * 1, reg=1e-3, num_classes=7, dtype=np.float64)
    solver = Solver(model, data,
                    update_rule='sgd',
                    optim_config={'learning_rate': lr,}, lr_decay = 0.85,
                    num_epochs=100, batch_size=70,
                    print_every=100)

    solver.train()
    solver._save_checkpoint() #save

    #val_acc = solver.best_val_acc
    #acc = max(solver.train_acc_history)
    #json_log = open("output_test.json", mode='wt', buffering=1)
    #json_log.write(json.dumps({'Learning Rate': lr, 'accuracy': acc, 'val_acc': val_acc, }) + '\n')
    print("Solver best val acc = " + str(solver.best_val_acc) )
    return solver.best_val_acc


bo = BayesianOptimization(lambda lr: get_vals(lr), {"lr":(1e-6,1e-3)})

bo.explore({"lr":(1e-6,1e-3)})
bo.maximize(init_points=2, n_iter=25, kappa=10,acq="ucb") #, acq="ucb"

'''

##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################
import matplotlib.pyplot as plt

plt.subplot(2, 1, 1)
plt.title("Training loss")
plt.plot(solver.loss_history, "o")
plt.xlabel('Iteration')
plt.subplot(2, 1, 2)
plt.title('Accuracy')
plt.plot(solver.train_acc_history, '-o', label='train')
plt.plot(solver.val_acc_history, '-o', label='val')
plt.plot([0.5] * len(solver.val_acc_history), 'k--')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.gcf().set_size_inches(15, 12)
plt.show()
'''