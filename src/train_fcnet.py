import numpy as np

from src.fcnet import FullyConnectedNet
from src.utils.solver import Solver
from src.utils.data_utils import get_CIFAR10_data

"""
TODO: Use a Solver instance to train a TwoLayerNet that achieves at least 50% 
accuracy on the validation set.
"""
###########################################################################
#                           BEGIN OF YOUR CODE                            #
###########################################################################
#data = get_CIFAR10_data(num_training = 40000, num_validation= 2000, num_test= 2000)
data = get_CIFAR10_data()
model = FullyConnectedNet(hidden_dims=[80], reg=0, dropout=0) #weight_scale=1e-4,

solver = Solver(model, data,
                update_rule='sgd',
                optim_config={'learning_rate': 1e-3}, lr_decay= 0.85,
                num_epochs=50, batch_size=70,
                print_every=500)
solver.train()

solver._save_checkpoint() #save

# model = FullyConnectedNet(hidden_dims=([80]), reg=0.0, num_classes=10, dtype= np.float64, dropout=0.3, seed=0)
#
# solver = Solver(model, data,
#                 update_rule='sgd',
#                 optim_config={'learning_rate': 0.2*1e-3,},
#                 num_epochs=150, batch_size=32,
#                 print_every=100)


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