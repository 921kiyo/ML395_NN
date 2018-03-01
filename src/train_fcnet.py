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
data = get_CIFAR10_data()
model = FullyConnectedNet(hidden_dims=[128], reg=1e-3, num_classes=10)

solver = Solver(model, data,
                update_rule='sgd',
                optim_config={'learning_rate': 1e-3,}, lr_decay = 0.85,
                num_epochs=100, batch_size=70,
                print_every=100)
solver.train()

solver._save_checkpoint() #save

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
