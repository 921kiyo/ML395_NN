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
model = FullyConnectedNet(hidden_dims=[128], reg=1e-4, num_classes=10, dtype = np.float64)

solver = Solver(model, data,
                update_rule='sgd',
                optim_config={'learning_rate': 1e-3,}, lr_decay = 0.85,
                num_epochs=30, batch_size=65,
                print_every=1000)
solver.train()

acc = solver.check_accuracy(data['X_train'], data['y_train'])
print("Training accuracy {} on {} examples".format(acc, len(data['X_train'])))
acc = solver.check_accuracy(data['X_val'], data['y_val'])
print("validation accuracy {} on {} examples".format(acc, len(data['X_val'])))
acc = solver.check_accuracy(data['X_test'], data['y_test'])
print("Test accuracy {} on {} examples".format(acc, len(data['X_test'])))

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
##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################
