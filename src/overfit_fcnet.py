import numpy as np

from src.fcnet import FullyConnectedNet
from src.utils.solver import Solver
from src.utils.data_utils import get_CIFAR10_data

"""
TODO: Overfit the network with 50 samples of CIFAR-10
"""
###########################################################################
#                           BEGIN OF YOUR CODE                            #
###########################################################################
data = get_CIFAR10_data(num_training = 50)
model = FullyConnectedNet(hidden_dims=[128], reg=0.0, num_classes=10)

solver = Solver(model, data,
                update_rule='sgd',
                optim_config={'learning_rate': 1e-3,},
                num_epochs=20,
                batch_size=25,
                print_every=100)
solver.train()
##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################
