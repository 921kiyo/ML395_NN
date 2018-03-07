import numpy as np
#from bayes_opt import BayesianOptimization
from src.fcnet import FullyConnectedNet
from src.utils.solver import Solver
from src.utils.data_utils import get_FeR2013_data
import matplotlib.pyplot as plt
import json

kwargs = None
json_log = None
json_log2 = None

"""
TODO: Use a Solver instance to train a TwoLayerNet that achieves at least 50%
accuracy on the validation set.
"""
###########################################################################
#                           BEGIN OF YOUR CODE                            #
###########################################################################
data = get_FeR2013_data()

#CREATE A MODEL WITH THE HYPERPARAMETERS AND RETURN THE BEST_VAL_ACC
def get_vals(lr,hidden_dims1,hidden_dims2,lr_decay,reg,drop):
    global json_log2,json_log
    lr = 10 ** lr
    model = FullyConnectedNet(hidden_dims=[int(hidden_dims1),int(hidden_dims2)], input_dim=48 * 48 * 1, reg=reg, num_classes=7, dtype=np.float64,dropout=drop)
    solver = Solver(model, data,
                    update_rule='sgd_momentum',
                    optim_config={'learning_rate': lr,}, lr_decay = lr_decay,
                    num_epochs=50, batch_size=70,
                    print_every=1000000)

    solver.train()
    solver._save_checkpoint()

    #SAVE THE VALUES TO A FILE
    val_acc = solver.best_val_acc
    acc = max(solver.train_acc_history)
    loss = min(solver.loss_history)
    json_log2.write(json.dumps({'Learning Rate': lr,
                               'accuracy': acc, 'val_acc': val_acc ,
                                'loss': loss,"lr_decay":lr_decay,
                                'dropout':drop,'reg':reg,
                                'layer_1': hidden_dims1,'layer_2': hidden_dims2}) + ',\n')
    
    json_log.write(json.dumps({'Learning Rate': lr,
                               'accuracy': solver.train_acc_history,
                                'val_acc': solver.val_acc_history,
                                'loss': solver.loss_history,"lr_decay":lr_decay,
                                'dropout':drop,'reg':reg,
                               'layer_1': hidden_dims1,'layer_2': hidden_dims2}) + ',\n')
    return solver.best_val_acc

#RETURNS A GRID BETWEEN TWO VALUES
def return_grid(value,dist):
    grid = np.arange(value[0],value[1]+dist/5,dist/5,dtype=np.float64)
    return grid

def minimise_lr_and_lr_decay(old_val,**args):
    global kwargs
    if args is not None:
        max_lr = None
        max_lr_dec = None
        max_val = None
        dist = -2.1 + 4
        grid = return_grid([-4,-2.1], dist)
        for lr in grid:
            dist2 = 1 - 0.7
            grid2 = return_grid([0.7, 1], dist2)
            for lr_decay in grid2:
                values = []
                max_item = None
                max_val = 0
                args['lr_decay'] = lr_decay
                args['lr'] = lr
                print(args)
                values.append(get_vals(**args))
                if values[-1]>max_val:
                    max_val = values[-1]
                    max_lr = lr
                    max_lr_dec = lr_decay
        print ("Maximum Val was : " + str(max_val) + " Lr " + str(max_lr) + " lr_dec " + str(max_lr_dec))

lr_decay = 0.94
reg = 0
lr = -3
hd1 = 100
hd2 = 100
drop =0
path = "/vol/bitbucket/ML_pickle"
kwargs = {'lr': lr ,'hidden_dims1' : hd1,'hidden_dims2': hd2,'lr_decay':lr_decay,'reg':reg,'drop':drop}

#OPTIMISE THE LEARNING RATE BETWEEN 10**-6 and 10**-2.1
json_log2 = open("tune_lr_lr_decay_together_grid.json", mode='wt', buffering=1)
json_log = open("tune_lr_lr_decay_together_grid_.json", mode='wt', buffering=1)
minimise_lr_and_lr_decay(old_val=0,**kwargs)
json_log2.close()
json_log.close()

print(kwargs)
