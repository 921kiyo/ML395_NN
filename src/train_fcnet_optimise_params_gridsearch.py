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
                    num_epochs=100, batch_size=70,
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
    grid = np.arange(value[0],value[1]+dist/10,dist/10,dtype=np.float64)
    return grid

def minimise(old_val,**args):#lr,hidden_dims1,hidden_dims2,lr_decay,reg):
    global kwargs
    if args is not None:

        for key, value in args.items():
            #IF THE KEY CONTAINS A LIST - BOUNDARIES FOR CURRENT
            #PARAMETER SEARCH
            if isinstance(value, list):
                dist = value[1] - value[0]
                grid = return_grid(value,dist)
                print("New Grid is :" + str(grid))
                values = []
                max_item = None
                max_val = 0
                #FIND THE BEST VALUE IN THE GRID
                for item in range(len(grid)):
                    print("Trying item no: " +str(item) + " and value : " + str(grid[item]))
                    args[key] = grid[item]
                    values.append(get_vals(**args))
                    if values[-1]>max_val:
                        max_val = values[-1]
                        max_item = item

                #CREATE NEW BOUNDARY AROUND THE BEST FOUND VALUE IN THE PREVIOUS GRID SEARCH
                args[key] = [grid[max_item]-dist/10, grid[max_item]+dist/10]

                #IF THIS ITERATION HAS MADE A SIGNIFICANT DIFFERENCE > 1% RECURSE WITH NEW BOUNDARY
                if(old_val <= max_val-0.01):
                    old_val = max_val
                    print("Best Accuracy is :" + str(max_val))
                    minimise(old_val=old_val,**args)
                #ELSE FINISH TUNING THE HYPERPARAMETER
                else:
                    kwargs[key] = grid[max_item]
                    print("****************************************************")
                    print("               NO GAIN FINISHED TUNING " + key)
                    print("           BEST VALUE IS " + str(max_val))
                    print("****************************************************")

lr_decay = 0.94
reg = 0
lr = -3
hd1 = 544
hd2 = 801
drop =0
path = "/vol/bitbucket/ML_pickle"
kwargs = {'lr': lr ,'hidden_dims1' : hd1,'hidden_dims2': hd2,'lr_decay':lr_decay,'reg':reg,'drop':drop}
for i in range(3):
    #OPTIMISE THE LEARNING RATE BETWEEN 10**-6 and 10**-2.1
    json_log2 = open("tune_lr_grid.json", mode='wt', buffering=1)
    json_log = open("tune_lr_grid_.json", mode='wt', buffering=1)
    kwargs['lr'] = [-6,-2.1]
    minimise(old_val=0,**kwargs)
    json_log2.close()
    json_log.close()


    #OPTIMISE THE LEARNING DECAY RATE BETWEEN 0 and 1
    json_log2 = open("tune_lr_decay_grid.json", mode='wt', buffering=1)
    json_log = open("tune_lr_decay_grid_.json", mode='wt', buffering=1)
    kwargs['lr_decay'] = [0,1]
    minimise(old_val=0,**kwargs)
    json_log2.close()
    json_log.close()

    #OPTIMISE HIDDEN LAYER 2 DIMENSIONS BETWEEN 100 and 1000
    json_log2 = open("tune_reg_grid.json", mode='wt', buffering=1)
    json_log = open("tune_reg_grid_.json", mode='wt', buffering=1)
    kwargs['reg'] = [0,5]
    minimise(old_val=0,**kwargs)
    json_log2.close()
    json_log.close()
    kwargs['reg'] = 0

    #OPTIMISE HIDDEN LAYER 2 DIMENSIONS BETWEEN 100 and 1000
    json_log2 = open("tune_drop_grid.json", mode='wt', buffering=1)
    json_log = open("tune_drop_grid_.json", mode='wt', buffering=1)
    kwargs['drop'] = [0,1]
    minimise(old_val=0,**kwargs)
    json_log2.close()
    json_log.close()
    kwargs['drop'] = 0

    #OPTIMISE HIDDEN LAYER 1 DIMENSIONS BETWEEN 100 and 1000
    json_log2 = open("tune_hiddem_dims1_grid.json", mode='wt', buffering=1)
    json_log = open("tune_hiddem_dims1_grid_.json", mode='wt', buffering=1)
    kwargs['hidden_dims1'] = [10,1000]
    print(kwargs)
    minimise(old_val=0,**kwargs)
    json_log2.close()
    json_log.close()

    #OPTIMISE HIDDEN LAYER 2 DIMENSIONS BETWEEN 100 and 1000
    json_log2 = open("tune_hidden_dims2_grid.json", mode='wt', buffering=1)
    json_log = open("tune_hidden_dims2_grid_.json", mode='wt', buffering=1)
    kwargs['hidden_dims2'] = [10,1000]
    minimise(old_val=0,**kwargs)
    json_log2.close()
    json_log.close()



print(kwargs)
