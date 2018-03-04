import numpy as np
from bayes_opt import BayesianOptimization
from src.fcnet import FullyConnectedNet
from src.utils.solver import Solver
from src.utils.data_utils import get_FeR2013_data
import matplotlib.pyplot as plt
import json

kwargs = None
json_log2 = open("tune_lr_grid.json", mode='wt', buffering=1)
"""
TODO: Use a Solver instance to train a TwoLayerNet that achieves at least 50%
accuracy on the validation set.
"""
###########################################################################
#                           BEGIN OF YOUR CODE                            #
###########################################################################
data = get_FeR2013_data()

def get_vals(lr,hidden_dims1,hidden_dims2,lr_decay,reg):
    global json_log2
    lr = 10 ** lr
    model = FullyConnectedNet(hidden_dims=[int(hidden_dims1),int(hidden_dims2)], input_dim=48 * 48 * 1, reg=reg, num_classes=7, dtype=np.float64)
    solver = Solver(model, data,
                    update_rule='sgd',
                    optim_config={'learning_rate': lr,}, lr_decay = lr_decay,
                    num_epochs=3, batch_size=70,
                    print_every=1000000)

    solver.train()
    solver._save_checkpoint() #save
    val_acc = solver.best_val_acc
    acc = max(solver.train_acc_history)
    loss = min(solver.loss_history)
    json_log2.write(json.dumps({'Learning Rate': lr,
                               'accuracy': acc, 'val_acc': val_acc ,'loss': loss,
                               'layer_1': hidden_dims1,'layer_2': hidden_dims2}) + '\n')
    return solver.best_val_acc

def return_grid(value,dist):
    grid = np.arange(value[0],value[1]+dist/10,dist/10,dtype=np.float64)
    return grid


def minimise(old_val,**args):#lr,hidden_dims1,hidden_dims2,lr_decay,reg):
    global kwargs
    if args is not None:
        for key, value in args.items():
            if isinstance(value, list):
                dist = value[1] - value[0]
                grid = return_grid(value,dist)
                print("New Grid is :" + str(grid))
                values = []
                max_item = None
                max_val = 0
                for item in range(len(grid)):
                    print("Trying item no: " +str(item) + " and value : " + str(grid[item]))
                    args[key] = grid[item]
                    values.append(get_vals(**args))
                    if values[-1]>max_val:
                        max_val = values[-1]
                        max_item = item

                if max_item == len(grid)-1:
                    args[key] = [grid[max_item]-dist/10, grid[max_item]+dist/10]

                if(old_val <= max_val-0.1):
                    old_val = max_val
                    print("Best Accuracy is :" + str(max_val))
                    minimise(old_val=old_val,**args)
                else:
                    kwargs[key] = grid[max_item]
                    print("****************************************************")
                    print("               NO GAIN FINISHED TUNING " + key)
                    print("           BEST VALUE IS " + str(max_val))
                    print("****************************************************")


def kwargs_edit(**args):
    global kwargs
    kwargs['lr'] = 5

lr_decay = 0.85
reg = 0#1e-3
lr = 1e-3
hd1 = 100
hd2 = 100
to_sort = 'lr'
kwargs = {'lr': [-6,-3] ,'hidden_dims1' : hd1,'hidden_dims2': hd2,'lr_decay':lr_decay,'reg':reg}
minimise(old_val=0,**kwargs)
json_log2.close()
json_log2 = open("tune_lr_decay_grid.json", mode='wt', buffering=1)
kwargs['lr_decay'] = [0,1]
minimise(old_val=0,**kwargs)
json_log2.close()
json_log2 = open("tune_hiddem_dims1_grid.json", mode='wt', buffering=1)
kwargs['hidden_dims1'] = [10,1000]
minimise(old_val=0,**kwargs)
json_log2.close()
json_log2 = open("tune_hidden_dims2_grid.json", mode='wt', buffering=1)
kwargs['hidden_dims2'] = [10,1000]
minimise(old_val=0,**kwargs)
json_log2.close()

kwargs_edit(**kwargs)
print(kwargs)



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