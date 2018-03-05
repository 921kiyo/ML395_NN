# Function to load an test the model for Q5

# Loading
import pickle
import os
from scipy import misc
import numpy as np
from src.get_acc import get_labels
import matplotlib.pyplot as plt
from src.utils.data_utils import get_FeR2013_data

def predict(X, num_samples=None, batch_size=100, model = None):

    N = X.shape[0]
    # Compute predictions in batches
    num_batches = N // batch_size
    if N % batch_size != 0:
        num_batches += 1
    y_pred = []
    for i in range(num_batches):
        start = i * batch_size
        end = (i + 1) * batch_size
        scores = model.loss(X[start:end])
        y_pred.append(np.argmax(scores, axis=1))
    y_pred = np.hstack(y_pred)
    return y_pred

obj = get_FeR2013_data()
path_to_model = 'Q5mod_epoch_61.pkl'
path = "/home/greg/Desktop/Q5/ML395_NN"
path_to_images = os.path.join(path, 'datasets','FER2013','Train')


model_data = pickle.load(open(path_to_model,'rb'))
model = model_data['model']
model.dropout_params['train'] = False

image_names = sorted(os.listdir(path_to_images))

n = len(image_names)
predictions = np.zeros(n)

labs = get_labels()

right, all = 0,0

for i in range(n):
    im = misc.imread(os.path.join(path_to_images,image_names[i]), mode= 'F')
    fac = np.ones(shape=im.shape)*135
    #mean_im = np.squeeze(obj['mean_image'],axis=2)
    #im = im - fac
    #im = im - mean_im
    #misc.imsave("im_out.png",im)
    imex = np.expand_dims(im, axis=0)
    #misc.imsave("imex_out.png",imex)
    #na = image_names[0]
    p = predict(imex, model=model)
    predictions[i] = p[0]

    truth = labs["Train/" + image_names[i]]
    all += 1
    if truth == p[0]:
        right+=1

print("Train acc is {}".format(right/all))

plt.hist(predictions)