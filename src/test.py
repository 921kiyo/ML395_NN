# Function to load an test the model for Q5
import numpy as np
import pickle
import os
import glob
from scipy import misc
from PIL import Image

# Make this a member function of the model?
def predict(X,batch_size=100, model = None):
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

# Paths for model and test data
path_to_model = 'Q5mod_epoch_61.pkl'
path_to_images = "/home/greg/Desktop/Q5/ML395_NN/datasets/FER2013/Train"


def test_fer_model(img_folder, model_path):
    # Load the model from pickle and set to testing mode
    model_data = pickle.load(open(model_path,'rb'))
    model = model_data['model']
    model.dropout_params['train'] = False

    # Get image names
    image_names = sorted(glob.glob(path_to_images + "/*.jpg"))
    n = len(image_names)

    # Load images and predict in batches
    batch_size = 1000

    predictions = []
    test_data = []
    n_batch = 0
    for i in range(0,n):
        # Load a batch of grayscale images
        im = Image.open(os.path.join(path_to_images,image_names[i])) #im = misc.imread(os.path.join(path_to_images,image_names[i]), mode= 'F')
        im = im.convert('F')

        # SUBTRACT mean here


        imex = np.expand_dims(im, axis=0)
        test_data.append(imex)
        n_batch += 1
        if n_batch == batch_size or i == n-1:
            # Predict on the batch and append results to overall predictions
            con = np.concatenate(test_data, axis=0)
            p_batch = predict(con, model=model)
            predictions.append(p_batch)
            test_data = []
            n_batch = 0

    # Return predictions for entire directory as np.array
    predictions = np.concatenate(predictions)
    return  predictions



overall_pred = test_fer_model(path_to_images, path_to_model)
from src.get_acc import get_labels
labs = get_labels()
keys_in_order = sorted(list(labs.keys()))
labs_in_order = []

for j in range(len(keys_in_order)):
    if keys_in_order[j][1] != 'e':
        labs_in_order.append(labs[keys_in_order[j]])

labs_in_order = np.array(labs_in_order)

print("Accuracy is: {}".format(np.mean(labs_in_order == overall_pred)))



