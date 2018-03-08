'''''''''
This file contains the implementations of the prediction functions for Q5 and Q6:
* Example usage of both prediction functions can be seen in src.generate_result.py
* Question 6 was implemented using Keras with the Tensorflow back-end
* The order of predictions is that which is given by sorted(glob.glob(img_folder + "/*.jpg")) on the target directory
* The model submitted for Q5 is at:
* The model submitted for Q6 is at:
'''''''''
import pickle
import os
import glob
from PIL import Image

def test_fer_model(img_folder="/vol/bitbucket/395ML_NN_Data/datasets/FER2013/Test", model_path="/homes/kk3317/Desktop/ML395_NN/pkl/Q5mod_epoch_20.pkl"):
    # Load the model from pickle and set to testing mode
    model_data = pickle.load(open(model_path,'rb'))
    model = model_data['model']
    model.dropout_params['train'] = False

    # Get image names
    image_names = sorted(glob.glob(img_folder + "/*.jpg"))
    n = len(image_names)

    # Load images and predict in batches
    batch_size = 1000

    predictions = []
    test_data = []
    n_batch = 0
    for i in range(0,n):
        # Load a batch of grayscale images to predict on
        im = Image.open(os.path.join(img_folder,image_names[i])) #im = misc.imread(os.path.join(img_folder,image_names[i]), mode= 'F')
        im = im.convert('F')
        # Subtract the mean of training set the model was trained with from each image
        im -= np.squeeze(model.mean_image, axis=2)

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


# Prediction function for Q5 feedforward network
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

def test_deep_fer_model(img_folder="/vol/bitbucket/395ML_NN_Data/datasets/FER2013/Test", model_path='/homes/kk3317/Desktop/ML395_NN/'):
    # Get image names
    image_names = sorted(glob.glob(img_folder + "/*.jpg"))
    n = len(image_names)

    # Load images and predict in batches
    batch_size = 1000

    predictions = []
    test_data = []
    n_batch = 0
    for i in range(0,n):
        # Load a batch of grayscale images
        im = Image.open(image_names[i])
        im = im.convert('F')
        imex = np.expand_dims(im, axis=0)
        test_data.append(imex)
        n_batch += 1

        if n_batch == batch_size or i == n-1:
            # Predict on the batch and append results to overall predictions
            con = np.concatenate(test_data, axis=0)
            p_batch = predict_deep(con,batch_size=con.shape[0], model=model_path)
            predictions.append(p_batch)
            test_data = []
            n_batch = 0

    # Return predictions for entire directory as np.array
    predictions = np.concatenate(predictions)
    print("predictions ", predictions)
    return  predictions

# Prediction for Q6 CNN
def predict_deep(X,batch_size=100, model = None):
    from vgg_net import VGG #Only import tensorflow when using the deep network
    N = X.shape[0]
    X = np.expand_dims(X,axis=3)
    # Apply normalisation of test image to match training range of 0-1
    X = X/255
    vgg = VGG(cached_model=model)
    predictions = np.argmax(vgg.model.predict(X,batch_size=100), axis=1)
    return predictions
