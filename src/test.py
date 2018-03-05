# Function to load an test the model for Q5
import numpy as np
import pickle
import os
import glob
from scipy import misc
from PIL import Image

from eval.matrix import *

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


def test_fer_model(img_folder, model_path):
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
        # Load a batch of grayscale images
        im = Image.open(os.path.join(img_folder,image_names[i])) #im = misc.imread(os.path.join(img_folder,image_names[i]), mode= 'F')
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

# Paths for model and test data
# path_to_model = 'Q5mod_epoch_20.pkl'
path_to_model = '/homes/kk3317/Desktop/ML2/Q5mod_epoch_20.pkl'
# path_to_images = "/home/greg/Desktop/Q5/ML395_NN/datasets/FER2013/Train"
path_to_images = "/vol/bitbucket/395ML_NN_Data/datasets/FER2013/Test"

overall_pred = test_fer_model(img_folder, path_to_model)
from src.get_acc import get_labels
labs = get_labels()
keys_in_order = sorted(list(labs.keys()))
labs_in_order = []

for j in range(len(keys_in_order)):
    if keys_in_order[j][1] != 'r':
        labs_in_order.append(labs[keys_in_order[j]])

labs_in_order = np.array(labs_in_order)

print("Accuracy is: {}".format(np.mean(labs_in_order == overall_pred)))


from sklearn.metrics import confusion_matrix, f1_score

f1_score = f1_score(labs_in_order, overall_pred, average=None)
print("F1 Score: ")
print(f1_score)

recall = recall_score(labs_in_order, overall_pred, average=None)
print("Recall: ")
print(recall)

precision = precision_score(labs_in_order, overall_pred, average=None)
print("Precision: ")
print(precision)

confusion = confusion_matrix(labs_in_order, overall_pred)
print("Confusion: ")
print(confusion)

np.set_printoptions(precision=2)
# Plot non-normalized confusion matrix
plt.figure()
class_names = labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
plot_confusion_matrix(confusion, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(confusion, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()
