''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# OUTPUT DATA AND CONFUSION MATRIX FOR REPORT QUESTION 5 and 6
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
from eval.matrix import *
from src.test import test_fer_model, test_deep_fer_model
from src.utils.data_utils import labels_in_order
from sklearn.metrics import confusion_matrix, f1_score
import os

# Question to print results for:
question = 6

# Path to FER data
path_to_fer = "/home/greg/Desktop/Q5_Final/ML395_NN/datasets/FER2013"

# Path to feedforward
ff = '/home/greg/Desktop/Q5_Final/ML395_NN/src/intermediate_epoch_35.pkl' #'/homes/kk3317/Desktop/ML2/Q5mod_epoch_20.pkl' #'Q5mod_epoch_20.pkl'
# Path to CNN
cnn = '/home/greg/Desktop/Q5_Final/ML395_NN/src/question6/models/vgg_netvgg.hdf5' #'/homes/kk3317/Desktop/ML2/Q5mod_epoch_20.pkl' #'Q5mod_epoch_20.pkl'


# Make predictions
overall_pred = None
if question == 5:
    overall_pred = test_fer_model(os.path.join(path_to_fer,"Test"), ff)
if question == 6:
    overall_pred = test_deep_fer_model(os.path.join(path_to_fer, "Test"), cnn)

# Get labels in same order as images
labs_in_order = labels_in_order(fer_folder=path_to_fer, set='test')

# Print results
print("Accuracy is: {}".format(np.mean(labs_in_order == overall_pred)))

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
