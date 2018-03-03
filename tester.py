from builtins import range
from six.moves import cPickle as pickle
import numpy as np
import os
from scipy.misc import imread
import platform
#import pandas as pd
import cv2

def get_image_names(directory_):
    files = os.listdir(directory_)
    image_present = False
    images = []
    for file in files:
        if file.endswith('.jpg') or file.endswith('.png'):
            images.append(file)
            image_present = True

    if not image_present:
        print("Could not find any Images!")
    return images

def get_image(filepath):
    img = cv2.imread(filepath)
    resized_image = np.expand_dims(img, axis=0)
    return resized_image

def dstack_folder(directory_,image_list):
    #image_list = get_image_names(directory_)
    images = get_image(os.path.join(directory_, image_list[0]))
    if len(image_list) > 1:
        for image in image_list[1:]:
            new_image = get_image(os.path.join(directory_, image))
            images = np.concatenate([new_image,images],axis = 0)
    return images

def get_image_labels(dir_,image_list,df):
    labels = []
    for image in image_list:
        row = df[df['img'] == str(dir_ + "/" + image)]
        if len(row) > 0:
            labels.append(row.iloc[0,1])
            print(row.iloc[0,1])
    return labels

def load_FeR2013(dir_):

    data_folder = os.path.join(dir_,'datasets','Fer2013pu','public')
    labels = os.path.join(data_folder,'labels_public.txt')
    training_folder = os.path.join(data_folder,"Train")
    test_folder = os.path.join(data_folder,"Test")

    df = pd.read_csv(labels,header = 0, sep = ',', engine='python')
    image_list = get_image_names(training_folder)
    ytrain = get_image_labels(dir_, image_list, df)
    xtrain = dstack_folder(training_folder,image_list)

    image_list = get_image_names(test_folder)
    xtest = dstack_folder(test_folder, image_list)
    ytest = get_image_labels(dir_,image_list, df)
    return xtrain,ytrain,xtest,ytest



def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return  pickle.load(f)
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))

def load_FER_2013(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        print(datadict)
        X_train = datadict['X_train']
        X_test = datadict['X_test']
        Y_train = datadict['y_train']
        Y_test = datadict['X_train']
        return X_train, Y_train, X_test,Y_test


path = "C:/Users/Peter/Documents/Machine_Learning/ML395_NN"

X_train,Y_train,X_test,Y_test = load_FER_2013(os.path.join(path,"datasets\FER2013_data.pickle"))
print(X_train.shape)



'''
xtrain,ytrain,xtest,ytest = load_FeR2013(path)

df = pd.DataFrame([xtrain,ytrain,xtest,ytest])
df.to_pickle("FerData")
df = pd.DataFrame(list(xtrain))
df.to_pickle("xtrain")
df = pd.DataFrame(ytrain)
df.to_pickle("ytrain")
df = pd.DataFrame(list(xtest))
df.to_pickle("xtest")
df = pd.DataFrame(ytest)
df.to_pickle("ytest")
df = pd.read_pickle("FerData")
print(df.columns)


print(xtrain.shape)
print(len(ytrain))
print(xtest.shape)
print(len(ytest))'''