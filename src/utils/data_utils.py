from builtins import range
from six.moves import cPickle as pickle
import numpy as np
import os
from scipy.misc import imread
import platform

def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return  pickle.load(f)
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))

def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
        Y = np.array(Y)
        return X, Y

def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte


def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000,
                     subtract_mean=True):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for classifiers. These are the same steps as we used for the SVM, but
    condensed to a single function.
    """
    # Load the raw CIFAR-10 data
    path = "/vol/bitbucket/395ML_NN_Data/"
    #path = "C:/Users/Peter/Documents/Machine_Learning/ML395_NN"
    cifar10_dir = os.path.join(path,'datasets','cifar-10-batches-py')
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    if subtract_mean:
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image

    # Transpose so that channels come first
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()

    # Package data into a dictionary
    return {
      'X_train': X_train, 'y_train': y_train,
      'X_val': X_val, 'y_val': y_val,
      'X_test': X_test, 'y_test': y_test,
    }

def load_FER_2013(filename):
    """ Load picke file to dictionary """
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X_train = datadict['X_train']
        X_test = datadict['X_test']
        Y_train = datadict['y_train']
        Y_test = datadict['X_train']
        return X_train, Y_train, X_test, Y_test


def load_FER_2013_not_pickle(fer_folder):
    """ load single batch of cifar """
    datadict = {}
    # Get image names
    train_path = os.path.join(fer_folder,"Train")
    test_path = os.path.join(fer_folder, "Test")

    image_names = sorted(glob.glob(train_path + "/*.jpg"))
    n = len(image_names)

    # Load images and predict in batches
    train_data = []
    for i in range(0,n):
        # Load a batch of grayscale images
        im = Image.open(os.path.join(img_folder,image_names[i]))
        im = im.convert('F')
        imex = np.expand_dims(im, axis=0)
        train_data.append(imex)

    datadict['X_train'] = np.concatenate(test_data, axis=0)

    image_names = sorted(glob.glob(test_path + "/*.jpg"))
    n = len(image_names)

    # Load images and predict in batches
    test_data = []
    for i in range(0,n):
        # Load a batch of grayscale images
        im = Image.open(os.path.join(img_folder,image_names[i]))
        im = im.convert('F')
        imex = np.expand_dims(im, axis=0)
        train_data.append(imex)

    datadict['X_test'] = np.concatenate(test_data, axis=0)

    return  datadict


def get_FeR2013_data(num_training=27709, num_validation=1000, num_test=1000,
                     subtract_mean=True):
    """
    Load the FER2013 from a pickle file and perform preprocessing to prepare
    it for classifiers. These are the same steps as we used for the SVM, but
    condensed to a single function.

    Using the pickled dataset
    """
    path = "/vol/bitbucket/395ML_NN_Data/"
    #path = "C:/Users/Peter/Documents/Machine_Learning/ML395_NN"
    fer2013_dir = "/vol/bitbucket/ML_pickle" #os.path.join(path,'datasets')
    X_train, y_train, X_test, y_test = load_FER_2013(os.path.join(fer2013_dir,"FER2013_data.pickle"))


    # Subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    if subtract_mean:
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image

    # Transpose so that channels come first
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()

    # Package data into a dictionary
    return {
      'X_train': X_train, 'y_train': y_train,
      'X_val': X_val, 'y_val': y_val,
      'X_test': X_test, 'y_test': y_test,
    }

def get_FeR2013_data_not_pickle(num_training=27709, num_validation=1000, num_test=1000,
                     subtract_mean=True):
    """
    Load the FER2013 dataset from disk and perform preprocessing to prepare
    it for classifiers. These are the same steps as we used for the SVM, but
    condensed to a single function.
    """

    # Load the raw FER2013 Data
    path = "/vol/bitbucket/395ML_NN_Data/datasets/FER2013"
    X_train, y_train, X_test, y_test = load_FER_2013_not_pickle(path)

    # Load the raw CIFAR-10 data
    path = "/vol/bitbucket/395ML_NN_Data/"
    #path = "C:/Users/Peter/Documents/Machine_Learning/ML395_NN"
    fer2013_dir = "/vol/bitbucket/ML_pickle" #os.path.join(path,'datasets')
    X_train, y_train, X_test, y_test = load_FER_2013(os.path.join(fer2013_dir,"FER2013_data.pickle"))


    # Subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    if subtract_mean:
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image

    # Transpose so that channels come first
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()

    # Package data into a dictionary
    return {
      'X_train': X_train, 'y_train': y_train,
      'X_val': X_val, 'y_val': y_val,
      'X_test': X_test, 'y_test': y_test,
    }
