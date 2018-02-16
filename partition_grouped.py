'''
Splits a folder with source\subdirs\images into:
    test_data\subdirs\images
    validation_data\subdirs\images
    training_data\subdirs\images

Images are grouped according to their timestamps e.g.
file name '5000169217429_2017-06-24_13.46.42.629_p_049.jpg'
has time stamp: '2017-06-24_13.46.42.629'

The order of the groups is shuffled and then the groups are copied into
the test, validation and training folders according the the ratios specified.

'''
import os
import random
import sys
import shutil


def sort_data(source,train,validate):
    # Set percentage of images for testing and validation (remainder is training set)
    validate_pct = 0.1
    train_pct = 1 - validate_pct

    #source = "product-image-dataset"
    #train = "training_data"
    #validate = "validation_data"
    #test = "test_data"
    dir = os.getcwd()

    source_dir = os.path.join(dir, source)
    validate_dir = os.path.join(dir, validate)
    train_dir = os.path.join(dir, train)

    # Delete directories if they already exist, and make new ones
    for d in [validate_dir, train_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    # Loop through the subdirectories of the source data (i.e image class folders)
    for subdir in os.listdir(source_dir):

        # Skip any items in the source directory that are not directories
        if not os.path.isdir(os.path.join(source_dir, subdir)):
            continue

        # Make the corresponding subdirectories in each of the destination directories
        os.makedirs(os.path.join(validate_dir, subdir))
        os.makedirs(os.path.join(train_dir, subdir))

        # Loop through the image files in a subdirectory
        files =  os.listdir(os.path.join(source_dir, subdir))
        length_files = len(files)
        #Make a list from 0 to n_groups and randomise the order
        random.shuffle(files)

        #Split random list into test, validation and training directories
        path = os.path.join(source_dir, subdir)
        j=0

        while(j <  length_files*(validate_pct)):
            shutil.copy(os.path.join(path, files[j]), os.path.join(validate_dir, subdir))
            j+=1

        while(j <= length_files):
            shutil.copy(os.path.join(path, files[j]), os.path.join(train_dir, subdir))
            j+=1
