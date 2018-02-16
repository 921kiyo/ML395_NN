import pandas as pd
import os
import numpy as np

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



def format_images(dir_):
    data_folder = os.path.join('datasets','Fer2013pu','public')
    labels = os.path.join(data_folder,'labels_public.txt')
    training_folder = os.path.join(data_folder,dir_)

    df = pd.read_csv(labels,header = 0, sep = ',', engine='python')
    print(df.columns.values)
    folders = range(0,7)

    for folder in folders:
        if not os.path.exists(os.path.join(training_folder,str(folder))):
            os.mkdir(os.path.join(training_folder,str(folder)))
    #print("working")
    for image in get_image_names(training_folder):
        row = df[df['img'] == str(dir_ + "/" + image)]
        if len(row) > 0:
            #print(str(row.iloc[0,1]))
            os.rename(os.path.join(training_folder,image),os.path.join(training_folder,str(row.iloc[0,1]),image))