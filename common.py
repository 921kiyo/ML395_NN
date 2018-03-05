#File containing the main parameters used for the neural network
import os
#vgg_net images MUST BE DIVISIBLE BY 4!!!
IM_HEIGHT = 48
IM_WIDTH = 48

NUMBER_CLASSES = 7
NUMBER_EPOCHS = 10
NUMBER_CHANNELS = 1
BATCH_SIZE = 64

SEND_TO_SLACK = False

MODEL_OUTPUTS_FOLDER = "MODEL_OUTPUTS"
CHECKPOINTS_FOLDER = os.path.join('MODEL_OUTPUTS','checkpoints')
MODEL_SAVE_FOLDER = os.path.join('MODEL_OUTPUTS','models')
TENSORBOARD_LOGS_FOLDER = os.path.join('MODEL_OUTPUTS','logs')
INTERMEDIATE_FILE = os.path.join('MODEL_OUTPUTS','checkpoints','intermediate.hdf5')
TRAIN_DATA = os.path.join("datasets","Fer2013pu","public","Train")
VALIDATE_DATA = os.path.join("datasets","Fer2013pu","public","Test")
