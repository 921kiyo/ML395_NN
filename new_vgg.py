from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from common import *
import numpy as np
from Load_Images import *
from keras.optimizers import Adadelta


class VGG(object):
    def __init__(self,lr=0.001,cached_model= None):

        self.model_name = "vgg_net"
        self.model_input = (1, IM_HEIGHT, IM_WIDTH, NUMBER_CHANNELS)

        for folder in [MODEL_OUTPUTS_FOLDER,CHECKPOINTS_FOLDER,MODEL_SAVE_FOLDER,TENSORBOARD_LOGS_FOLDER]:
            if not os.path.exists(folder):
                os.mkdir(folder)

        self.model = Sequential()
        self.model.add(Conv2D(64, 5, 5, border_mode='valid',
                                input_shape=(IM_WIDTH, IM_HEIGHT, 1)))
        self.model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
        self.model.add(keras.layers.convolutional.ZeroPadding2D(padding=(2, 2), dim_ordering='tf'))
        self.model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))

        self.model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='tf'))
        self.model.add(Conv2D(64, 3, 3))
        self.model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
        self.model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='tf'))
        self.model.add(Conv2D(64, 3, 3))
        self.model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
        self.model.add(keras.layers.convolutional.AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

        self.model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='tf'))
        self.model.add(Conv2D(128, 3, 3))
        self.model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
        self.model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='tf'))
        self.model.add(Conv2D(128, 3, 3))
        self.model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))

        self.model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='tf'))
        self.model.add(keras.layers.convolutional.AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(1024))
        self.model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1024))
        self.model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
        self.model.add(Dropout(0.2))

        self.model.add(Dense(7))

        self.model.add(Activation('softmax'))

        ada = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
        self.model.compile(loss='categorical_crossentropy',
                      optimizer=ada,
                      metrics=['accuracy'])


        if cached_model is not None:
            self.model = load_model(cached_model)

    def train(self,train_directory_, validation_directory_,model_description,epochs):
        self.model_name += model_description

        tdatagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            shear_range=0.2,
            rotation_range=40,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images

        '''
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.35,
            zoom_range=[0.7, 1.3],
            channel_shift_range=0.2,'''


        datagen = ImageDataGenerator(
            rescale=1. / 255)

        train_generator = tdatagen.flow_from_directory(
            train_directory_,
            color_mode="grayscale",
            target_size=(IM_HEIGHT, IM_WIDTH),
            batch_size=BATCH_SIZE,
            class_mode="categorical")

        validate_generator = datagen.flow_from_directory(
            validation_directory_,
            color_mode="grayscale",
            target_size=(IM_HEIGHT, IM_WIDTH),
            batch_size=BATCH_SIZE,
            class_mode="categorical")  # CHANGE THIS!!!

        self.model.fit_generator(train_generator, validation_data=validate_generator,callbacks=[
                                                             keras.callbacks.TerminateOnNaN(),
                                                             keras.callbacks.ModelCheckpoint(filepath=INTERMEDIATE_FILE,
                                                                                             monitor='val_loss',
                                                                                             verbose=0,
                                                                                             save_best_only=False,
                                                                                             save_weights_only=False,
                                                                                             mode='auto', period=1),
                                                             keras.callbacks.TensorBoard(log_dir=TENSORBOARD_LOGS_FOLDER,
                                                                                         histogram_freq=0,
                                                                                         batch_size=BATCH_SIZE,
                                                                                         write_graph=True,
                                                                                         write_grads=False,

                                                                                         write_images=True,
                                                                                         embeddings_freq=0,
                                                                                         embeddings_layer_names=None,
                                                                                         embeddings_metadata=None)],epochs=epochs)

        current_directory = os.path.dirname(os.path.abspath(__file__))
        print("Model saved to " + os.path.join(current_directory, os.path.pardir, MODEL_SAVE_FOLDER,self.model_name) + '.hdf5')
        if not os.path.exists(MODEL_SAVE_FOLDER):
            os.makedirs(MODEL_SAVE_FOLDER)
        self.model.save(os.path.join(MODEL_SAVE_FOLDER,str(self.model_name + '.hdf5')))


    def predict(self,input_data):
        """
        Given data from 1 frame, predict where the ships should be sent.

        :param input_data: numpy array of shape (PLANET_MAX_NUM, PER_PLANET_FEATURES)
        :return: 1-D numpy array of length (PLANET_MAX_NUM) describing percentage of ships
        that should be sent to each planet
        """
        # CHANGED THIS!!!!
        input_data = input_data / 255
        predictions = self.model.predict(input_data, verbose=False)
        return np.array(predictions[0])


#format_images("Train")
#format_images("Test")
#sort_data(SOURCE_DATA,TRAIN_DATA,VALIDATE_DATA)
vgg = VGG()
vgg.train(TRAIN_DATA ,VALIDATE_DATA,'vgg',NUMBER_EPOCHS)