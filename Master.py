#from new_vgg import *
from common import *
from vgg_net import *

'''
TRAIN_DATA = os.path.join("datasets","NEW_NUMBERING_REACT","new_train")
VALIDATE_DATA = os.path.join("datasets","NEW_NUMBERING_REACT","new_test")


datagen = ImageDataGenerator()

validate_generator = datagen.flow_from_directory(VALIDATE_DATA, target_size = (IM_HEIGHT,IM_WIDTH), class_mode = "categorical")
'''
vgg = VGG(cached_model = os.path.join("MODEL_OUTPUTS","checkpoints","intermediate.hdf5"))
#print(vgg.model.evaluate_generator(validate_generator))


TRAIN_DATA = os.path.join("datasets","Fer2013pu","public","Train")
VALIDATE_DATA = os.path.join("datasets","Fer2013pu","public","Test")
vgg.train(TRAIN_DATA ,VALIDATE_DATA,'final_stage_old_network',25)

'''
vgg = VGG()
vgg.train(TRAIN_DATA ,VALIDATE_DATA,'pre_trained_old',30)


TRAIN_DATA = os.path.join("datasets","Fer2013pu","public","Train")
VALIDATE_DATA = os.path.join("datasets","Fer2013pu","public","Test")



model_loc = os.path.join('MODEL_OUTPUTS','models','vgg_netpre_trained_old.hdf5')


vgg = VGG(cached_model = model_loc)
vgg.train(TRAIN_DATA ,VALIDATE_DATA,'final_stage_old_network',25)
'''




#TRAIN_DATA =
#VALIDATE_DATA = vgg = VGG_new()
