from keras.models import Model
from keras.layers import Input, Concatenate, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D
from keras.layers import BatchNormalization, Reshape, Layer
from keras.layers import Activation, Flatten, Dense
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.metrics import categorical_accuracy
from keras import backend as K
from keras import losses

def model(input_shape=(64,64,3), classes=3, kernel_size = 3, filter_depth = (12,24,48,96,192)):
    
    img_input = Input(shape=input_shape)
   
    # Encoder
    
    #296x296
    conv1 = Conv2D(filter_depth[0], (kernel_size, kernel_size), padding="same")(img_input)
    act1 = Activation("relu")(conv1)
    conv2 = Conv2D(filter_depth[0], (kernel_size, kernel_size), padding="same")(act1)
    act2 = Activation("relu")(conv2)
    pool1 = MaxPooling2D(pool_size=(2, 2))(act2)
    
    #148x184
    conv3 = Conv2D(filter_depth[1], (kernel_size, kernel_size), padding="same")(pool1)
    act3 = Activation("relu")(conv3)
    conv4 = Conv2D(filter_depth[1], (kernel_size, kernel_size), padding="same")(act3)
    act4 = Activation("relu")(conv4)
    pool2 = MaxPooling2D(pool_size=(2, 2))(act4)
    
    #74x74
    conv5 = Conv2D(filter_depth[2], (kernel_size, kernel_size), padding="same")(pool2)
    act5 = Activation("relu")(conv5)
    conv6 = Conv2D(filter_depth[2], (kernel_size, kernel_size), padding="same")(act5)
    act6 = Activation("relu")(conv6)
    pool3 = MaxPooling2D(pool_size=(2, 2))(act6)
    
    # Flat
    
    #37x37
    conv7 = Conv2D(filter_depth[3], (kernel_size, kernel_size), padding="same")(pool3)
    act7 = Activation("relu")(conv7)
    conv8 = Conv2D(filter_depth[3], (kernel_size, kernel_size), padding="same")(act7)
    act8 = Activation("relu")(conv8)
    
    #37x37
    x = Conv2D(filter_depth[3], (kernel_size, kernel_size), padding="same")(act8)
    x = Activation("relu")(x)
    x = Conv2D(filter_depth[3], (kernel_size, kernel_size), padding="same")(x)
    act8 = Activation("relu")(x)
    
    # Decoder
    
    up1 = UpSampling2D(size=(2, 2))(act8)
    conv9 = Conv2D(filter_depth[2], (kernel_size, kernel_size), padding="same")(up1)
    act9 = Activation("relu")(conv9)
    conv10 = Conv2D(filter_depth[2], (kernel_size, kernel_size), padding="same")(act9)
    act10 = Activation("relu")(conv10)
    concat1 = Concatenate()([act6,act10])
    # 74x74
    
    up2 = UpSampling2D(size=(2, 2))(concat1)
    conv11 = Conv2D(filter_depth[1], (kernel_size, kernel_size), padding="same")(up2)
    act11 = Activation("relu")(conv11)
    conv12 = Conv2D(filter_depth[1], (kernel_size, kernel_size), padding="same")(act11)
    act12 = Activation("relu")(conv12)
    concat2 = Concatenate()([act4,act12])
    # 148x148
    
    up3 = UpSampling2D(size=(2, 2))(concat2)
    conv13 = Conv2D(filter_depth[0], (kernel_size, kernel_size), padding="same")(up3)
    act13 = Activation("relu")(conv13)
    conv14 = Conv2D(filter_depth[0], (kernel_size, kernel_size), padding="same")(act13)
    act14 = Activation("relu")(conv14)
    concat2 = Concatenate()([act2,act14])
    
    conv15 = Conv2D(classes, (1,1), padding="valid")(concat2)
    
    reshape15 = Reshape((input_shape[0]*input_shape[1],classes))(conv15)
    act15 = Activation("softmax")(reshape15)
    
    model = Model(img_input, act15)

    return model
