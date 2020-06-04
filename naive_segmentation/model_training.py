import tensorflow as tf
import keras
import os
import cv2 as cv
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
import yaml
import pandas as pd
import argparse

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Model, Sequential, load_model
from keras.layers import * 
from keras import backend as K
from keras import losses
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger, TensorBoard, EarlyStopping
from keras.metrics import categorical_accuracy

from XNet import model as xnet
from UNet import model as unet


def parse():
    '''
    Parse input arguments
    '''
    parser = argparse.ArgumentParser(description='Train XNet or UNet model for binary classification given a YAML file')

    parser.add_argument(
        '--config_file',
        dest='config_file',
        help='Config YAML file full path',
        type=str
    )

    args = parser.parse_args()

    return args
        

class Train:
    """
    Given a config YAML file containin training parameters, train a segmentation model for binary classification
    """
    
    def __init__(self, config_file):
        """
        config_file: YAML config file
        """
        self.config_file = config_file

        self.read_yaml()
        self.read_csvs()
        self.image_label_test()

    def check_path(self, directory):
        'Check if a directory exists which should exitst'

        if os.path.exists(directory):
            pass
        else:
            raise ValueError('Directory not found: {}'.format(directory))

    def read_yaml(self):
        'Read YAML config file to extract training parameters'

        with open(self.config_file, 'rb') as f:
            config = yaml.load(f)

        self.architecture = config['architecture']
        if self.architecture == 'xnet' or self.architecture == 'unet':
            pass
        else:
            raise ValueError('The architecture must be xnet or unet, you passed {}'.format(self.architecture))

        self.tile_dir = config['tile dir']
        self.check_path(self.tile_dir)

        self.label_dir = config['label dir']
        self.check_path(self.label_dir)

        self.model_dir = config['model dir']
        self.check_path(self.model_dir)

        self.model_name = config['model name']
        self.model_path = self.model_dir + self.model_name

        self.train_file = config['training parameters']['train names'] # csv file containing training file names
        self.check_path(self.train_file)

        self.val_file = config['training parameters']['val names'] # csv file containing validation file names
        self.check_path(self.val_file)

        self.filter_depth = []
        for i in config['training parameters']['filter depth']:
            self.filter_depth.append(config['training parameters']['filter depth'][i])
        
        self.batch_size = config['training parameters']['batch size'] # int
        self.epochs_before_es = config['training parameters']['epochs before ES'] # int
        self.epochs_after_es = config['training parameters']['epochs after ES'] # int
        self.steps_per_epoch = config['training parameters']['steps per epoch'] # int
        self.validation_steps = config['training parameters']['validation steps'] # int

        self.background_weight = config['training parameters']['class weights']['background']
        self.flood_weight = config['training parameters']['class weights']['flood']

        print ('Setting background weight to: {}, and flood weight to: {}'.format(self.background_weight, self.flood_weight))
        
    def read_csvs(self):
        'Read train and validation names csv files'

        ## Note: need to alter this to account for the file names column and convert to numpy array
        self.train_images = np.array(pd.read_csv(self.train_file)).flatten()
        self.val_images = np.array(pd.read_csv(self.val_file)).flatten()

        print ('Training on {} images'.format(len(self.train_images)))
        print ('Validating on {} images'.format(len(self.val_images)))

    def image_label_test(self):
        'Check image and label tile shapes and consistency'

        # randomly choose an index
        test_index = np.random.randint(len(self.train_images))
        test_image = Image.open(self.tile_dir + self.train_images[test_index])
        test_image = np.array(test_image)
        self.height, self.width = test_image.shape
        print ('Find image dimensions to be {},{}'.format(self.height, self.width))
        
        test_label = Image.open(self.label_dir + self.train_images[test_index])
        

    def train_generator(self):
        'Custom train generator'
        
        no_images = len(self.train_images)
        while True:
            indices = np.asarray(range(0, no_images))
            shuffle(indices)
            for idx in range(0, len(indices), self.batch_size):
                batch_indices = indices[idx:idx+self.batch_size]
                batch_indices.sort()
                batch_indices = batch_indices.tolist()

                images = []
                labels = []
                for i in batch_indices:
                    try:
                        image = Image.open(self.tile_dir + self.train_images[i])
                        image = np.array(image).reshape(self.height,self.width,1)

                        label = Image.open(self.label_dir + self.train_images[i])
                        label = np.array(label)
                        b = np.zeros((label.size, label.max()+1))
                        b[np.arange(label.size),label.flatten()] = 1
                        label = b

                        image = image - np.mean(image)
                        image = (image-np.min(image))/(np.max(image) - np.min(image)) 
                        images.append(image)
                        labels.append(label)
                    except:
                        pass

                images = np.array(images)
                images = images.reshape(-1,self.height,self.width,1)
                labels = np.array(labels)

                yield(images,labels)


    def val_generator(self):
        'Custom validation generator'
        
        no_images = len(self.val_images)
        while True:
            indices = np.asarray(range(0, no_images))
            shuffle(indices)
            for idx in range(0, len(indices), self.batch_size):
                batch_indices = indices[idx:idx+self.batch_size]
                batch_indices.sort()
                batch_indices = batch_indices.tolist()

                images = []
                labels = []
                for i in batch_indices:
                    try:
                        image = Image.open(self.tile_dir + self.val_images[i])
                        image = np.array(image).reshape(self.height,self.width,1)

                        label = Image.open(self.label_dir + self.val_images[i])
                        label = np.array(label)
                        b = np.zeros((label.size, label.max()+1))
                        b[np.arange(label.size),label.flatten()] = 1
                        label = b

                        image = image - np.mean(image)
                        image = (image-np.min(image))/(np.max(image) - np.min(image)) 
                        images.append(image)
                        labels.append(label)
                    except:
                        pass

                images = np.array(images)
                images = images.reshape(-1,self.height,self.width,1)
                labels = np.array(labels)

                yield(images,labels)

    def pixel_wise_loss(self,y_true, y_pred):
        
        pos_weight = tf.constant([[float(self.background_weight), float(self.flood_weight)]])
        loss = tf.nn.weighted_cross_entropy_with_logits(
            y_true,
            y_pred,
            pos_weight,
            name=None
        )

        return K.mean(loss,axis=-1)

    def compile_model(self):
        'Compile model'

        if self.architecture == 'xnet':
            print ('Training on XNet')
            self.mod = xnet(input_shape=(self.height,self.width,1), classes=2, kernel_size = 3, filter_depth = self.filter_depth)
        if self.architecture == 'unet':
            print ('Training on UNet')
            self.mod = unet(input_shape=(self.height,self.width,1), classes=2, kernel_size = 3, filter_depth = self.filter_depth)

        #self.mod.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
        self.mod.compile(optimizer = Adam(lr = 1e-4), loss = self.pixel_wise_loss, metrics = ['accuracy'])

            
    def fit(self):
        'Fit model using training and validation data and the generators'

        self.compile_model()
        
        train_gen = self.train_generator()
        val_gen = self.val_generator()

        if not os.path.exists(self.model_path):
            print('Creating {}'.format(self.model_path))
            os.mkdir(self.model_path)

        csv_logger_before_es = CSVLogger(self.model_path + "/logging_before_es.csv")
        csv_logger_after_es = CSVLogger(self.model_path + "/logging_after_es.csv")
        earlystop = EarlyStopping(monitor="val_loss", min_delta = 0, patience = 10, verbose = 1, mode = 'min')
        checkpoint_acc = ModelCheckpoint(self.model_path + "/acc.h5", monitor = "val_acc", verbose = 1, save_best_only = True, save_weights_only = False, mode = "auto", period = 1)
        checkpoint_loss = ModelCheckpoint(self.model_path + "/loss.h5", monitor = "val_loss", verbose = 1, save_best_only = True, save_weights_only = False, mode = "auto", period = 1)
        
        self.mod.fit_generator(train_gen, steps_per_epoch=self.steps_per_epoch, epochs=self.epochs_before_es, callbacks=[csv_logger_before_es,checkpoint_acc,checkpoint_loss], validation_data = val_gen, validation_steps=self.validation_steps)
        
        self.mod.fit_generator(train_gen, steps_per_epoch=self.steps_per_epoch, epochs=self.epochs_after_es, callbacks=[csv_logger_after_es,checkpoint_acc,checkpoint_loss,earlystop], validation_data = val_gen, validation_steps=self.validation_steps)


if __name__ == "__main__":
    
    args = parse()

    train = Train(args.config_file)

    train.fit()
