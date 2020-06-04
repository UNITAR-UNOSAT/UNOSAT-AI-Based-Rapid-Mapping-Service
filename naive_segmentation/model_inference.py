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

from tqdm import tqdm

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Model, Sequential, load_model
from keras.layers import * 
from keras import backend as K
from keras import losses
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger, TensorBoard, EarlyStopping
from keras.metrics import categorical_accuracy

#from XNet import model

def parse():
    '''
    Parse input arguments
    '''
    parser = argparse.ArgumentParser(description='Test XNet or UNet model for binary classification given a YAML file')

    parser.add_argument(
        '--config_file',
        dest='config_file',
        help='Config YAML file full path',
        type=str
    )

    args = parser.parse_args()

    return args


class Test:

    def __init__(self, config_file):
        """
        config_file: YAML config file
        """
        self.config_file = config_file

        self.read_yaml()
        self.import_model()
        self.read_csvs()
        self.image_label_test()
        self.load_model()

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
        self.check_path(self.model_path)
        self.pred_path = self.model_path + '/predictions/'

        self.batch_size = config['training parameters']['batch size'] # int
        
        self.background_weight = config['training parameters']['class weights']['background']
        self.flood_weight = config['training parameters']['class weights']['flood']

        self.test_file = config['testing parameters']['test names'] # csv file containing testing file names
        self.check_path(self.test_file)

    def import_model(self):
        'Given an architecture, import model'
        
        if self.architecture == 'xnet':
            from XNet import model
            print ('Imported XNet')
        if self.architecture == 'unet':
            from UNet import model
            print ('Imported UNet')

    def read_csvs(self):
        'Read train and validation names csv files'

        ## Note: need to alter this to account for the file names column and convert to numpy array
        self.test_images = np.array(pd.read_csv(self.test_file)).flatten()

        print ('Testing on {} images'.format(len(self.test_images)))

    def image_label_test(self):
        'Check image and label tile shapes and consistency'

        # randomly choose an index
        test_index = np.random.randint(len(self.test_images))
        test_image = Image.open(self.tile_dir + self.test_images[test_index])
        test_image = np.array(test_image)
        self.height, self.width = test_image.shape
        print ('Find image dimensions to be {},{}'.format(self.height, self.width))
        
        test_label = Image.open(self.label_dir + self.test_images[test_index])

    def pixel_wise_loss(self,y_true, y_pred):
        
        pos_weight = tf.constant([[float(self.background_weight), float(self.flood_weight)]])
        loss = tf.nn.weighted_cross_entropy_with_logits(
            y_true,
            y_pred,
            pos_weight,
            name=None
        )

        return K.mean(loss,axis=-1)

    def load_model(self):

        try:
            self.mod = load_model(self.model_path + '/loss.h5')
            print ('Model loaded with inbuilt Keras loss function')
        except:
            self.mod = load_model(self.model_path + '/loss.h5', custom_objects={'pixel_wise_loss': self.pixel_wise_loss})
            print ('Model loaded with custom loss function')

    def load_images(self, image_names):

        images = []
        for image_name in image_names:
            image = Image.open(self.tile_dir + image_name)
            image = np.array(image).reshape(self.height,self.width,1)

            image = image - np.mean(image)
            image = (image-np.min(image))/(np.max(image) - np.min(image)) 
            images.append(image)

        images = np.array(images)
        images = images.reshape(-1,self.height,self.width,1)

        return images

    def save_predictions(self, preds, image_names):

        for idx, pred in enumerate(preds):
            np.save(self.pred_path + image_names[idx].split('.')[0], pred)
            
    def test(self):

        if not os.path.exists(self.pred_path):
            print ('Creating {}'.format(self.pred_path))
            os.mkdir(self.pred_path)

        no_images = len(self.test_images)
        indices = np.arange(0, no_images, self.batch_size)
        indices = np.append(indices, no_images-1)

        for i in tqdm(range(len(indices)-1)):
            images = self.load_images(self.test_images[indices[i]:indices[i+1]])
            preds = self.mod.predict(images)
            self.save_predictions(preds, self.test_images[indices[i]:indices[i+1]])


if __name__ == "__main__":

    args = parse()

    test = Test(args.config_file)

    test.test()
