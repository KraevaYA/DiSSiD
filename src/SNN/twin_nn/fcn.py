import numpy as np

# FCN model
import tensorflow.keras as keras
import tensorflow as tf
import time
import os
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import utils


class twin_FCN:
    
    def __init__(self, input_shape=None):
        
        self.model = self.build_model(input_shape)
        
        return
    
    
    def build_model(self, input_shape):
        
        input_layer = keras.layers.Input(input_shape)
        
        conv1 = keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')(input_layer)
        conv1 = keras.layers.BatchNormalization()(conv1)
        conv1 = keras.layers.Activation(activation='relu')(conv1)
        
        conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
        conv2 = keras.layers.BatchNormalization()(conv2)
        conv2 = keras.layers.Activation('relu')(conv2)
        
        conv3 = keras.layers.Conv1D(filters=128, kernel_size=3, padding='same')(conv2)
        conv3 = keras.layers.BatchNormalization()(conv3)
        conv3 = keras.layers.Activation('relu')(conv3)
        
        conv4 = keras.layers.Conv1D(filters=64, kernel_size=7, padding='same')(conv3)
        conv4 = keras.layers.BatchNormalization()(conv4)
        conv4 = keras.layers.Activation('relu')(conv4)
        
        gap_layer = keras.layers.GlobalAveragePooling1D()(conv3)

        model = keras.models.Model(inputs=input_layer, outputs=gap_layer)

        return model
    
    
    def get_model(self):
        
        return self.model

