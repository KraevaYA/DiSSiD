import tensorflow.keras as keras
import tensorflow as tf

from utils import *


class twin_MLP:
    
    def __init__(self, input_shape=None, build=True):
        
        if (build == True):
            self.model = self.build_model(input_shape)
    
        return
    
    
    def build_model(self, input_shape):
        
        input_layer = keras.layers.Input(input_shape)
        
        input_layer_flattened = keras.layers.Flatten()(input_layer)
        
        layer_1 = keras.layers.Dropout(0.1)(input_layer_flattened)
        layer_1 = keras.layers.Dense(500, activation='relu')(layer_1)
        
        layer_2 = keras.layers.Dropout(0.2)(layer_1)
        layer_2 = keras.layers.Dense(500, activation='relu')(layer_2)
        
        layer_3 = keras.layers.Dropout(0.2)(layer_2)
        layer_3 = keras.layers.Dense(500, activation='relu')(layer_3)
        
        output_layer = keras.layers.Dropout(0.3)(layer_3)
        output_layer = keras.layers.Dense(100, activation='relu')(output_layer)
        
        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
                      
        return model

    
    def get_model(self):
        
        return self.model
