import numpy as np
import time
import os
import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras import layers

from twin_nn import mlp
from twin_nn import fcn
#from twin_nn import inception
from twin_nn import resnet
import utils


class timecallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.times = []
        # use this value as reference to calculate cumulative time taken
        self.timetaken = time.process_time()
    def on_epoch_end(self,epoch,logs = {}):
        self.times.append((epoch, time.process_time() - self.timetaken))
    def on_train_end(self,logs = {}):
        from operator import itemgetter
        previous_time = 0
        for item in self.times:
            previous_time = item[1]
        return previous_time


class SiameseNN:
    
    def __init__(self, output_dir, input_shape=None, nn_type='ResNet', epochs=20, batch_size=32, margin=1, l_emb=64, mpdist_k=0.05, optimizer='adam', save_params=True):
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.margin = margin  # margin for constrastive loss
        self.optimizer = optimizer
        self.shuffle = True
        self.nn_type = nn_type
        self.l_emb = l_emb
        self.mpdist_k = mpdist_k
        self.train_time = 0
        
        self.save_params = save_params
        print(self.save_params)
        
        self.output_dir = output_dir
        self.model_dir = os.path.join(self.output_dir, 'models')
        #self.plots_dir = os.path.join(self.output_dir, 'plots')
        #self.metrics_dir = os.path.join(self.output_dir, 'metrics')
        
        utils.create_directory(self.model_dir)
        self.model = self.build_SNN_model(input_shape)
        
        if (self.save_params):
            utils.save_model_summary(self.model, os.path.join(self.model_dir, 'snn_summary.txt'))
            #utils.save_snn_params(self.nn_type, self.epochs, self.batch_size, self.margin, self.optimizer, os.path.join(self.model_dir, 'snn_params.json'))
            self.model.save_weights(os.path.join(self.model_dir, 'model_init.hdf5'))
    
        return
    
    
    def _euclidean_distance(self, vects):

        x, y = vects
        
        sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    
        return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))


    def _l1_distance(self, vects):
    
        x, y = vects
        sum_square = tf.math.reduce_sum(tf.math.abs(x - y), axis=1, keepdims=True)
    
        return tf.math.maximum(sum_square, tf.keras.backend.epsilon())
    
    def _mpdist_distance(self, vects):
    
        x, y = vects
        m = x.shape[1]
        l = self.l_emb
        k = self.mpdist_k
        
        if (m == l):
            topk_dist_idx = 0
        else:
            topk_dist_idx = round(2*k*(m-l+1))

        p_ab = []
        for i in range(m-l+1):
            small_subs_a = x[:,i:i+l]
            for j in range(m-l+1):
                small_subs_b = y[:,j:j+l]
                ed_dist = tf.math.reduce_sum(tf.math.square(small_subs_a - small_subs_b), axis=1, keepdims=True)
                if (j == 0):
                    min_ed_dist = ed_dist
                else:
                    min_ed_dist = tf.math.minimum(ed_dist, min_ed_dist)
            p_ab.append(min_ed_dist)

        p_ba = []
        for i in range(m-l+1):
            small_subs_b = y[:,i:i+l]
            for j in range(m-l+1):
                small_subs_a = x[:,j:j+l]
                ed_dist = tf.math.reduce_sum(tf.math.square(small_subs_b - small_subs_a), axis=1, keepdims=True)
                if (j == 0):
                    min_ed_dist = ed_dist
                else:
                    min_ed_dist = tf.math.minimum(ed_dist, min_ed_dist)
            p_ba.append(min_ed_dist)

        p_abba = p_ab + p_ba
        p_abba = tf.transpose(p_abba)[0]
        
        sort_p_abba = tf.sort(p_abba, axis=1)
        
        indices = [topk_dist_idx]
        mpdist = tf.gather(sort_p_abba, indices, axis=1)
    
        return mpdist
    
    
    def _loss(self, margin=1):

        def contrastive_loss(y_true, y_pred):
                
            square_pred = tf.math.square(y_pred)
            margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))

            return tf.math.reduce_mean((1 - y_true) * square_pred + (y_true) * margin_square)
            
        return contrastive_loss
            
            
    def build_SNN_model(self, input_shape):
    
        input_1 = layers.Input(input_shape)
        input_2 = layers.Input(input_shape)
        
        if (self.nn_type == 'MLP'):
            twin_nn = mlp.MLP(input_shape)
        if (self.nn_type == 'FCN'):
            twin_nn = fcn.twin_FCN(input_shape)
        if (self.nn_type == 'ResNet'):
            twin_nn = resnet.ResNet(input_shape)
        #if (self.nn_type == 'Inception'):
        #   twin_model = inception.Classifier_Inception(input_shape, build=self.build)

        twin_nn_model = twin_nn.get_model()
        
        if (self.save_params):
            utils.save_model_summary(twin_nn_model, os.path.join(self.model_dir, 'twin_nn_summary.txt'))

        embedding_1 = twin_nn_model(input_1)
        embedding_2 = twin_nn_model(input_2)
        
        #merge_layer = layers.Lambda(self._l1_distance)([embedding_1, embedding_2])
        merge_layer = layers.Lambda(self._mpdist_distance)([embedding_1, embedding_2])
        #normal_layer = tf.keras.layers.BatchNormalization()(merge_layer)
        
        SNN_model = keras.Model(inputs=[input_1, input_2], outputs=merge_layer)
        
        SNN_model.compile(loss=self._loss(self.margin), optimizer=self.optimizer) #, metrics=["accuracy"]
        
        return SNN_model
    
    
    def fit(self, x_train_1, x_train_2, x_val_1, x_val_2, labels_train, labels_val):
        
        timetaken = timecallback()
        callbacks = [
                     keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001),
                     timetaken
                     ]
                     
        history = self.model.fit([x_train_1, x_train_2], labels_train,
                       validation_data=([x_val_1, x_val_2], labels_val),
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       shuffle=self.shuffle,
                       verbose=True, callbacks=callbacks)
        
        self.train_time = timetaken.on_train_end()
    
        self.model.save_weights(os.path.join(self.model_dir, 'weights.hdf5'))
        utils.save_snn_history(history, os.path.join(self.model_dir, 'history.json'))
        
        keras.backend.clear_session()
    

    def predict(self, x_test_1, x_test_2):
        
        predictions = []
 
        model_path = os.path.join(self.model_dir, 'weights.hdf5')
        if os.path.exists(model_path):
            tf.compat.v1.reset_default_graph() 
            self.model.load_weights(model_path)
        else:
            print("The model doesn't exist")
        
        start_timer = time.time()
        predictions = self.model.predict([x_test_1, x_test_2])
        end_timer = time.time()
    
        test_time = end_timer - start_timer
    
        return predictions, test_time
