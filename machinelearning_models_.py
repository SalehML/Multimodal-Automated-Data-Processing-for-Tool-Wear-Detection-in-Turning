# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 13:40:22 2024

@author: svalizad
"""

import tensorflow as tf
from ViT_Code import create_vit_classifier
from Transformer_TS import transformer_encoder
import gc

class time_series_feature_extractor:
    
    def __init__(self, input_shape, dropout, l2, layers):
        
        self.input_shape = input_shape
        
        self.dropout = dropout
        
        self.layers = layers
        
        self.l2 = l2
        
    def mlpClassifier(self):
        
        neuron_1 = 30
        
        input_layer = tf.keras.layers.Input(shape = (5,), name = "Input_Layer")
        first_layer = tf.keras.layers.Dense(neuron_1, kernel_regularizer=tf.keras.regularizers.l2(self.l2), name = "Class_I")(input_layer)
        first_layer = tf.keras.layers.Dropout(self.dropout)(first_layer)
        first_layer = tf.keras.activations.relu(first_layer)
        
        
        mlp_model = tf.keras.models.Model(input_layer, first_layer)
        
        return mlp_model
    
    def feature_extractor(self, policy):
        
        if policy == 0:
            
            RNN_input = tf.keras.layers.Input(shape = self.input_shape, name = "Time_Input")
            
            RNN_input_mask = tf.keras.layers.Masking(mask_value = 0.0)(RNN_input)
            
            for _ in range(self.layers):
            
                RNN_input_mask = tf.keras.layers.LSTM(128, return_sequences=True, activation = "tanh", kernel_regularizer = tf.keras.regularizers.l2(self.l2))(RNN_input_mask)
        
            RNN_second = tf.keras.layers.LSTM(256 , return_sequences=False, activation = "tanh", kernel_regularizer = tf.keras.regularizers.l2(self.l2))(RNN_input_mask)
            RNN_second = tf.keras.layers.Dropout(self.dropout, name = "Dropout_III_II")(RNN_second)
            
            model = tf.keras.models.Model(RNN_input, RNN_second)
        
        
        #elif policy == 1:
            
            #RNN_input = tf.keras.layers.Input(shape = self.input_shape, name = "Time_Input")
            
            #RNN_input_mask = tf.keras.layers.Masking(mask_value = 0.0)(RNN_input)
            
            #for _ in range(self.layers):
                
                #RNN_input_mask = transformer_encoder(RNN_input, 16, 4, 2, self.dropout)
                
            #RNN_second = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_last")(RNN_input_mask)
            
            #model = tf.keras.models.Model(RNN_input, RNN_second)
            
        
        
        elif policy == 1:
            
            RNN_input = tf.keras.layers.Input(shape = self.input_shape, name = "Time_Input")
            
            RNN_input_mask = tf.keras.layers.Masking(mask_value = 0.0)(RNN_input)
            
            for _ in range(self.layers):
                
                RNN_input_mask = tf.keras.layers.GRU(128, return_sequences=True, activation = "tanh", kernel_regularizer = tf.keras.regularizers.l2(self.l2))(RNN_input_mask)
            
            
            RNN_second = tf.keras.layers.GRU(256 , return_sequences=False, activation = "tanh", kernel_regularizer = tf.keras.regularizers.l2(self.l2))(RNN_input_mask)
            RNN_second = tf.keras.layers.Dropout(self.dropout, name = "Dropout_III_II")(RNN_second)
            
            model = tf.keras.models.Model(RNN_input, RNN_second)
        
        elif policy > 1:
            
            model = self.mlpClassifier()
    
        return model
    

class image_feature_extractor:
    
    def __init__(self):
        
        self.input_shape = (224, 224, 3)
        self.weights = "imagenet"

    def feature_extractor(self, policy):
        
        if policy == 0:
        
            feature_extractor = tf.keras.applications.vgg16.VGG16(include_top = False, 
                                                             weights = self.weights,
                                                             input_shape = self.input_shape
                                                             )
        elif policy == 1:
            
            feature_extractor = tf.keras.applications.vgg19.VGG19(include_top = False, 
                                                                 weights = self.weights,
                                                                 input_shape = self.input_shape
                                                                 )
            
        elif policy == 2:
            
            feature_extractor = tf.keras.applications.resnet50.ResNet50(include_top = False, 
                                                                 weights = self.weights,
                                                                 input_shape = self.input_shape
                                                                 )
            
        elif policy == 3:
            
            feature_extractor = tf.keras.applications.ResNet101V2(include_top = False, 
                                                                 weights = self.weights,
                                                                 input_shape = self.input_shape
                                                                 )
            
            
        elif policy == 4:
            
            feature_extractor = tf.keras.applications.inception_v3.InceptionV3(include_top = False, 
                                                                 weights = self.weights,
                                                                 input_shape = self.input_shape
                                                                 )
            
        elif policy == 5:
            
            feature_extractor = tf.keras.applications.xception.Xception(include_top = False, 
                                                                 weights = self.weights,
                                                                 input_shape = self.input_shape
                                                                 )
            
        
        elif policy == 6:
            
            feature_extractor = create_vit_classifier()
            
        return feature_extractor
        
        """
        if policy < 5:
            
        
            feature_extractor_input = feature_extractor.input
            
            feature_extractor_output = feature_extractor.output
            
            representation = tf.keras.layers.LayerNormalization(epsilon = 1e-6)(feature_extractor_output)
            
            representation = tf.keras.layers.Flatten()(representation)
            
            feature_extractor_section = tf.keras.models.Model(feature_extractor_input, representation)
            
            return feature_extractor_section
    
        elif policy == 5:
        
            return feature_extractor
        """


def image_data(policy):
    
    model_feature = image_feature_extractor()
        
    feature_extractor_I = model_feature.feature_extractor(policy)
        
    try:
        feature_extractor_II = tf.keras.models.clone_model(feature_extractor_I)
    except:
        
        feature_extractor_II = model_feature.feature_extractor(policy)
        
    for layer in feature_extractor_I.layers:
        
        try:
            layer._name = layer._name + str('_flank_image')
        except:
            layer.name = layer.name + str('_flank_image')
        
    for layer in feature_extractor_II.layers:
        
        try:
            layer._name = layer._name + str('_rake_image')
        except:
            layer.name = layer.name + str('_rake_image')
        
    del model_feature
    
    gc.collect()
        
    return feature_extractor_I, feature_extractor_II

def time_series_data(policy, dropout, l2, layers):
    
    if policy <= 1:
    
        model_feature = time_series_feature_extractor((1500, 5), dropout, l2, layers)
        
        feature_extractor = model_feature.feature_extractor(policy)
        
    elif policy > 1:
        
        model_feature = time_series_feature_extractor((5, ), dropout, l2, layers)
        
        feature_extractor = model_feature.feature_extractor(policy)
    
        
    for layer in feature_extractor.layers:
        
        try:
            layer._name = layer._name + str('_time_series')
        except:
            layer.name = layer.name + str('_time_series')
        
    del model_feature
    
    gc.collect()
        
    return feature_extractor
        
    