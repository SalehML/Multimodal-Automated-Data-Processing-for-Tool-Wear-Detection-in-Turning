# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 19:51:48 2023

@author: Saleh
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
import numpy as np

lr = 0.001

feature_neurons = 15

Loss_function = tf.keras.losses.BinaryCrossentropy()

show_summary = 0
        
Model_optimizer = tf.keras.optimizers.Adam(learning_rate = lr, 
                                           beta_1 = 0.99, 
                                           beta_2 = 0.95)

def design_deep_neural_net(name, transfer_learning):
    
    if transfer_learning == True:
        
        VGG_features_CNN_I = tf.keras.applications.vgg19.VGG19(include_top=False, weights= "imagenet", input_shape=(224,224,3))
        VGG_features_CNN_I.trainable = False
        VGG_features_CNN_I._name = 'VGG_CNN_I'

        VGG_features_CNN_I.summary() if show_summary == 1 else print("")

        VGG_features_CNN_II = tf.keras.applications.vgg19.VGG19(include_top=False, weights= "imagenet", input_shape=(224,224,3))
        VGG_features_CNN_II.trainable = False
        VGG_features_CNN_II._name = 'VGG_CNN_II'

        VGG_features_CNN_II.summary() if show_summary == 1 else print("")
    
        CNN_I_Inputs = tf.keras.layers.Input(shape=(224, 224, 3), name = "Image_Input_I")
        feature_I_extractor = VGG_features_CNN_I(CNN_I_Inputs)
        
        
        CNN_II_Inputs = tf.keras.layers.Input(shape=(224, 224, 3), name = "Image_Input_II")
        feature_II_extractor = VGG_features_CNN_II(CNN_II_Inputs)
        
        
    elif transfer_learning == False:
        
        filter_I, kernel_size_I, stride_I = 32, 2, 1
        
        filter_II, kernel_size_II, stride_II = 64, 2, 1
        
        filter_III, kernel_size_III, stride_III = 64, 2, 1
        
        filter_IV, kernel_size_IV, stride_IV = 128, 2, 1
        
        CNN_I_Inputs = tf.keras.layers.Input(shape=(200, 200, 3), name = "Image_Input_I")
        CNN_I_I = tf.keras.layers.Conv2D(filter_I, kernel_size_I, stride_I, padding = "same", activation = "relu")(CNN_I_Inputs)
        CNN_I_II = tf.keras.layers.Conv2D(filter_I, kernel_size_I, stride_I, padding = "same", activation = "relu")(CNN_I_I)
        MaxPool_I_I = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(CNN_I_II)
        
        CNN_I_III = tf.keras.layers.Conv2D(filter_II, kernel_size_II, stride_II, padding = "same", activation = "relu")(MaxPool_I_I)
        CNN_I_IV = tf.keras.layers.Conv2D(filter_II, kernel_size_II, stride_II, padding = "same", activation = "relu")(CNN_I_III)
        MaxPool_I_II = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(CNN_I_IV)
        
        CNN_I_V = tf.keras.layers.Conv2D(filter_III, kernel_size_III, stride_III, padding = "same", activation = "relu")(MaxPool_I_II)
        CNN_I_VI = tf.keras.layers.Conv2D(filter_III, kernel_size_III, stride_III, padding = "same", activation = "relu")(CNN_I_V)
        MaxPool_I_III = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(CNN_I_VI)
        
        CNN_I_VII = tf.keras.layers.Conv2D(filter_IV, kernel_size_IV, stride_IV, padding = "same", activation = "relu")(MaxPool_I_III)
        CNN_I_VIII = tf.keras.layers.Conv2D(filter_IV, kernel_size_IV, stride_IV, padding = "same", activation = "relu")(CNN_I_VII)
        feature_I_extractor = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2,2))(CNN_I_VIII)



        CNN_II_Inputs = tf.keras.layers.Input(shape=(200, 200, 3), name = "Image_Input_II")
        CNN_II_I = tf.keras.layers.Conv2D(filter_I, kernel_size_I, stride_I, padding = "same", activation = "relu")(CNN_II_Inputs)
        CNN_II_II = tf.keras.layers.Conv2D(filter_I, kernel_size_I, stride_I, padding = "same", activation = "relu")(CNN_II_I)
        MaxPool_II_I = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(CNN_II_II)
        
        CNN_II_III = tf.keras.layers.Conv2D(filter_II, kernel_size_II, stride_II, padding = "same", activation = "relu")(MaxPool_II_I)
        CNN_II_IV = tf.keras.layers.Conv2D(filter_II, kernel_size_II, stride_II, padding = "same", activation = "relu")(CNN_II_III)
        MaxPool_II_II = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(CNN_II_IV)
        
        CNN_II_V = tf.keras.layers.Conv2D(filter_III, kernel_size_III, stride_III, padding = "same", activation = "relu")(MaxPool_II_II)
        CNN_II_VI = tf.keras.layers.Conv2D(filter_III, kernel_size_III, stride_III, padding = "same", activation = "relu")(CNN_II_V)
        MaxPool_II_III = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(CNN_II_VI)
        
        CNN_II_VII = tf.keras.layers.Conv2D(filter_IV, kernel_size_IV, stride_IV, padding = "same", activation = "relu")(MaxPool_II_III)
        CNN_II_VIII = tf.keras.layers.Conv2D(filter_IV, kernel_size_IV, stride_IV, padding = "same", activation = "relu")(CNN_II_VII)
        feature_II_extractor = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2,2))(CNN_II_VIII)
        
        
    flatten_I = tf.keras.layers.Flatten(name = "Flatten_I")(feature_I_extractor)
    class_1 = tf.keras.layers.Dense(feature_neurons , kernel_regularizer = tf.keras.regularizers.l2(0.01),  name = "Class_I")(flatten_I)
    class_1 = tf.keras.activations.relu(class_1)
    class_1 = tf.keras.layers.Dropout(0.25, name = "Dropout_I")(class_1)
    
    flatten_II = tf.keras.layers.Flatten(name = "Flatten_II")(feature_II_extractor)
    class_2 = tf.keras.layers.Dense(feature_neurons, kernel_regularizer = tf.keras.regularizers.l2(0.01), name = "Class_II")(flatten_II)
    class_2 = tf.keras.activations.relu(class_2)
    class_2 = tf.keras.layers.Dropout(0.25, name = "Dropout_II")(class_2)
    
    RNN_input = tf.keras.layers.Input(shape = (60, 8), name = "Time_Input")
    RNN_first = tf.keras.layers.LSTM(64, return_sequences=True, activation = "tanh", kernel_regularizer = tf.keras.regularizers.l2(0.01), name = "RNN_1")(RNN_input)
    #RNN_first = tf.keras.layers.Dropout(0.3, name = "Dropout_III_I")(RNN_first)
    RNN_second = tf.keras.layers.LSTM(128 , return_sequences=False, activation = "tanh", kernel_regularizer = tf.keras.regularizers.l2(0.01), name = "RNN_2")(RNN_first)
    RNN_second = tf.keras.layers.Dropout(0.25, name = "Dropout_III_II")(RNN_second)
    class_3 = tf.keras.layers.Dense(feature_neurons, kernel_regularizer = tf.keras.regularizers.l2(0.01), name = "RNN_III")(RNN_second)
    class_3 = tf.keras.activations.relu(class_3)
    class_3 = tf.keras.layers.Dropout(0.25, name = "Dropout_III_III")(class_3)
    
    
    concat = tf.keras.layers.concatenate([class_1, class_2, class_3], name = "Concat")
    
    concar_1 = tf.keras.layers.concatenate([class_1, class_2], name = "Concat_Image")
    
    concar_2 = tf.keras.layers.concatenate([class_1, class_3], name = "Concat_Image_I_LSTM")
    
    concar_3 = tf.keras.layers.concatenate([class_2, class_3], name = "Concat_Image_II_LSTM")
    
    if name == "Full_model":
        
        concat = tf.keras.layers.BatchNormalization(name = "Batch_Full")(concat)
        """
        classifier_Input = tf.keras.layers.Dense(8, kernel_regularizer = tf.keras.regularizers.l2(0.001), name = "CLassifier_Dense_I")(concat)
        classifier_one = tf.keras.activations.relu(classifier_Input)
        classifier_one = tf.keras.layers.Dropout(0.2, name = "Dropout_Classifier_I")(classifier_one)
        classifier_one = tf.keras.layers.BatchNormalization(name = "Batch_I")(classifier_one)
        
        classifier_two = tf.keras.layers.Dense(16, kernel_regularizer = tf.keras.regularizers.l2(0.001), name = "CLassifier_Dense_II")(classifier_one)
        classifier_two = tf.keras.activations.relu(classifier_two)
        classifier_two = tf.keras.layers.Dropout(0.2, name = "Dropout_Classifier_II")(classifier_two)
        classifier_two = tf.keras.layers.BatchNormalization(name = "Batch_II")(classifier_two)
        """
        classifier_output = tf.keras.layers.Dense(2, activation="softmax", name = "Classifier_III")(concat )
        
        
        Full_model = tf.keras.models.Model([CNN_I_Inputs, CNN_II_Inputs, RNN_input], classifier_output)
        
        return Full_model
    
    elif name == "CNN_I_submodel":
        
        concat = tf.keras.layers.BatchNormalization(name = "Batch_CNN_I")(class_1)
        """
        classifier_Input = tf.keras.layers.Dense(8, kernel_regularizer = tf.keras.regularizers.l2(0.001), name = "CLassifier_Dense_I")(class_1)
        classifier_one = tf.keras.activations.relu(classifier_Input)
        classifier_one = tf.keras.layers.Dropout(0.2, name = "Dropout_Classifier_I")(classifier_one)
        classifier_one = tf.keras.layers.BatchNormalization(name = "Batch_I")(classifier_one)
        
        classifier_two = tf.keras.layers.Dense(16, kernel_regularizer = tf.keras.regularizers.l2(0.001), name = "CLassifier_Dense_II")(classifier_one)
        classifier_two = tf.keras.activations.relu(classifier_two)
        classifier_two = tf.keras.layers.Dropout(0.2, name = "Dropout_Classifier_II")(classifier_two)
        classifier_two = tf.keras.layers.BatchNormalization(name = "Batch_II")(classifier_two)
        """
        classifier_output = tf.keras.layers.Dense(2, activation="softmax", name = "Classifier_III")(concat)
        
        CNN_I_submodel = tf.keras.models.Model(CNN_I_Inputs, classifier_output)
        
        return CNN_I_submodel
    
    elif name == "CNN_II_submodel":
        
        concat = tf.keras.layers.BatchNormalization(name = "Batch_CNN_II")(class_2)
        """
        classifier_Input = tf.keras.layers.Dense(8, kernel_regularizer = tf.keras.regularizers.l2(0.001), name = "CLassifier_Dense_I")(class_2)
        classifier_one = tf.keras.activations.relu(classifier_Input)
        classifier_one = tf.keras.layers.Dropout(0.2, name = "Dropout_Classifier_I")(classifier_one)
        classifier_one = tf.keras.layers.BatchNormalization(name = "Batch_I")(classifier_one)
        
        classifier_two = tf.keras.layers.Dense(16, kernel_regularizer = tf.keras.regularizers.l2(0.001), name = "CLassifier_Dense_II")(classifier_one)
        classifier_two = tf.keras.activations.relu(classifier_two)
        classifier_two = tf.keras.layers.Dropout(0.2, name = "Dropout_Classifier_II")(classifier_two)
        classifier_two = tf.keras.layers.BatchNormalization(name = "Batch_II")(classifier_two)
        """
        classifier_output = tf.keras.layers.Dense(2, activation="softmax", name = "Classifier_III")(concat)
        
        CNN_II_submodel = tf.keras.models.Model(CNN_II_Inputs, classifier_output)
        
        return CNN_II_submodel
    
    elif name == "RNN_submodel":
        
        concat = tf.keras.layers.BatchNormalization(name = "Batch_RNN")(class_3)
        """
        classifier_Input = tf.keras.layers.Dense(8, kernel_regularizer = tf.keras.regularizers.l2(0.001), name = "CLassifier_Dense_I")(class_3)
        classifier_one = tf.keras.activations.relu(classifier_Input)
        classifier_one = tf.keras.layers.Dropout(0.2, name = "Dropout_Classifier_I")(classifier_one)
        classifier_one = tf.keras.layers.BatchNormalization(name = "Batch_I")(classifier_one)
        
        classifier_two = tf.keras.layers.Dense(16, kernel_regularizer = tf.keras.regularizers.l2(0.001), name = "CLassifier_Dense_II")(classifier_one)
        classifier_two = tf.keras.activations.relu(classifier_two)
        classifier_two = tf.keras.layers.Dropout(0.2, name = "Dropout_Classifier_II")(classifier_two)
        classifier_two = tf.keras.layers.BatchNormalization(name = "Batch_II")(classifier_two)
        """
        classifier_output = tf.keras.layers.Dense(2, kernel_regularizer = tf.keras.regularizers.l2(0.01), activation="softmax", name = "Classifier_III")(concat)
        
        RNN_submodel = tf.keras.models.Model(RNN_input, classifier_output)
        
        return RNN_submodel
    
    elif name == "CNN_I_II_submodel":
        
        concat = tf.keras.layers.BatchNormalization(name = "Batch_Full")(concar_1)
        """
        classifier_Input = tf.keras.layers.Dense(8, kernel_regularizer = tf.keras.regularizers.l2(0.001), name = "CLassifier_Dense_I")(concat)
        classifier_one = tf.keras.activations.relu(classifier_Input)
        classifier_one = tf.keras.layers.Dropout(0.2, name = "Dropout_Classifier_I")(classifier_one)
        classifier_one = tf.keras.layers.BatchNormalization(name = "Batch_I")(classifier_one)
        
        classifier_two = tf.keras.layers.Dense(16, kernel_regularizer = tf.keras.regularizers.l2(0.001), name = "CLassifier_Dense_II")(classifier_one)
        classifier_two = tf.keras.activations.relu(classifier_two)
        classifier_two = tf.keras.layers.Dropout(0.2, name = "Dropout_Classifier_II")(classifier_two)
        classifier_two = tf.keras.layers.BatchNormalization(name = "Batch_II")(classifier_two)
        """
        classifier_output = tf.keras.layers.Dense(2, activation="softmax", name = "Classifier_III")(concat )
        
        
        Full_model = tf.keras.models.Model([CNN_I_Inputs, CNN_II_Inputs], classifier_output)
        
        return Full_model
    
    elif name == "Full_model_I":
        
        concat = tf.keras.layers.BatchNormalization(name = "Batch_Full")(concar_2)
        """
        classifier_Input = tf.keras.layers.Dense(8, kernel_regularizer = tf.keras.regularizers.l2(0.001), name = "CLassifier_Dense_I")(concat)
        classifier_one = tf.keras.activations.relu(classifier_Input)
        classifier_one = tf.keras.layers.Dropout(0.2, name = "Dropout_Classifier_I")(classifier_one)
        classifier_one = tf.keras.layers.BatchNormalization(name = "Batch_I")(classifier_one)
        
        classifier_two = tf.keras.layers.Dense(16, kernel_regularizer = tf.keras.regularizers.l2(0.001), name = "CLassifier_Dense_II")(classifier_one)
        classifier_two = tf.keras.activations.relu(classifier_two)
        classifier_two = tf.keras.layers.Dropout(0.2, name = "Dropout_Classifier_II")(classifier_two)
        classifier_two = tf.keras.layers.BatchNormalization(name = "Batch_II")(classifier_two)
        """
        classifier_output = tf.keras.layers.Dense(2, activation="softmax", name = "Classifier_III")(concat)
        
        
        Full_model = tf.keras.models.Model([CNN_I_Inputs, RNN_input], classifier_output)
        
        return Full_model
    
    elif name == "Full_model_II":
        
        concat = tf.keras.layers.BatchNormalization(name = "Batch_Full")(concar_3)
        """
        classifier_Input = tf.keras.layers.Dense(8, kernel_regularizer = tf.keras.regularizers.l2(0.001), name = "CLassifier_Dense_I")(concat)
        classifier_one = tf.keras.activations.relu(classifier_Input)
        classifier_one = tf.keras.layers.Dropout(0.2, name = "Dropout_Classifier_I")(classifier_one)
        classifier_one = tf.keras.layers.BatchNormalization(name = "Batch_I")(classifier_one)
        
        classifier_two = tf.keras.layers.Dense(16, kernel_regularizer = tf.keras.regularizers.l2(0.001), name = "CLassifier_Dense_II")(classifier_one)
        classifier_two = tf.keras.activations.relu(classifier_two)
        classifier_two = tf.keras.layers.Dropout(0.2, name = "Dropout_Classifier_II")(classifier_two)
        classifier_two = tf.keras.layers.BatchNormalization(name = "Batch_II")(classifier_two)
        """
        classifier_output = tf.keras.layers.Dense(2, activation="softmax", name = "Classifier_III")(concat)
        
        
        Full_model = tf.keras.models.Model([CNN_II_Inputs, RNN_input], classifier_output)
        
        return Full_model

def mlpClassifier():
    
    neuron_1, neuron_2 = np.random.randint(1, 100), np.random.randint(1, 100)
    
    drop_out_1 = np.round(np.random.random(), 2) 
    
    drop_out_2 = np.round(np.random.random(), 2)
    
    input_layer = tf.keras.layers.Input(shape = (8,), name = "Input_Layer")
    first_layer = tf.keras.layers.Dense(neuron_1,  name = "Class_I")(input_layer)
    first_layer = tf.keras.layers.Dropout(drop_out_1)(first_layer)
    first_layer = tf.keras.activations.relu(first_layer)
    second_layer = tf.keras.layers.Dense(neuron_2,  name = "Class_I")(input_layer)
    second_layer = tf.keras.layers.Dropout(drop_out_2)(second_layer)
    second_layer = tf.keras.activations.relu(second_layer)
    third_layer = tf.keras.layers.Dense(2, activation = 'softmax', name = 'output_layer')(second_layer)
    
    mlp_model = tf.keras.models.Model(input_layer, third_layer)
    
    return mlp_model

def model_definer(model_policy, transfer_learning):
    
    if model_policy == 0:
        model_name = 'RF'
        model = ML_models(model_name, transfer_learning)
        model_type = "Table_model"
        #print('ML model: RF')
        #print('------------------------')
        
    elif model_policy == 1:
        model_name = 'SVM'
        model = ML_models(model_name, transfer_learning)
        model_type = "Table_model"
        #print('ML model: SVM')
        #print('------------------------')
        
    elif model_policy == 2:
        model_name = 'ANN'
        model = ML_models(model_name, transfer_learning)

        model.compile(optimizer = Model_optimizer,
                           loss = Loss_function,
                           metrics = "accuracy")
        
        model_type = "Table_model"
        #print('ML model: ANN')
        #print('------------------------')
        
    elif model_policy == 3:
        model_name = 'LR'
        model = ML_models(model_name, transfer_learning)
        model_type = "Table_model"
        #print('ML model: LR')
        #print('------------------------')
        
    elif model_policy == 4:
        model_name = 'NB'
        model = ML_models(model_name, transfer_learning)
        model_type = "Table_model"
        #print('ML model: NB')
        #print('------------------------')
        
    elif model_policy == 5:
        
        model_name = "CNN_I_submodel" 
        model = ML_models(model_name, transfer_learning)
        
        model.compile(optimizer = Model_optimizer,
                           loss = Loss_function,
                           metrics = "accuracy")
        
        model_type = "CNN"
        #print('ML model: CNN model Flank View')
        #print('------------------------------')
        
    elif model_policy == 6:
        
        model_name = "CNN_II_submodel" 
        model = ML_models(model_name, transfer_learning)
        
        model.compile(optimizer = Model_optimizer,
                           loss = Loss_function,
                           metrics = "accuracy")
        
        model_type = "CNN"
        #print('ML model: CNN model Rake View')
        #print('------------------------------')
        
    elif model_policy == 7:
        
        model_name = "CNN_I_II_submodel" 
        model = ML_models(model_name, transfer_learning)
        
        model.compile(optimizer = Model_optimizer,
                           loss = Loss_function,
                           metrics = "accuracy")
        
        model_type = "CNN"
        #print('ML model: CNN model flank and rake view')
        #print('------------------------------')
        
        
    elif model_policy == 8:
        
        model_name = "RNN_submodel" 
        model = ML_models(model_name, transfer_learning)
        
        model.compile(optimizer = Model_optimizer,
                           loss = Loss_function,
                           metrics = "accuracy")
        
        model_type = "LSTM"
        #print('ML model: LSTM model')
        #print('------------------------------')
        
    elif model_policy == 9:
        
        model_name = "Full_model" 
        model = ML_models(model_name, transfer_learning)
        
        model.compile(optimizer = Model_optimizer,
                           loss = Loss_function,
                           metrics = "accuracy")
        
        model_type = "Multi_model"
        #print('ML model: Full Model - Flank_Rake_Time Series data')
        #print('------------------------------')
        
    elif model_policy == 10:
        
        model_name = "Full_model_I" 
        model = ML_models(model_name, transfer_learning)
        
        model.compile(optimizer = Model_optimizer,
                           loss = Loss_function,
                           metrics = "accuracy")
        
        model_type = "Multi_model"
        #print('ML model: Full Model - Flank_Time Series data')
        #print('------------------------------')
        
    elif model_policy == 11:
        
        model_name = "Full_model_II" 
        model = ML_models(model_name, transfer_learning)
        
        model.compile(optimizer = Model_optimizer,
                           loss = Loss_function,
                           metrics = "accuracy")
        
        model_type = "Multi_model"
        #print('ML model: Full Model - Rake_Time Series data')
        #print('------------------------------')
        
    return model, model_type, model_name
        
        
def ML_models(name, transfer_learning):
    
    if name == 'RF':
        
        ml_model = RandomForestClassifier(n_estimators = 10000, criterion='entropy') 
    elif name == 'SVM':
       
        ml_model = SVC(kernel='linear', max_iter = 2000)
    elif name == 'LR':
       
        ml_model = LogisticRegression(random_state = 42, solver = "newton-cg", max_iter = 1000)
    elif name == 'ANN':
        
        #ml_model = MLPClassifier(solver='lbfgs', alpha = lr, hidden_layer_sizes=(10, 2), random_state=1)
        ml_model = mlpClassifier()
        
    elif name == 'NB':
        
        ml_model = GaussianNB()
        
    elif name =='CNN_I_submodel':
        
        ml_model = design_deep_neural_net(name, transfer_learning)
        
    elif name =='CNN_II_submodel':
        
        ml_model = design_deep_neural_net(name, transfer_learning)
        
    elif name =='CNN_I_II_submodel':
        
        ml_model = design_deep_neural_net(name, transfer_learning)
   
    elif name =='RNN_submodel':
        
        ml_model = design_deep_neural_net(name, transfer_learning)
        
    elif name =='Full_model':
        
        ml_model = design_deep_neural_net(name, transfer_learning)
        
    elif name =='Full_model_I':
        
        ml_model = design_deep_neural_net(name, transfer_learning)
        
    elif name =='Full_model_II':
        
        ml_model = design_deep_neural_net(name, transfer_learning)
        
    return ml_model


"""
transfer_learning = True

model, model_type, model_name =  model_definer(11, transfer_learning)

print(model_name)

"""