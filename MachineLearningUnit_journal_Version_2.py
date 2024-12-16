"""
Automated ML - Porject II
ML model Unit

Program Developer: Saleh ValizadehSotubadi
Advisor: Dr. Vinh Nguyen

Department of Mechanical Engineering-Engineering Mechanics (MEEM)
Michigan Technological University

Date: 01/09/2023

"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR) # or logging.INFO, logging.WARNING, etc.

import numpy as np
from sklearn.metrics import accuracy_score
import pickle
import gc
#from pickle_data import load_save_data
from machinelearning_models_ import image_data, time_series_data
from feature_engineering import feature_extraction
#from Preprocessing_section import preprocess_data
#from data_prepration_module import data_preparation_method_I, data_preparation_method_II
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, roc_curve, matthews_corrcoef
#from Data_Load import generate_pickle_data
from hyper_parameter_tuning import set_hyperparameters, set_reward
from Data_Load_Flash import generate_pickle_data
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight


def cost_function(accuracy, F1):
    
    accuracy -= 0.75
        
    accuracy = 3/(1 + np.exp(-accuracy))
        
    accuracy -= 1.5
    
    F1 -= 0.70
    
    F1 = 4/(1 + np.exp(-F1))
    
    F1 -= 2.0
        
    return accuracy  + F1

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return TP, FP, TN, FN

class AutoML_Network:
    
    def __init__(self, policy, learning_rate, beta_1, beta_2, epochs, batch, dropout, l2, k_fold, neurons, layers):
        
        self.policy = policy
        
        self.neurons = int(neurons)
        
        self.epochs = int(epochs)
        
        self.batch = int(batch)
        
        self.layers = int(layers)
        
        self.dropout = dropout
        
        self.k_fold = int(k_fold)
        
        self.l2 = l2
        
        self.trainable = False
        
        self.loss_function = tf.keras.losses.BinaryCrossentropy()
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate,
                                                  beta_1=beta_1,
                                                  beta_2 = beta_2)
        
        self.lr_schedulor = tf.keras.callbacks.ReduceLROnPlateau(monitor = "val_accuracy", 
                                                            factor = 0.995,
                                                            patience = 2,
                                                            min_lr = 0.0001)

        self.early_schedulor = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy",
                                                           min_delta = 0,
                                                           patience = 4
                                                           )
        
        self.checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath='/home/svalizad/Desktop/ADP/ADP_Journal Paper/best_model.keras',  # Path to save the model
                                                                 save_best_only=True,  # Saves only the best model
                                                                 monitor='val_accuracy',  # Monitors validation accuracy
                                                                 mode='max',  # Mode for monitoring
                                                                 verbose=0)

    def Network(self):
            
        data_dir = '/content/gdrive/MyDrive/ADP_Journal Paper'
    
            #file_lists = os.listdir(data_dir)
                
        """
            Uncomment this and comment the following 2 lines if the Datasets.pickle is not in the directory
            
            if 'Datasets.pickle' not in os.listdir(data_dir):
                    
                self.datasets = generate_pickle_data()
                    
            else:
                
                with open(os.path.join(data_dir, 'Datasets.pickle'), 'rb') as file:
                    
                    self.datasets = pickle.load(file)
        """
        with open(os.path.join(data_dir, 'Datasets.pickle'), 'rb') as file:
                
            self.datasets = pickle.load(file)
                    
        os.chdir('/content/gdrive/MyDrive/ADP_Journal Paper')

        indexes = np.random.permutation(len(self.datasets['labels']))

        self.datasets['flank_images'], self.datasets['rake_images'] = self.datasets['flank_images'][indexes], self.datasets['rake_images'][indexes]

        self.datasets['time_series_data'], self.datasets['labels'] = self.datasets['time_series_data'][indexes], self.datasets['labels'][indexes]
                        
        image_policy, time_series_policy = self.policy // 6, np.mod(self.policy, 6)
            
        if time_series_policy > 1:
                
            self.datasets['time_series_data'] = feature_extraction(self.datasets['time_series_data'], time_series_policy)
            
        self.img_feature_I, self.img_feature_II = image_data(image_policy)
            
        self.time_series_feature = time_series_data(time_series_policy, self.dropout, self.l2, self.layers)
            
        if image_policy < 6:
                
            self.img_feature_I.trainable = self.trainable
                
            self.img_feature_II.trainable = self.trainable
                
            
        acc, f1 = self.model_design()
            
        del self.datasets
            
        gc.collect()
            
        return acc, f1
        
    def perf_measure(y_actual, y_hat):
        TP = 0
        FP = 0
        TN = 0
        FN = 0

        for i in range(len(y_hat)): 
            if y_actual[i]==y_hat[i]==1:
               TP += 1
            if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
               FP += 1
            if y_actual[i]==y_hat[i]==0:
               TN += 1
            if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
               FN += 1

        return TP, FP, TN, FN


    def MCC(self, TP, FP, TN, FN):
        
        
        try:
            mcc = ((TP*TN) - (FN * FP))/np.sqrt((TP + FN)*(TP + FP)*(TN + FN)*(TN + FP))
        except:
            mcc = 0
        
        return mcc
    
    def F1_score(self, TP, FP, FN):
        
        try:
        
            F1 = TP/(TP + 0.5*(FP + FN))
        except:
            F1 = 0.5
        
        return F1
    
    def model_design(self):
        
        
        img_flank_normalized = tf.keras.layers.LayerNormalization(epsilon = 1e-6)(self.img_feature_I.output)
        img_flank_flatten = tf.keras.layers.Flatten(name = "Flatten_flank")(img_flank_normalized)
        class_flank = tf.keras.layers.Dense(self.neurons , kernel_regularizer = tf.keras.regularizers.l2(self.l2),  name = "Class_flank")(img_flank_flatten)
        class_flank = tf.keras.activations.relu(class_flank)
        class_flank = tf.keras.layers.Dropout(self.dropout, name = "Dropout_flank")(class_flank)
        
        img_rake_normalized = tf.keras.layers.LayerNormalization(epsilon = 1e-6)(self.img_feature_II.output)
        img_rake_flatten = tf.keras.layers.Flatten(name = "Flatten_rake")(img_rake_normalized)
        class_rake = tf.keras.layers.Dense(self.neurons , kernel_regularizer = tf.keras.regularizers.l2(self.l2),  name = "Class_rake")(img_rake_flatten)
        class_rake = tf.keras.activations.relu(class_rake)
        class_rake = tf.keras.layers.Dropout(self.dropout, name = "Dropout_rake")(class_rake)
        
        time_series_normalized = tf.keras.layers.LayerNormalization(epsilon = 1e-6)(self.time_series_feature.output)
        time_series_flatten = tf.keras.layers.Flatten(name = "Flatten_time")(time_series_normalized)
        class_time = tf.keras.layers.Dense(self.neurons , kernel_regularizer = tf.keras.regularizers.l2(self.l2),  name = "Class_time")(time_series_flatten)
        class_time = tf.keras.activations.relu(class_time)
        class_time = tf.keras.layers.Dropout(self.dropout, name = "Dropout_time")(class_time)
        
        concat = tf.keras.layers.concatenate([class_flank, class_rake, class_time], name = "Concat")
        
        concat = tf.keras.layers.BatchNormalization(name = "Batch_Full")(concat)
        
        classifier_output = tf.keras.layers.Dense(2, activation="softmax", name = "Classifier_III")(concat)
        
        Full_model = tf.keras.models.Model([self.img_feature_I.input, self.img_feature_II.input, self.time_series_feature.input], classifier_output)
    
        acc, f1 = self.train_model(Full_model)
        
        return acc, f1
        
    def augment_image(self,image):
    
        datagen = ImageDataGenerator(
            rotation_range=5,  # rotate images randomly by 10 degrees
            width_shift_range=0.1,  # shift images horizontally by 10% of the total width
            height_shift_range=0.1,  # shift images vertically by 10% of the total height
            shear_range=0.2,  # shear transformation with a shear intensity of 0.2
            zoom_range=0.1,  # zoom images by up to 20%
            horizontal_flip=False,  # randomly flip images horizontally
            vertical_flip=False,  # don't flip images vertically
            fill_mode='nearest'  # strategy to fill in newly created pixels
        )
            
        batch_size = self.batch
        
        augmented_data_generator = datagen.flow(image, batch_size=batch_size)
        
        num_batches = len(image) // batch_size
        
        num_samples = len(image)
        
        augmented_data = []
        for _ in range(num_batches):
            batch = augmented_data_generator.__next__()
            augmented_data.append(batch)
                
        if num_samples % batch_size != 0:
            remaining_samples = num_samples % batch_size
            batch = augmented_data_generator.__next__()
            augmented_data.append(batch[:remaining_samples])
        
        X_train_augment = np.concatenate(augmented_data, axis=0)
        
        return X_train_augment
    
    def train_model(self, model):
        
        train_ratio = 0.92
        
        train_num = int(len(self.datasets['labels']) * train_ratio)
        
        image_flank_train, image_rake_train, time_series_train = self.datasets['flank_images'][:train_num], self.datasets['rake_images'][:train_num], self.datasets['time_series_data'][:train_num]
        
        train_label_train = self.datasets['labels'][:train_num]
        
        image_flank_test, image_rake_test, time_series_test = self.datasets['flank_images'][train_num:], self.datasets['rake_images'][train_num:], self.datasets['time_series_data'][train_num:]
        
        label_test = self.datasets['labels'][train_num:]
        
        model.compile(loss = self.loss_function,
                      optimizer = self.optimizer,
                      metrics = ["accuracy"])
        
        k_folds = self.k_fold 
        
        kf = KFold(n_splits=k_folds, shuffle=True)
        
        for train_index, test_index in kf.split(image_flank_train):
            
            Train_flank_image_I, Train_rake_image_I, Train_time_series_I = image_flank_train[train_index], image_rake_train[train_index], time_series_train[train_index]
            
            Train_Label_I = train_label_train[train_index]
            
            #Train_flank_image_I, Train_rake_image_I = self.augment_image(Train_flank_image_I), self.augment_image(Train_rake_image_I)
            
            Train_Label_I_cat = tf.keras.utils.to_categorical(Train_Label_I)
            
            Val_flank_image, Val_rake_image, Val_time_series = image_flank_train[test_index], image_rake_train[test_index], time_series_train[test_index]
            
            Val_flank_image, Val_rake_image = self.augment_image(Val_flank_image), self.augment_image(Val_rake_image)
            
            Val_Label = train_label_train[test_index]
            
            Val_Label_cat = tf.keras.utils.to_categorical(Val_Label)
            
            class_labels = np.unique(Train_Label_I)
            class_weights = compute_class_weight('balanced', classes=class_labels, y=Train_Label_I)
            class_weight_dict = dict(enumerate(class_weights))

            

            try:
                history = model.fit([Train_flank_image_I, Train_rake_image_I, Train_time_series_I], Train_Label_I_cat,
                                         epochs = self.epochs,
                                         batch_size = self.batch,
                                         validation_data=([Val_flank_image, Val_rake_image, Val_time_series], Val_Label_cat),
                                         verbose=0,
                                         callbacks=[self.lr_schedulor, self.early_schedulor])
                
                stat_train = 1
            
            except:
                
                stat_train = 0
                
        if  stat_train == 1:
        
            predictions_full = model.predict([image_flank_test, image_rake_test, time_series_test])
            
            prediction_full_label = predictions_full.argmax(axis=-1)
            
            TP, FP, TN, FN = perf_measure(label_test, prediction_full_label)
            
            acc = (TP + TN)/(TP + FP + TN + FN)
            
            #mcc = self.MCC(TP, FP, TN, FN)
            
            F1 = self.F1_score(TP, FP, FN)
            
        else:
                
            acc, F1 = 0, 0
            
            
        del model, history
        
        del image_flank_test, image_rake_test, time_series_test, Train_flank_image_I, Train_rake_image_I, Train_time_series_I, Val_flank_image, Val_rake_image, Val_time_series, Val_Label_cat, Val_Label
        
        del Train_Label_I_cat, Train_Label_I, predictions_full, prediction_full_label, label_test, TP, FP, TN, FN
        
        gc.collect()
        
        return acc, F1

def AutoML_Model(policy):
    
    hyper_parameters = set_hyperparameters(policy)
    
    learning_rate, beta_1, beta_2, epochs, batch = hyper_parameters[0], hyper_parameters[2], hyper_parameters[3], 30, int(hyper_parameters[1])
    
    dropout, l2 = hyper_parameters[4], hyper_parameters[5]
    
    k_fold = hyper_parameters[6]
    
    neurons = hyper_parameters[7]
    
    layers = hyper_parameters[8]
    
    model = AutoML_Network(policy, learning_rate, beta_1, beta_2, epochs, batch, dropout, l2, k_fold, neurons, layers)
    
    acc, F1 = model.Network()
    
    reward = cost_function(acc, F1)
    
    set_reward(policy, reward)
    
    print(reward)
    
    del model, learning_rate, beta_1, beta_2, epochs, batch
    
    gc.collect()
    
    return reward