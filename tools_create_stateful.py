import numpy as np
import time
import datetime
import os
import pickle
import random
import h5py

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Add, ReLU, LSTM, ConvLSTM2D
from tensorflow.keras.layers import Conv2D, Conv3D, MaxPooling2D, concatenate, Input, AveragePooling2D, TimeDistributed
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.applications import EfficientNetB0, VGG16

import matplotlib.pyplot as plt

from config import *

# we needed a `non-stateful' version of the model, which allows resetting of RNN states in between batches
# but when we run at test time, we want a `stateful' version (RNN states are not reset between forward passes)
# seems the only way to achieve this in keras is to create a twin version of the model and copy weights across
# yes, I regret using keras for this project, yes, I will use pytorch in future projects



# save_dir = 'F:/02_saved_models_hdd'
# save_dir = 'C:/Users/Tim Pearce/Google Drive/Google Drive all/05. misc geeky/03_csgo_bot/02_remotemodels' # load model from here
save_dir = 'H:/My Drive/Google Drive all/05. misc geeky/03_csgo_bot/02_remotemodels' # load model from here
save_dir = 'F:/2021/01_remotemodels_overflow' # load model from here
save_dir_overflow = 'F:/2021/01_remotemodels_overflow' # save stateful model to here

# list of model names to convert to stateful version
model_names = ['July_remoterun7_g9_4k_n32_recipe_ton96__f16']
# model_names = ['ak47_only_55k_b4_dmexpert_8']
# model_names += ['ak47_only_55k_b4_dmexpert_12']
# model_names += ['ak47_only_55k_b4_dmexpert_16']


for weights_name in model_names:

    def stateful_create(model, model_name):
        # after training the model defined above, we create a twin, but with
        # stateful=True,return_sequences=False,unroll=True
        # which makes for much faster inference time when running
        # seems ridiculous, but stackoverflow suggests is only way
        
        # create same model, but with stateful=True and different shape

        if 'lowres' not in model_name:
            input_shape_batch = (1, 1,csgo_img_dimension[0],csgo_img_dimension[1],3)
            base_model = EfficientNetB0(weights='imagenet',input_shape=(input_shape[1:]),include_top=False,drop_connect_rate=0.2)

            intermediate_model= Model(inputs=base_model.input, outputs=base_model.layers[161].output)

            input_1 = Input(shape=input_shape_batch,name='main_in')
            x = TimeDistributed(intermediate_model)(input_1)
        else:
            input_shape_batch = (1, 1,csgo_img_dimension[0],csgo_img_dimension[1],3)
            input_shape_lowres = (N_TIMESTEPS,100,187,3)
            base_model = EfficientNetB0(weights='imagenet',input_shape=(input_shape_lowres[1:]),include_top=False,drop_connect_rate=0.2)
            intermediate_model= Model(inputs=base_model.input, outputs=base_model.layers[161].output)

            # input_1 = Input(shape=input_shape_batch,name='main_in',batch_shape=1)
            input_1 = Input(batch_input_shape=input_shape_batch,name='main_in')
            x = TimeDistributed(tf.keras.layers.experimental.preprocessing.Resizing(input_shape_lowres[1], input_shape_lowres[2], interpolation="bilinear"))(input_1)
            x = TimeDistributed(intermediate_model)(x)

        if 'drop' in weights_name:
            # x = TimeDistributed(Dropout(0.5))(x)
            x = ConvLSTM2D(filters=256,kernel_size=(3,3),stateful=True,return_sequences=True,dropout=0.5, recurrent_dropout=0.5)(x)
        else:
            x = ConvLSTM2D(filters=256,kernel_size=(3,3),stateful=True,return_sequences=True)(x)
        if 'extra' in weights_name:
            if 'drop' in weights_name:
                x = ConvLSTM2D(filters=256,kernel_size=(3,3),stateful=True,return_sequences=True,dropout=0.5, recurrent_dropout=0.5)(x)
            else:
                x = ConvLSTM2D(filters=256,kernel_size=(3,3),stateful=True,return_sequences=True)(x)
        x = TimeDistributed(Flatten())(x)
        if 'LSTM' in weights_name:
            if 'drop' in weights_name:
                x = TimeDistributed(Dropout(0.5))(x)
                x = LSTM(256,stateful=True,return_sequences=True,dropout=0., recurrent_dropout=0.)(x)
                x = TimeDistributed(Dropout(0.5))(x)
            else:
                x = LSTM(256,stateful=True,return_sequences=True)(x)

        # need return_sequences for convlstm2d otherwise squashes first dim
        dense_5 = x
        output_1 = Dense(n_keys, activation='sigmoid')(dense_5)
        output_2 = Dense(n_clicks, activation='sigmoid')(dense_5)
        output_3 = Dense(n_mouse_x, activation='softmax')(dense_5) # softmax since mouse is mutually exclusive
        output_4 = Dense(n_mouse_y, activation='softmax')(dense_5) 
        output_5 = Dense(1, activation='linear')(dense_5)
        output_all = concatenate([output_1,output_2,output_3,output_4,output_5], axis=-1)
        model_stateful = Model(input_1, output_all)
        model_stateful.compile(loss=custom_loss,optimizer=opt, metrics=[Lclk_acc,m_x_acc,m_y_acc,wasd_acc,j_acc,Rclk_acc])
        
        # copy weights
        for nb, layer in enumerate(model.layers):
            print(nb, layer.name)
            model_stateful.layers[nb].set_weights(layer.get_weights())
        tp_save_model(model_stateful, save_dir_overflow, model_name+'_stateful')

        return



    start_time=time.time()

    # pick up training from earlier spot
    print('-- loading model from saved file --')
    model = tp_load_model(save_dir, weights_name)
    # TODO: if .p file exists, load and check if matches current config
    hypers_load = pickle.load(open(save_dir+'/'+weights_name+'.p', 'rb'))
    print(hypers_load)


    print(model.summary())

    # loss to minimise
    def custom_loss(y_true, y_pred):

        # wasd keys
        loss1a = losses.binary_crossentropy(y_true[:,:,0:4], 
                                            y_pred[:,:,0:4])
        # space key
        loss1b = losses.binary_crossentropy(y_true[:,:,4:5], 
                                            y_pred[:,:,4:5])
        # reload key
        loss1c = losses.binary_crossentropy(y_true[:,:,n_keys-1:n_keys], 
                                            y_pred[:,:,n_keys-1:n_keys])
        # all other keys
        # loss1d = losses.binary_crossentropy(y_true[:,:,5:n_keys-1], 
        #                                     y_pred[:,:,5:n_keys-1])
        # left click
        loss2a = losses.binary_crossentropy(y_true[:,:,n_keys:n_keys+1], 
                                            y_pred[:,:,n_keys:n_keys+1])
        # right click
        # loss2b = losses.binary_crossentropy(y_true[:,:,n_keys+1:n_keys+n_clicks], 
        #                                     y_pred[:,:,n_keys+1:n_keys+n_clicks])
        # mouse move x
        loss3 = losses.categorical_crossentropy(y_true[:,:,n_keys+n_clicks:n_keys+n_clicks+n_mouse_x], 
                                                y_pred[:,:,n_keys+n_clicks:n_keys+n_clicks+n_mouse_x])
        # mouse move y
        loss4 = losses.categorical_crossentropy(y_true[:,:,n_keys+n_clicks+n_mouse_x:n_keys+n_clicks+n_mouse_x+n_mouse_y], 
                                                y_pred[:,:,n_keys+n_clicks+n_mouse_x:n_keys+n_clicks+n_mouse_x+n_mouse_y])

        # critic loss -- measuring between consecutive time steps
        #  = ((reward_t + gamma  v_t+1) - v_t)^2
        # can't really decide whether to use reward_t or reward_t+1, but guess it doesn't matter too much
        loss_crit = 100*losses.MSE(y_true[:,:-1,n_keys+n_clicks+n_mouse_x+n_mouse_y:n_keys+n_clicks+n_mouse_x+n_mouse_y+1]
                           + GAMMA*y_pred[:,1:,n_keys+n_clicks+n_mouse_x+n_mouse_y:n_keys+n_clicks+n_mouse_x+n_mouse_y+1]
                           ,y_pred[:,:-1,n_keys+n_clicks+n_mouse_x+n_mouse_y:n_keys+n_clicks+n_mouse_x+n_mouse_y+1])

        return K.concatenate([loss1a, loss1b, loss1c, loss2a, loss3, loss4, loss_crit])




    # metrics for each part of interest - useful for debugging
    def wasd_acc(y_true, y_pred):
        return keras.metrics.binary_accuracy(y_true[:,:,0:4], y_pred[:,:,0:4])

    def j_acc(y_true, y_pred): # other keys, space, ctrl, shift, 1,2,3, r
        return keras.metrics.binary_accuracy(y_true[:,:,4:5], y_pred[:,:,4:5])

    def oth_keys_acc(y_true, y_pred): # other keys, space, ctrl, shift, 1,2,3, r
        return keras.metrics.binary_accuracy(y_true[:,:,5:n_keys], y_pred[:,:,5:n_keys])

    def Lclk_acc(y_true, y_pred):
        return keras.metrics.binary_accuracy(y_true[:,:,n_keys:n_keys+1], y_pred[:,:,n_keys:n_keys+1],threshold=0.5)
        # relative to proportion that don't fire 
        # return keras.metrics.binary_accuracy(y_true[:,n_keys:n_keys+1], y_pred[:,n_keys:n_keys+1],threshold=0.5) - (1 - keras.backend.mean(keras.backend.greater(y_true[:,n_keys:n_keys+1], 0.5)))

    def Rclk_acc(y_true, y_pred):
        return keras.metrics.binary_accuracy(y_true[:,:,n_keys+1:n_keys+n_clicks], y_pred[:,:,n_keys+1:n_keys+n_clicks],threshold=0.5)

    def m_x_acc(y_true, y_pred):
        return keras.metrics.categorical_accuracy(y_true[:,:,n_keys+n_clicks:n_keys+n_clicks+n_mouse_x], 
                                                  y_pred[:,:,n_keys+n_clicks:n_keys+n_clicks+n_mouse_x])
    def m_y_acc(y_true, y_pred):
        return keras.metrics.categorical_accuracy(y_true[:,:,n_keys+n_clicks+n_mouse_x:n_keys+n_clicks+n_mouse_x+n_mouse_y], 
                                                  y_pred[:,:,n_keys+n_clicks+n_mouse_x:n_keys+n_clicks+n_mouse_x+n_mouse_y])

    def crit_mse(y_true, y_pred):
        return 10*losses.MSE(y_true[:,:-1,n_keys+n_clicks+n_mouse_x+n_mouse_y:n_keys+n_clicks+n_mouse_x+n_mouse_y+1]
                                               + GAMMA*y_pred[:,1:,n_keys+n_clicks+n_mouse_x+n_mouse_y:n_keys+n_clicks+n_mouse_x+n_mouse_y+1]
                                               ,y_pred[:,:-1,n_keys+n_clicks+n_mouse_x+n_mouse_y:n_keys+n_clicks+n_mouse_x+n_mouse_y+1])

    def w_1(y_true, y_pred):
        return keras.backend.mean(keras.backend.greater(y_true[:,:,0], 0.5))

    def no_fire(y_true, y_pred):
        return 1 - keras.backend.mean(keras.backend.greater(y_true[:,:,n_keys:n_keys+1], 0.5))

    def m_x_0(y_true, y_pred):
        return keras.backend.mean(keras.backend.greater(y_true[:,:,n_keys+n_clicks+int(np.floor(n_mouse_x/2))], 0.5))

    def m_y_0(y_true, y_pred):
        return keras.backend.mean(keras.backend.greater(y_true[:,:,n_keys+n_clicks+n_mouse_x+int(np.floor(n_mouse_y/2))], 0.5))


    opt = optimizers.Adam(learning_rate=0.0001)
    model.compile(loss=custom_loss,optimizer=opt, metrics=[Lclk_acc,no_fire,m_x_acc,m_y_acc,wasd_acc])
    print('successfully compiled model')

    stateful_create(model, weights_name)




