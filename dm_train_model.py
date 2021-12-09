import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" # single GPU
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3" # use multiple GPUs

import tensorflow as tf

strategy = tf.distribute.MirroredStrategy(["GPU:0"])
# strategy = tf.distribute.MirroredStrategy(["GPU:0","GPU:1","GPU:2", "GPU:3"])
print('\nnumber of devices using for training: {}'.format(strategy.num_replicas_in_sync))

import numpy as np
import time
import datetime
import pickle
import random
import h5py

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Add, ReLU, LSTM, ConvLSTM2D
from tensorflow.keras.layers import Conv2D, Conv3D, MaxPooling2D, concatenate, Input, AveragePooling2D, TimeDistributed, Dropout
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.applications import EfficientNetB0

from config import *

# inputs
batch_size = 1 # this is total batchsize using all GPUs, so make divisible by num_gpus!
l_rate = 0.0001

# training data location
file_name_stub = 'dm_july2021_expert_' # dm_july2021_ aim_july2021_expert_ dm_july2021_expert_
# file_name_stub = 'dm_6nov_aim_' 
folder_name = '/mfs/TimPearce/01_csgo/01_trainingdata/' 
starting_num = 1 # lowest file name to use in training
highest_num = 30 # highest file name to use in training 4000, 5500, 190, 45, 10

# whether to save model if training and where
model_name = 'ak47_m41a_55k_sub_drop_'
save_dir = '/mfs/TimPearce/01_csgo/02_savedmodels'
SAVE_MODEL = True

# whether to resume training from a previous model
IS_LOAD_WEIGHTS_AND_MODEL=False
weights_name = 'test_model_1'

# which subselection of dataset to train on
IS_SUBSELECT = False
SUB_PROB = 0.4
SUB_TYPE = 'ak' # ak or akm4 or all
OVERSAMPLE_LOWFREQ_REGION=False

# where are the metadata .npy files? only needed if subselecting
curr_vars_folder = '/mfs/TimPearce/01_csgo/03_currvars/'
if file_name_stub == 'dm_july2021_expert_':
    curr_vars_stub = 'currvarsv2_dm_july2021_expert_'
else:
    curr_vars_stub = 'currvarsv2_dm_july2021_'


start_time=time.time()
with strategy.scope():

    if IS_LOAD_WEIGHTS_AND_MODEL:
        # pick up training from earlier spot
        print('-- loading model from saved file --')
        model = tp_load_model(save_dir, weights_name)
        # TODO: if .p file exists, load and check if matches current config
        hypers_load = pickle.load(open(save_dir+'/'+weights_name+'.p', 'rb'))
        print(hypers_load)

    else:
        # useful tutorial for building, https://keras.io/getting-started/functional-api-guide/
        print('-- building model from scratch --')

        base_model = EfficientNetB0(weights='imagenet',input_shape=(input_shape[1:]),include_top=False,drop_connect_rate=0.2)
        if 'randinit' in model_name:
            print('random initialisation!\n\n')
            base_model = EfficientNetB0(weights=None,input_shape=(input_shape[1:]),include_top=False,drop_connect_rate=0.2)
        base_model.trainable = True

        intermediate_model= Model(inputs=base_model.input, outputs=base_model.layers[161].output)
        intermediate_model.trainable = True

        input_1 = Input(shape=input_shape,name='main_in')
        x = TimeDistributed(intermediate_model)(input_1)


        if 'drop' in model_name:
            if 'big' in model_name:
                x = ConvLSTM2D(filters=512,kernel_size=(3,3),stateful=False,return_sequences=True,dropout=0.5, recurrent_dropout=0.5)(x)
            else:
                x = ConvLSTM2D(filters=256,kernel_size=(3,3),stateful=False,return_sequences=True,dropout=0.5, recurrent_dropout=0.5)(x)
        else:
            if 'big' in model_name:
                x = ConvLSTM2D(filters=512,kernel_size=(3,3),stateful=False,return_sequences=True)(x)
            else:
                x = ConvLSTM2D(filters=256,kernel_size=(3,3),stateful=False,return_sequences=True)(x)
        if 'extra' in model_name:
            if 'drop' in model_name:
                x = ConvLSTM2D(filters=256,kernel_size=(3,3),stateful=False,return_sequences=True,dropout=0.5, recurrent_dropout=0.5)(x)
            else:
                x = ConvLSTM2D(filters=256,kernel_size=(3,3),stateful=False,return_sequences=True)(x)

        x = TimeDistributed(Flatten())(x)
        if 'LSTM' in model_name:
            if 'drop' in model_name:
                x = TimeDistributed(Dropout(0.5))(x)
                x = LSTM(256,stateful=False,return_sequences=True,dropout=0., recurrent_dropout=0.)(x)
                x = TimeDistributed(Dropout(0.5))(x)
            else:
                x = LSTM(256,stateful=False,return_sequences=True)(x)


        # 2) set up auxillary input,  which can have previous actions, as well as aux info like health, ammo, team
        aux_input = Input(shape=(int(ACTIONS_PREV*(aux_input_length))),name='aux_in')
        if AUX_INPUT_ON:
            flat = concatenate([flat, aux_input], axis=1)
        else:
            pass

        # 3) add shared fc layers
        dense_5 = x

        # 4) set up outputs, sepearate outputs will allow seperate losses to be applied
        output_1 = TimeDistributed(Dense(n_keys, activation='sigmoid'))(dense_5)
        output_2 = TimeDistributed(Dense(n_clicks, activation='sigmoid'))(dense_5)
        output_3 = TimeDistributed(Dense(n_mouse_x, activation='softmax'))(dense_5) # softmax since mouse is mutually exclusive
        output_4 = TimeDistributed(Dense(n_mouse_y, activation='softmax'))(dense_5) 
        output_5 = TimeDistributed(Dense(1, activation='linear'))(dense_5) 
        # output_all = concatenate([output_1,output_2,output_3,output_4], axis=-1)
        output_all = concatenate([output_1,output_2,output_3,output_4,output_5], axis=-1)


        # 5) finish model definition
        if AUX_INPUT_ON:
            model = Model([input_1, aux_input], output_all)
        else:
            model = Model(input_1, output_all)

        # hacky method to update trained model's value of N_TIMESTEPS -- 
        # 1 set config to new N_TIMESTEPS
        # 2 load prev model w above code snippet
        # 3 create new model but ignore above line and run below instead
        # 4 manually copy the .p and opt files under new name
        # eg cp July_remoterun7_g9_4k_n32_recipe__d12.p July_remoterun7_g9_4k_n32_recipe_ton96_.p
        # eg cp July_remoterun7_g9_4k_n32_recipe__d12_opt.pkl July_remoterun7_g9_4k_n32_recipe_ton96__opt.pkl
        if False:
            model_new = Model(input_1, output_all)
            for nb, layer in enumerate(model.layers):
                model_new.layers[nb].set_weights(layer.get_weights())
            tp_save_model(model_new, save_dir, model_name) #+'28_N96')

            # model_val = Model(input_1, output_all)
            # for i in range(0,8):
            #     model.layers[i].set_weights(model_ws.layers[i].get_weights())
            # tp_save_model(model, save_dir, model_name)

    print(model.summary())

    # loss to minimise
    def custom_loss(y_true, y_pred):
        # y_true is shape [n_batch, n_timesteps, n_keys+n_clicks+n_mouse_x+n_mouse_y+n_reward+n_advantage]
        # where n_reward and n_advantage must =1
        # y_pred is shape [n_batch, n_timesteps, n_keys+n_clicks+n_mouse_x+n_mouse_y+n_val]
        # we'll use y_true to send in reward and eventually original advantage fn (or could recompute this?)

        # wasd keys
        loss1a = losses.binary_crossentropy(y_true[:,:,0:4], 
                                            y_pred[:,:,0:4])
        # space key
        loss1b = losses.binary_crossentropy(y_true[:,:,4:5], 
                                            y_pred[:,:,4:5])
        # reload key
        loss1c = losses.binary_crossentropy(y_true[:,:,n_keys-1:n_keys], 
                                            y_pred[:,:,n_keys-1:n_keys])

        # weapon switches, 1,2,3
        loss1d = losses.binary_crossentropy(y_true[:,:,n_keys-4:n_keys-1], 
                                            y_pred[:,:,n_keys-4:n_keys-1])

        # all other keys
        # loss1d = losses.binary_crossentropy(y_true[:,:,5:n_keys-1], 
        #                                     y_pred[:,:,5:n_keys-1])
        # left click
        loss2a = losses.binary_crossentropy(y_true[:,:,n_keys:n_keys+1], 
                                            y_pred[:,:,n_keys:n_keys+1])
        # right click
        loss2b = losses.binary_crossentropy(y_true[:,:,n_keys+1:n_keys+n_clicks], 
                                            y_pred[:,:,n_keys+1:n_keys+n_clicks])
        # mouse move x
        loss3 = losses.categorical_crossentropy(y_true[:,:,n_keys+n_clicks:n_keys+n_clicks+n_mouse_x], 
                                                y_pred[:,:,n_keys+n_clicks:n_keys+n_clicks+n_mouse_x])
        # mouse move y
        loss4 = losses.categorical_crossentropy(y_true[:,:,n_keys+n_clicks+n_mouse_x:n_keys+n_clicks+n_mouse_x+n_mouse_y], 
                                                y_pred[:,:,n_keys+n_clicks+n_mouse_x:n_keys+n_clicks+n_mouse_x+n_mouse_y])

        # critic loss -- measuring between consecutive time steps
        #  = ((reward_t + gamma  v_t+1) - v_t)^2
        # can't really decide whether to use reward_t or reward_t+1, but guess it doesn't matter too much
        loss_crit = 10*losses.MSE(y_true[:,:-1,n_keys+n_clicks+n_mouse_x+n_mouse_y:n_keys+n_clicks+n_mouse_x+n_mouse_y+1]
                           + GAMMA*y_pred[:,1:,n_keys+n_clicks+n_mouse_x+n_mouse_y:n_keys+n_clicks+n_mouse_x+n_mouse_y+1]
                           ,y_pred[:,:-1,n_keys+n_clicks+n_mouse_x+n_mouse_y:n_keys+n_clicks+n_mouse_x+n_mouse_y+1])

        return K.concatenate([loss1a, loss1b, loss1c, loss2a, loss3, loss4, loss_crit])
        # return K.concatenate([loss1a, loss2a, loss3, loss4])




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
        return 100*losses.MSE(y_true[:,:-1,n_keys+n_clicks+n_mouse_x+n_mouse_y:n_keys+n_clicks+n_mouse_x+n_mouse_y+1]
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


    opt = optimizers.Adam(learning_rate=l_rate)
    # model.compile(loss=custom_loss,optimizer=opt, metrics=[Lclk_acc,no_fire,m_x_acc,m_x_0,m_y_acc,m_y_0])
    model.compile(loss=custom_loss,optimizer=opt, metrics=[Lclk_acc,no_fire,m_x_acc,m_y_acc,wasd_acc,crit_mse])
    # model.compile(loss=custom_loss,optimizer=opt, metrics=[Lclk_acc,no_fire,m_x_acc,m_y_acc,wasd_acc])
    print('successfully compiled model')


# data generator
class DataGenerator(keras.utils.Sequence):
# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    def __init__(self, list_IDs, batch_size=32, shuffle=True):
        self.list_IDs = list_IDs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end() # run this once to start

    def __len__(self):
        # the number of batches per epoch - how many times are we calling this generator altogether
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        # generate one batch of data, index is the batch number

        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        # updates indexes after each epoch
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

        # could do subsampling at this stage, by 
        # using ID in format 'filenum-framenum-containkillevent'

    def __data_generation(self, list_IDs_temp):

        # set up empty arrays to fill
        x_shape = list(input_shape)
        x_shape.insert(0,self.batch_size)
        # y_shape = [self.batch_size,n_keys+n_clicks+n_mouse_x+n_mouse_y]
        y_shape = [self.batch_size,N_TIMESTEPS,n_keys+n_clicks+n_mouse_x+n_mouse_y+1+1] # add 1 for reward, 1 for adv

        X = np.empty(x_shape)
        y = np.empty(y_shape)

        for i, ID in enumerate(list_IDs_temp): 
            # print(i,end='\r')
            # ID is in format 'filenum-framenum'
            ID = ID.split('-')
            file_num = int(ID[0])
            frame_num = int(ID[1])+np.random.randint(0,N_JITTER-1)
            frame_num = np.minimum(frame_num,999-N_TIMESTEPS)
            frame_num = np.maximum(frame_num,0)

            # quicker way reading from hdf5
            file_name = folder_name + 'hdf5_'+file_name_stub + str(file_num) + '.hdf5'
            h5file = h5py.File(file_name, 'r')

            for j in range(N_TIMESTEPS):
                X[i,j] = h5file['frame_'+str(frame_num+j)+'_x'][:] # /255
                y[i,j,:-2] = h5file['frame_'+str(frame_num+j)+'_y'][:]

                help_i = h5file['frame_'+str(frame_num+j)+'_helperarr'][:]
                kill_i = help_i[0]
                dead_i = help_i[1]
                shoot_i = y[i,j,n_keys:n_keys+1] # all these are binary variables
                reward_i = kill_i - 0.5*dead_i - 0.01*shoot_i # this is reward function
                y[i,j,-2:] = (reward_i,0.) # 0. is a placeholder for original advantage

                # for mouse, we're going to use a manual hack to remove most extreme 2 classes
                if y[i,j,n_keys+n_clicks] == 1:
                    y[i,j,n_keys+n_clicks] = 0
                    y[i,j,n_keys+n_clicks+2] = 1
                elif y[i,j,n_keys+n_clicks+1] == 1:
                    y[i,j,n_keys+n_clicks+1] = 0
                    y[i,j,n_keys+n_clicks+2] = 1
                elif y[i,j,n_keys+n_clicks+n_mouse_x-1] == 1:
                    y[i,j,n_keys+n_clicks+n_mouse_x-1] = 0
                    y[i,j,n_keys+n_clicks+n_mouse_x-3] = 1
                elif y[i,j,n_keys+n_clicks+n_mouse_x-2] == 1:
                    y[i,j,n_keys+n_clicks+n_mouse_x-2] = 0
                    y[i,j,n_keys+n_clicks+n_mouse_x-3] = 1

                # same for mouse y as of 20 aug
                if y[i,j,n_keys+n_clicks+n_mouse_x] == 1:
                    y[i,j,n_keys+n_clicks+n_mouse_x] = 0
                    y[i,j,n_keys+n_clicks+n_mouse_x+2] = 1
                elif y[i,j,n_keys+n_clicks+n_mouse_x+1] == 1:
                    y[i,j,n_keys+n_clicks+n_mouse_x+1] = 0
                    y[i,j,n_keys+n_clicks+n_mouse_x+2] = 1
                elif y[i,j,n_keys+n_clicks+n_mouse_x+n_mouse_y-1] == 1:
                    y[i,j,n_keys+n_clicks+n_mouse_x+n_mouse_y-1] = 0
                    y[i,j,n_keys+n_clicks+n_mouse_x+n_mouse_y-3] = 1
                elif y[i,j,n_keys+n_clicks+n_mouse_x+n_mouse_y-2] == 1:
                    y[i,j,n_keys+n_clicks+n_mouse_x+n_mouse_y-2] = 0
                    y[i,j,n_keys+n_clicks+n_mouse_x+n_mouse_y-3] = 1

            # add a manual hack here to make sure lclick is down  2 aug 2021
            # this is because firing rate of guns is slower than frame rate
            # yes I know should have done this at preprocessing stage...
            for j in range(1,N_TIMESTEPS-1):
                if y[i,j-1,n_keys:n_keys+1] == 1 and y[i,j+1,n_keys:n_keys+1] == 1:
                    y[i,j,n_keys:n_keys+1] = 1

            # 7 aug seem to need to fill in 1001 as well for spraying
            for j in range(1,N_TIMESTEPS-2):
                if y[i,j-1,n_keys:n_keys+1] == 1 and y[i,j+2,n_keys:n_keys+1] == 1:
                    y[i,j,n_keys:n_keys+1] = 1
                    y[i,j+1,n_keys:n_keys+1] = 1

            # TODO, include x_aux

            h5file.close()

            # do data aug
            # have the choice of mirroring image 
            # and accompanying mouse movement
            # this seemed to work ok for aim mode, but not deathmatch
            if IS_MIRROR:
                if np.random.rand()<0.3:
                    X[i] = np.flip(X[i],-2) # flip width dim
                    # also need to flip mouse x movement
                    y[i,:,n_keys+n_clicks:n_keys+n_clicks+n_mouse_x] = np.flip(y[i,:,n_keys+n_clicks:n_keys+n_clicks+n_mouse_x],axis=-1)
                    # also must flip 'a' and 'd' keys
                    akey = y[i,:,1]
                    dkey = y[i,:,3]
                    y[i,:,1] = dkey
                    y[i,:,3] = akey
            if True:
                # brightness
                if np.random.rand()<0.5: # was 0.2, raised to 0.5
                    # adjust in range 0.7 to 1.1, <1 darkesns, >1 brightens
                    bright = np.random.rand()*0.6+0.7
                    X[i] *= bright
                    X[i] = np.clip(X[i],0,255).astype(int)

                # contrast
                # follow https://stackoverflow.com/questions/49142561/change-contrast-in-numpy/49142934
                if np.random.rand()<0.5:
                    contrast = np.random.rand()*0.6+0.7
                    X[i] = np.clip(128 + contrast * X[i] - contrast * 128, 0, 255).astype(int)

        return X, y



# manually create a list of all possible files numbers and frame indexes
# of form 'filenum-framenum'
# ['1-3','1-4','1-5',...,'1-999','2-2',,'2-4',...]
# data_list = [str(x1)+'-'+str(x2) for x1 in np.arange(starting_num,highest_num+1) for x2 in np.arange(int(max(FRAMES_STACK*FRAMES_SKIP,ACTIONS_PREV)),1000)]

N_JITTER = 20 # number frames to randomly offset by, going forward only!
# data_list = [str(x1)+'-'+str(x2) for x1 in np.arange(starting_num,highest_num+1) for x2 in np.arange(0,1000-N_TIMESTEPS-int(N_JITTER),N_TIMESTEPS)]
data_list1 = [str(x1)+'-'+str(x2) for x1 in np.arange(starting_num,highest_num+1) for x2 in np.arange(0,1000-N_TIMESTEPS-int(N_JITTER),N_TIMESTEPS)]
data_list2 = [str(x1)+'-'+str(x2) for x1 in np.arange(starting_num,highest_num+1) for x2 in np.arange(0,1000-N_TIMESTEPS-int(N_JITTER),N_TIMESTEPS)]
data_list3 = [str(x1)+'-'+str(x2) for x1 in np.arange(starting_num,highest_num+1) for x2 in np.arange(0,1000-N_TIMESTEPS-int(N_JITTER),N_TIMESTEPS)]
data_list4 = [str(x1)+'-'+str(x2) for x1 in np.arange(starting_num,highest_num+1) for x2 in np.arange(0,1000-N_TIMESTEPS-int(N_JITTER),N_TIMESTEPS)]
data_list_full = [str(x1)+'-'+str(x2) for x1 in np.arange(starting_num,highest_num+1) for x2 in np.arange(0,1000-N_TIMESTEPS-int(N_JITTER),N_TIMESTEPS)]

# note: to do undersampling of non-kill events, we actually create 4 fixed dataloaders each drawing a sample
# with some probability
# I thought about doing this within the dataloader, but then get trouble with uneven batch sizes, 
# keeping track of what used and haven't etc
# this seemed like a pragmatic solution
# thinking about it now, a better alternative might be to create a new data loader dynamically before each epoch

# can do a subselection stage here where go through data files and select or delete if not terrorist team, or fire event etc    
if IS_SUBSELECT:

    # we go through currvars files
    # and create a massive dict in memory with just key = 'filenum_frame'
    # and values helper_i, y_i

    n_filer_per_chunk=100
    info_array = []
    weap_arr=[]
    subselect_helper_dict={}
    # for file_chunk in range(0,int(highest_num/n_filer_per_chunk)):
    for file_chunk in range(0,int(np.ceil(highest_num/n_filer_per_chunk))):
    # for file_chunk in range(0,2):
        curr_vars_file_i = curr_vars_stub+str(file_chunk*n_filer_per_chunk+1)+'_to_'+str((file_chunk+1)*n_filer_per_chunk)+'.npy'
        print('file_chunk',curr_vars_file_i,end='\r')
        dict_chunk = np.load(curr_vars_folder+curr_vars_file_i,allow_pickle=True)
        dict_chunk = dict_chunk.item()

        # dict_chunk['file_num2_frame_1']
        # dict_chunk['file_num999_frame_1']

        for key in dict_chunk.keys():
            dict_key = dict_chunk[key]

            if 'gsi_weap_active' in dict_key[0].keys():
                weap_act = dict_key[0]['gsi_weap_active']['name']
            else:
                weap_act = 'none found'

            # print(weap_act)
            if 'localpos1' in dict_key[0].keys():
                position_i = (dict_key[0]['localpos1'],dict_key[0]['localpos2'])
            else:
                position_i = (0,0)

            # helper_i, actions_inferred, weapon_str
            subselect_helper_dict[key] = [dict_key[2], dict_key[1], weap_act, position_i]

        # subselect_helper['file_num120_frame_445']
    def fn_subselect(data_list_in, subselect_helper_dict, other_prob = 0.1, IS_ADD_PREV=True):

        print('length of data_list before subselect', len(data_list_in))
        total_kill_count=0
        total_death_count=0
        total_motionless_segment=0
        total_ak47=0 # these ak47 variables are badly named, actually they capture whatever subset we're after
        total_ak47_kills=0
        for file_num in range(starting_num,highest_num+1):
            print('subsampling file',file_num,end='\r')
            # print('subsampling file',file_num)

            # frame_possibles = list(np.arange(0,1000-N_TIMESTEPS-int(N_JITTER/2),N_TIMESTEPS))
            frame_possibles = list(np.arange(0,1000-N_TIMESTEPS-int(N_JITTER),N_TIMESTEPS))
            frame_possibles.append(999)
            # print(frame_possibles)
            for seg_i in range(len(frame_possibles)-1):
                seg_start = frame_possibles[seg_i]
                seg_end = frame_possibles[seg_i+1]
                # print(seg_start, seg_end)
                mouse_motionless=0
                wasd_motionless=0
                ak47_per_seg=0
                kills_per_seg=0
                low_freq_region_per_seg=0
                # run through all frames in segment
                for frame_i in range(seg_start, seg_end):
                    [helper_i, actions_inf, weap_str, position_i] = subselect_helper_dict['file_num'+str(file_num)+'_frame_'+str(frame_i)]

                    if helper_i[0]>0:
                        total_kill_count+=1
                        kills_per_seg+=1
                        # prob_select=1.0

                    if helper_i[1]>0:
                        total_death_count+=1

                    if actions_inf[1] == 0 and actions_inf[2] == 0:
                        mouse_motionless+=1

                    if 'w' not in actions_inf[0] and 'a' not in actions_inf[0] and 's' not in actions_inf[0] and 'd' not in actions_inf[0]:
                        wasd_motionless+=1

                    if SUB_TYPE == 'ak':
                        if 'ak47' in weap_str:
                            total_ak47+=1
                            ak47_per_seg+=1
                        if 'ak47' in weap_str and helper_i[0]>0:
                            total_ak47_kills+=1
                    elif SUB_TYPE == 'akm4':
                        if 'ak47' in weap_str or 'm4a1' in weap_str:
                            total_ak47+=1
                            ak47_per_seg+=1
                        if ('ak47' in weap_str or 'm4a1' in weap_str) and helper_i[0]>0:
                            total_ak47_kills+=1
                    else:
                        total_ak47+=1
                        ak47_per_seg+=1
                        if helper_i[0]>0:
                            total_ak47_kills+=1

                    # this is outside T spawn
                    if position_i[0]<-800 and position_i[1]<500:
                        low_freq_region_per_seg+=1
                    elif position_i[0]<-1600 and position_i[1]>2000:
                        low_freq_region_per_seg+=1

                prob_select=other_prob # default is to select w other_prob

                # could also check for change of player via meta info

                if wasd_motionless/N_TIMESTEPS>0.8 and mouse_motionless/N_TIMESTEPS>0.8:
                    # if we were unable to find much movement in the segment
                    # don't sample this
                    prob_select=0.0
                    total_motionless_segment+=1

                if ak47_per_seg/N_TIMESTEPS<0.7:
                    prob_select=0.0

                # 27 Aug this was in the wrong place before, was adding prior segment before kill w any weapon
                if kills_per_seg>0 and ak47_per_seg/N_TIMESTEPS>=0.7:
                    prob_select=1.0
                    # also make sure prior segment is added incase it was deleted
                    if IS_ADD_PREV and seg_i>0 and str(file_num)+'-'+str(frame_possibles[seg_i-1]) not in data_list_in:
                        data_list_in.append(str(file_num)+'-'+str(frame_possibles[seg_i-1]))

                if OVERSAMPLE_LOWFREQ_REGION:
                    if prob_select>0: # don't include if already deleted
                        if low_freq_region_per_seg/N_TIMESTEPS>=0.5:
                            prob_select=1.0

                # based on what we saw in last frame...
                # if prob_select>0.0:
                    # if 'awp' in weap_str or 'ssg08' in weap_str:
                        # prob_select=0.0

                    # if 'ak47' in weap_str or 'm4a1' in weap_str:
                    # if 'ak47' in weap_str: # 27 aug trying this for ak47 only run
                        # prob_select=other_prob
                    # if 'ak47' not in weap_str: 
                        # prob_select = 0.

                # delete the segment w some prob
                if np.random.rand()>prob_select:
                    data_list_in.remove(str(file_num)+'-'+str(seg_start))

        print('length of data_list after subselect', len(data_list_in))
        print(total_motionless_segment, 'total_motionless_segment')
        print(total_kill_count, 'total kill events')
        print(total_death_count, 'total death events')
        print(total_ak47, 'total ak47 frames')
        print(total_ak47_kills, 'total ak47 kills')

        return data_list_in


    data_list1 = fn_subselect(data_list1, subselect_helper_dict, SUB_PROB, True)
    data_list2 = fn_subselect(data_list2, subselect_helper_dict, SUB_PROB, True)
    data_list3 = fn_subselect(data_list3, subselect_helper_dict, SUB_PROB, True)
    data_list4 = fn_subselect(data_list4, subselect_helper_dict, SUB_PROB, True)

    del subselect_helper_dict # try to free up some ram



print('data_list1 training on sequences: ',len(data_list1))
print('data_list1 training on frames: ',len(data_list1*N_TIMESTEPS))

print('data_list2 training on sequences: ',len(data_list2))
print('data_list2 training on frames: ',len(data_list2*N_TIMESTEPS))

print('data_list3 training on sequences: ',len(data_list3))
print('data_list3 training on frames: ',len(data_list3*N_TIMESTEPS))

print('data_list4 training on sequences: ',len(data_list4))
print('data_list4 training on frames: ',len(data_list4*N_TIMESTEPS))


np.random.shuffle(data_list1) # shuffle it (in place)
partition1 = {}
partition1['train'] = data_list1[:int(len(data_list1)*1.)]
partition1['validation'] = data_list1[int(len(data_list1)*0.995):]

np.random.shuffle(data_list2) # shuffle it (in place)
partition2 = {}
partition2['train'] = data_list2[:int(len(data_list2)*1.)]
partition2['validation'] = data_list2[int(len(data_list2)*0.995):]

np.random.shuffle(data_list3) # shuffle it (in place)
partition3 = {}
partition3['train'] = data_list3[:int(len(data_list3)*1.)]
partition3['validation'] = data_list3[int(len(data_list3)*0.995):]

np.random.shuffle(data_list4) # shuffle it (in place)
partition4 = {}
partition4['train'] = data_list4[:int(len(data_list4)*1.)]
partition4['validation'] = data_list4[int(len(data_list4)*0.995):]

# this is not subsampled
partition_full = {}
partition_full['tmp'] = data_list_full[:int(batch_size*2)]
partition_full['train_full'] = data_list_full[:int(len(data_list_full)*1.)]
partition_full['validation_full'] = data_list_full[int(len(data_list_full)*0.995):]


training_generator1 = DataGenerator(list_IDs=partition1['train'], batch_size=batch_size, shuffle=True)
validation_generator1 = DataGenerator(list_IDs=partition1['validation'], batch_size=batch_size, shuffle=True)

training_generator2 = DataGenerator(list_IDs=partition2['train'], batch_size=batch_size, shuffle=True)
validation_generator2 = DataGenerator(list_IDs=partition2['validation'], batch_size=batch_size, shuffle=True)

training_generator3 = DataGenerator(list_IDs=partition3['train'], batch_size=batch_size, shuffle=True)
validation_generator3 = DataGenerator(list_IDs=partition3['validation'], batch_size=batch_size, shuffle=True)

training_generator4 = DataGenerator(list_IDs=partition4['train'], batch_size=batch_size, shuffle=True)
validation_generator4 = DataGenerator(list_IDs=partition4['validation'], batch_size=batch_size, shuffle=True)

tmp_generator = DataGenerator(list_IDs=partition_full['tmp'], batch_size=batch_size, shuffle=True)
training_generator_full = DataGenerator(list_IDs=partition_full['train_full'], batch_size=batch_size, shuffle=True)
validation_generator_full = DataGenerator(list_IDs=partition_full['validation_full'], batch_size=batch_size, shuffle=True)


# load optimiser
if IS_LOAD_WEIGHTS_AND_MODEL:
    print('setting optimiser...')
    with strategy.scope():
        K.set_value(model.optimizer.lr, 0.0)
    hist = model.fit(tmp_generator,epochs=1,verbose=1) 
    # have to load the optimiser after we've compiled the model
    # model._make_train_function()
    model_path = os.path.join(save_dir, weights_name+'_opt.pkl')
    with open(model_path, 'rb') as f:
        opt_weight_values = pickle.load(f)
        with strategy.scope():
            model.optimizer.set_weights(opt_weight_values)
        # opt.set_weights(weight_values)

    # manually set the learning rate in case changed since save
    K.set_value(model.optimizer.lr, l_rate)
    print('updated optimiser to saved state')


print('starting to train...')
    
if False:
    # could just train final layers of model early on
    with strategy.scope():
        print('partial model training...')
        # model.trainable=True
        for layer in model.layers: layer.trainable = True
        model.layers[0].trainable=False
        model.layers[1].trainable=False
        # model.layers[2].trainable=False # this is conv layer!!
        model.compile(loss=custom_loss,optimizer=opt, metrics=[Lclk_acc,no_fire,m_x_acc,m_y_acc, m_x_0,m_y_0])



# I used a different training routine for different datasets
if file_name_stub == 'dm_july2021_':
    for iter_letter in ['a','b','c','d','e','f','g','h','i','j','k']:
        hist = model.fit(training_generator1,epochs=1,workers=4,verbose=1,use_multiprocessing=True, max_queue_size=20)
        tp_save_model(model, save_dir, model_name+iter_letter+'1')
        hist = model.fit(training_generator2,epochs=1,workers=4,verbose=1,use_multiprocessing=True, max_queue_size=20)
        tp_save_model(model, save_dir, model_name+iter_letter+'2')
        hist = model.fit(training_generator3,epochs=1,workers=4,verbose=1,use_multiprocessing=True, max_queue_size=20)
        tp_save_model(model, save_dir, model_name+iter_letter+'3')
        hist = model.fit(training_generator4,epochs=1,workers=4,verbose=1,use_multiprocessing=True, max_queue_size=20)
        tp_save_model(model, save_dir, model_name+iter_letter+'4')

if file_name_stub == 'aim_july2021_expert_' and IS_LOAD_WEIGHTS_AND_MODEL:
    hist = model.fit(training_generator_full,validation_data=validation_generator_full,epochs=4,workers=4,verbose=1,use_multiprocessing=True, max_queue_size=20) 
    # tp_save_model(model, save_dir, model_name+'4')
    hist = model.fit(training_generator_full,validation_data=validation_generator_full,epochs=4,workers=4,verbose=1,use_multiprocessing=True, max_queue_size=20) 
    # tp_save_model(model, save_dir, model_name+'8')
    hist = model.fit(training_generator_full,validation_data=validation_generator_full,epochs=4,workers=4,verbose=1,use_multiprocessing=True, max_queue_size=20) 
    # tp_save_model(model, save_dir, model_name+'12')
    hist = model.fit(training_generator_full,validation_data=validation_generator_full,epochs=4,workers=4,verbose=1,use_multiprocessing=True, max_queue_size=20) 
    tp_save_model(model, save_dir, model_name+'16')
    hist = model.fit(training_generator_full,validation_data=validation_generator_full,epochs=8,workers=4,verbose=1,use_multiprocessing=True, max_queue_size=20) 
    tp_save_model(model, save_dir, model_name+'24')
    hist = model.fit(training_generator_full,validation_data=validation_generator_full,epochs=8,workers=4,verbose=1,use_multiprocessing=True, max_queue_size=20) 
    tp_save_model(model, save_dir, model_name+'32')
    hist = model.fit(training_generator_full,validation_data=validation_generator_full,epochs=8,workers=4,verbose=1,use_multiprocessing=True, max_queue_size=20) 
    tp_save_model(model, save_dir, model_name+'40')
    hist = model.fit(training_generator_full,validation_data=validation_generator_full,epochs=4,workers=4,verbose=1,use_multiprocessing=True, max_queue_size=20) 
    tp_save_model(model, save_dir, model_name+'44')
    hist = model.fit(training_generator_full,validation_data=validation_generator_full,epochs=4,workers=4,verbose=1,use_multiprocessing=True, max_queue_size=20) 
    tp_save_model(model, save_dir, model_name+'48')
    # K.set_value(model.optimizer.lr, l_rate/10)
    hist = model.fit(training_generator_full,validation_data=validation_generator_full,epochs=4,workers=4,verbose=1,use_multiprocessing=True, max_queue_size=20) 
    tp_save_model(model, save_dir, model_name+'52')
    hist = model.fit(training_generator_full,validation_data=validation_generator_full,epochs=4,workers=4,verbose=1,use_multiprocessing=True, max_queue_size=20) 
    tp_save_model(model, save_dir, model_name+'56')
    hist = model.fit(training_generator_full,validation_data=validation_generator_full,epochs=4,workers=4,verbose=1,use_multiprocessing=True, max_queue_size=20) 
    tp_save_model(model, save_dir, model_name+'60')
    hist = model.fit(training_generator_full,validation_data=validation_generator_full,epochs=12,workers=4,verbose=1,use_multiprocessing=True, max_queue_size=20) 
    tp_save_model(model, save_dir, model_name+'72')
    hist = model.fit(training_generator_full,validation_data=validation_generator_full,epochs=12,workers=4,verbose=1,use_multiprocessing=True, max_queue_size=20) 
    tp_save_model(model, save_dir, model_name+'84')


if file_name_stub == 'dm_july2021_expert_' and IS_LOAD_WEIGHTS_AND_MODEL:
    if False:
        for iter_letter in ['a','b','c','d','e','f','g','h','i','j','k']:
            hist = model.fit(training_generator1,epochs=1,workers=4,verbose=1,use_multiprocessing=True, max_queue_size=20)
            tp_save_model(model, save_dir, model_name+iter_letter+'1')
            hist = model.fit(training_generator2,epochs=1,workers=4,verbose=1,use_multiprocessing=True, max_queue_size=20)
            tp_save_model(model, save_dir, model_name+iter_letter+'2')
            hist = model.fit(training_generator3,epochs=1,workers=4,verbose=1,use_multiprocessing=True, max_queue_size=20)
            tp_save_model(model, save_dir, model_name+iter_letter+'3')
            hist = model.fit(training_generator4,epochs=1,workers=4,verbose=1,use_multiprocessing=True, max_queue_size=20)
            tp_save_model(model, save_dir, model_name+iter_letter+'4')
    else:
        hist = model.fit(training_generator_full,validation_data=validation_generator_full,epochs=4,workers=4,verbose=1,use_multiprocessing=True, max_queue_size=20) 
        # tp_save_model(model, save_dir, model_name+'4')
        hist = model.fit(training_generator_full,validation_data=validation_generator_full,epochs=4,workers=4,verbose=1,use_multiprocessing=True, max_queue_size=20) 
        tp_save_model(model, save_dir, model_name+'8')
        hist = model.fit(training_generator_full,validation_data=validation_generator_full,epochs=4,workers=4,verbose=1,use_multiprocessing=True, max_queue_size=20) 
        tp_save_model(model, save_dir, model_name+'12')
        hist = model.fit(training_generator_full,validation_data=validation_generator_full,epochs=4,workers=4,verbose=1,use_multiprocessing=True, max_queue_size=20) 
        tp_save_model(model, save_dir, model_name+'16')
        hist = model.fit(training_generator_full,validation_data=validation_generator_full,epochs=4,workers=4,verbose=1,use_multiprocessing=True, max_queue_size=20) 
        tp_save_model(model, save_dir, model_name+'20')
        hist = model.fit(training_generator_full,validation_data=validation_generator_full,epochs=4,workers=4,verbose=1,use_multiprocessing=True, max_queue_size=20) 
        tp_save_model(model, save_dir, model_name+'24')
        hist = model.fit(training_generator_full,validation_data=validation_generator_full,epochs=4,workers=4,verbose=1,use_multiprocessing=True, max_queue_size=20) 
        tp_save_model(model, save_dir, model_name+'28')
        hist = model.fit(training_generator_full,validation_data=validation_generator_full,epochs=4,workers=4,verbose=1,use_multiprocessing=True, max_queue_size=20) 
        tp_save_model(model, save_dir, model_name+'32')
        hist = model.fit(training_generator_full,validation_data=validation_generator_full,epochs=4,workers=4,verbose=1,use_multiprocessing=True, max_queue_size=20) 
        tp_save_model(model, save_dir, model_name+'36')
        hist = model.fit(training_generator_full,validation_data=validation_generator_full,epochs=4,workers=4,verbose=1,use_multiprocessing=True, max_queue_size=20) 
        tp_save_model(model, save_dir, model_name+'40')
        hist = model.fit(training_generator_full,validation_data=validation_generator_full,epochs=4,workers=4,verbose=1,use_multiprocessing=True, max_queue_size=20) 
        tp_save_model(model, save_dir, model_name+'44')
        hist = model.fit(training_generator_full,validation_data=validation_generator_full,epochs=4,workers=4,verbose=1,use_multiprocessing=True, max_queue_size=20) 
        tp_save_model(model, save_dir, model_name+'48')
        hist = model.fit(training_generator_full,validation_data=validation_generator_full,epochs=4,workers=4,verbose=1,use_multiprocessing=True, max_queue_size=20) 
        tp_save_model(model, save_dir, model_name+'52')
        hist = model.fit(training_generator_full,validation_data=validation_generator_full,epochs=4,workers=4,verbose=1,use_multiprocessing=True, max_queue_size=20) 
        tp_save_model(model, save_dir, model_name+'56')
        hist = model.fit(training_generator_full,validation_data=validation_generator_full,epochs=4,workers=4,verbose=1,use_multiprocessing=True, max_queue_size=20) 
        tp_save_model(model, save_dir, model_name+'60')

if file_name_stub == 'dm_july2021_expert_' and not IS_LOAD_WEIGHTS_AND_MODEL:
    # training from scratch
    for iter_letter in ['a','b','c','d','e','f']:
        hist = model.fit(training_generator_full,validation_data=validation_generator_full,epochs=4,workers=4,verbose=1,use_multiprocessing=True, max_queue_size=20) 
        tp_save_model(model, save_dir, model_name+iter_letter+'4')
        hist = model.fit(training_generator_full,validation_data=validation_generator_full,epochs=4,workers=4,verbose=1,use_multiprocessing=True, max_queue_size=20) 
        tp_save_model(model, save_dir, model_name+iter_letter+'8')
        hist = model.fit(training_generator_full,validation_data=validation_generator_full,epochs=4,workers=4,verbose=1,use_multiprocessing=True, max_queue_size=20) 
        tp_save_model(model, save_dir, model_name+iter_letter+'12')
        hist = model.fit(training_generator_full,validation_data=validation_generator_full,epochs=4,workers=4,verbose=1,use_multiprocessing=True, max_queue_size=20) 
        tp_save_model(model, save_dir, model_name+iter_letter+'16')

if file_name_stub == 'aim_july2021_expert_' and not IS_LOAD_WEIGHTS_AND_MODEL:
    # training from scratch
    for iter_letter in ['a','b','c','d','e','f']:
        hist = model.fit(training_generator_full,validation_data=validation_generator_full,epochs=4,workers=4,verbose=1,use_multiprocessing=True, max_queue_size=20) 
        tp_save_model(model, save_dir, model_name+iter_letter+'4')
        hist = model.fit(training_generator_full,validation_data=validation_generator_full,epochs=4,workers=4,verbose=1,use_multiprocessing=True, max_queue_size=20) 
        tp_save_model(model, save_dir, model_name+iter_letter+'8')
        hist = model.fit(training_generator_full,validation_data=validation_generator_full,epochs=4,workers=4,verbose=1,use_multiprocessing=True, max_queue_size=20) 
        tp_save_model(model, save_dir, model_name+iter_letter+'12')
        hist = model.fit(training_generator_full,validation_data=validation_generator_full,epochs=4,workers=4,verbose=1,use_multiprocessing=True, max_queue_size=20) 
        tp_save_model(model, save_dir, model_name+iter_letter+'16')

if file_name_stub == 'dm_inferno_expert_' or file_name_stub == 'dm_mirage_expert_' or file_name_stub == 'dm_nuke_expert_':
    K.set_value(model.optimizer.lr, 0.00001)
    hist = model.fit(training_generator_full,validation_data=validation_generator_full,epochs=4,workers=4,verbose=1,use_multiprocessing=True, max_queue_size=20) 
    tp_save_model(model, save_dir, model_name+'4')
    hist = model.fit(training_generator_full,validation_data=validation_generator_full,epochs=4,workers=4,verbose=1,use_multiprocessing=True, max_queue_size=20) 
    tp_save_model(model, save_dir, model_name+'8')
    hist = model.fit(training_generator_full,validation_data=validation_generator_full,epochs=4,workers=4,verbose=1,use_multiprocessing=True, max_queue_size=20) 
    tp_save_model(model, save_dir, model_name+'12')
    hist = model.fit(training_generator_full,validation_data=validation_generator_full,epochs=4,workers=4,verbose=1,use_multiprocessing=True, max_queue_size=20) 
    tp_save_model(model, save_dir, model_name+'16')
    hist = model.fit(training_generator_full,validation_data=validation_generator_full,epochs=4,workers=4,verbose=1,use_multiprocessing=True, max_queue_size=20) 
    tp_save_model(model, save_dir, model_name+'20')
    hist = model.fit(training_generator_full,validation_data=validation_generator_full,epochs=4,workers=4,verbose=1,use_multiprocessing=True, max_queue_size=20) 
    tp_save_model(model, save_dir, model_name+'24')


print('took',np.round(time.time()-start_time,1),' secs\n')


