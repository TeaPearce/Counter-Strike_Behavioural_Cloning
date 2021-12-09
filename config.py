import numpy as np
import os
import time
import pickle

# this file stores parameters that are used across many other files
# also stores some key functions used in several places

loop_fps=16 # 16 is main one, but can try to run at 24

# dimensions of image to reduce to
# used when grabbing screen and also building NN
csgo_img_dimension = (150,280) # offset_height_top = 135, offset_height_bottom = 135, offset_sides = 100
csgo_game_res = (1024,768) # this is 4x3, windowed and down sized slightly
# btw mouse we use is 2.54 sensitivity, w raw input off

N_TIMESTEPS = 96 # number of time steps for lstm
IS_MIRROR = False # whether to double data with flipped image
GAMMA = 0.995 # reward decay for RL setting, val

input_shape = (N_TIMESTEPS,csgo_img_dimension[0],csgo_img_dimension[1],3)
input_shape_lstm_pred = (1,csgo_img_dimension[0],csgo_img_dimension[1],3) # need to say only one frame when predicting

# params for discretising mouse
mouse_x_possibles = [-1000.0,-500.0, -300.0, -200.0, -100.0, -60.0, -30.0, -20.0, -10.0, -4.0, -2.0, -0.0, 2.0, 4.0, 10.0, 20.0, 30.0, 60.0, 100.0, 200.0, 300.0, 500.0,1000.0]
mouse_y_possibles = [-200.0, -100.0, -50.0, -20.0, -10.0, -4.0, -2.0, -0.0, 2.0, 4.0, 10.0, 20.0, 50.0, 100.0, 200.0]
mouse_x_lim = (mouse_x_possibles[0],mouse_x_possibles[-1])
mouse_y_lim = (mouse_y_possibles[0],mouse_y_possibles[-1])

# below options are no longer used, are here due to previous agent iterations
IS_CONTRAST = False # whether to add contrast to image, REDUNDANT
FRAMES_STACK = 3 # how many frames to use as input, REDUNDANT
FRAMES_SKIP = 4 # how many frames to skip in between each of the frames stacked together, REDUNDANT
ACTIONS_PREV = 3 # how many previous actions (and rewards?) to use as aux input, REDUNDANT
AUX_INPUT_ON = False # whether to use aux input at all, REDUNDANT
DATA_STEP = 1 # whether to skip through training data (=1), only use every x steps, REDUNDANT

def mouse_preprocess(mouse_x, mouse_y):
    # clip and distcretise mouse
    mouse_x = np.clip(mouse_x, mouse_x_lim[0],mouse_x_lim[1])
    mouse_y = np.clip(mouse_y, mouse_y_lim[0],mouse_y_lim[1])

    # find closest in list
    mouse_x = min(mouse_x_possibles, key=lambda x_:abs(x_-mouse_x))
    mouse_y = min(mouse_y_possibles, key=lambda x_:abs(x_-mouse_y))

    return mouse_x, mouse_y

# how many slots were used for each action type?
n_keys = 11 # number of keyboard outputs, w,s,a,d,space,ctrl,shift,1,2,3,r
n_clicks = 2 # number of mouse buttons, left, right
n_mouse_x = len(mouse_x_possibles) # number of outputs on mouse x axis
n_mouse_y = len(mouse_y_possibles) # number of outputs on mouse y axis
n_extras = 3 # number of extra aux inputs, eg health, ammo, team. others could be weapon, kills, deaths
aux_input_length = n_keys+n_clicks+1+1+n_extras # aux uses continuous input for mouse this is multiplied by ACTIONS_PREV elsewhere

def reward_fn(kill, death, shoot):
    # all inputs should be one hot encoded
    # return kill - 0.5*death - 0.01*shoot 
    return kill - 0.5*death - 0.02*shoot 

def onehot_to_actions(y_preds):
    # assumes y_preds is a single vector - only one time frame
    # converts NN output, [0,0.9,0.1,0,0,0,0.3,...]
    # to list of actions, [keys_pressed, mouse_x, mouse_y, clicks]

    y_preds = y_preds.squeeze()

    # mouse_x_possibles = [-300.0,-250.0,-200.0,-150.0,-100.0,-50.0,-40.0,-30.0,-20.0,-18.0,-16.0,-14.0,-12.0,-10.0,-8.0,-6.0,-5.0,-4.0,-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0,4.0,5.0,6.0,8.0,10.0,12.0,14.0,16.0,18.0,20.0,30.0,40.0,50.0,100.0,150.0,200.0,250.0,300.0]
    # mouse_y_possibles = [-50.0,-40.0,-30.0,-20.0,-18.0,-16.0,-14.0,-12.0,-10.0,-8.0,-6.0,-5.0,-4.0,-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0,4.0,5.0,6.0,8.0,10.0,12.0,14.0,16.0,18.0,20.0,30.0,40.0,50.0]

    keys_pred = y_preds[0:n_keys]
    Lclicks_pred = y_preds[n_keys:n_keys+1]
    Rclicks_pred = y_preds[n_keys+1:n_keys+n_clicks]
    mouse_x_pred = y_preds[n_keys+n_clicks:n_keys+n_clicks+n_mouse_x]
    mouse_y_pred = y_preds[n_keys+n_clicks+n_mouse_x:n_keys+n_clicks+n_mouse_x+n_mouse_y]
    val_pred = y_preds[n_keys+n_clicks+n_mouse_x+n_mouse_y:n_keys+n_clicks+n_mouse_x+n_mouse_y+1][0]

    keys_pressed=[]
    keys_pressed_onehot = np.round(keys_pred)
    if keys_pressed_onehot[0]==1:
        keys_pressed.append('w')
    if keys_pressed_onehot[1]==1:
        keys_pressed.append('a')
    if keys_pressed_onehot[2]==1:
        keys_pressed.append('s')
    if keys_pressed_onehot[3]==1:
        keys_pressed.append('d')
    if keys_pressed_onehot[4]==1:
        keys_pressed.append('space')
    if keys_pressed_onehot[5]==1:
        keys_pressed.append('ctrl')
    if keys_pressed_onehot[6]==1:
        keys_pressed.append('shift')
    if keys_pressed_onehot[7]==1:
        keys_pressed.append('1')
    if keys_pressed_onehot[8]==1:
        keys_pressed.append('2')
    if keys_pressed_onehot[9]==1:
        keys_pressed.append('3')
    if keys_pressed_onehot[10]==1:
        keys_pressed.append('r')

    Lclicks = int(np.round(Lclicks_pred))
    Rclicks = int(np.round(Rclicks_pred))

    id = np.argmax(mouse_x_pred)
    mouse_x = mouse_x_possibles[id]
    id = np.argmax(mouse_y_pred)
    mouse_y = mouse_y_possibles[id]

    return [keys_pressed,mouse_x,mouse_y,Lclicks,Rclicks,val_pred]


def actions_to_onehot(keys_pressed, mouse_x, mouse_y, Lclicks, Rclicks):
    # again only does this for a single set of actions
    # converts list of actions,  [keys_pressed,mouse_x,mouse_y,Lclicks,Rclicks]
    # to one hot vectors for each item in list

    keys_pressed_onehot = np.zeros(n_keys)
    mouse_x_onehot = np.zeros(n_mouse_x)
    mouse_y_onehot = np.zeros(n_mouse_y)
    Lclicks_onehot = np.zeros(1)
    Rclicks_onehot = np.zeros(1)

    for key in keys_pressed:
        if key=='w':
            keys_pressed_onehot[0]=1
        elif key=='a':
            keys_pressed_onehot[1]=1
        elif key=='s':
            keys_pressed_onehot[2]=1
        elif key=='d':
            keys_pressed_onehot[3]=1
        elif key=='space':
            keys_pressed_onehot[4]=1
        elif key=='ctrl':
            keys_pressed_onehot[5]=1
        elif key=='shift':
            keys_pressed_onehot[6]=1
        elif key=='1':
            keys_pressed_onehot[7]=1
        elif key=='2':
            keys_pressed_onehot[8]=1
        elif key=='3':
            keys_pressed_onehot[9]=1
        elif key=='r':
            keys_pressed_onehot[10]=1

    Lclicks_onehot[0] = Lclicks
    Rclicks_onehot[0] = Rclicks

    # need to match mouse_x to possible values
    # to figure out its id
    id = mouse_x_possibles.index(mouse_x)
    mouse_x_onehot[id] = 1
    id = mouse_y_possibles.index(mouse_y)
    mouse_y_onehot[id] = 1

    assert mouse_x_onehot.sum() == 1
    assert mouse_y_onehot.sum() == 1

    return keys_pressed_onehot,Lclicks_onehot,Rclicks_onehot,mouse_x_onehot,mouse_y_onehot


def tp_load_model(save_dir, model_name):
    # load model from description json file and weight h5 file

    # load architecture
    from tensorflow.keras.models import model_from_json
    
    model_path = os.path.join(save_dir, model_name+'.json')
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    print('loaded model from ', model_path)

    # load weights
    model_path = os.path.join(save_dir, model_name+'.h5')
    model.load_weights(model_path)
    print('loaded weights from ', model_path)

    return model


def tp_save_model(model, save_dir, model_name):
    # save model from description json file and weight h5 file

    # create directory if needed
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # save model architecture to json - a string describing model
    model_json = model.to_json()
    model_path = os.path.join(save_dir, model_name+'.json')
    open(model_path, 'w').write(model_json)
    print('saved model to ', model_path)

    # save weights to HDF5
    model_path = os.path.join(save_dir, model_name+'.h5')
    model.save_weights(model_path)
    print('saved weights to ', model_path)

    # save optimiser state
    # https://stackoverflow.com/questions/49503748/save-and-load-model-optimizer-state
    import tensorflow.keras.backend as K
    symbolic_weights = getattr(model.optimizer, 'weights')
    weight_values = K.batch_get_value(symbolic_weights)
    model_path = os.path.join(save_dir, model_name+'_opt.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(weight_values, f)
    print('saved optimizer to ', model_path)

    # open later
    # with open('optimizer.pkl', 'rb') as f:
    #     weight_values = pickle.load(f)
    #     model.optimizer.set_weights(weight_values)

    # as of 11 nov going to also save a dictionary with key hyperparams from config file 
    hypers={}
    hypers['csgo_img_dimension'] = csgo_img_dimension
    hypers['csgo_game_res'] = csgo_game_res
    hypers['FRAMES_STACK'] = FRAMES_STACK
    hypers['N_TIMESTEPS'] = N_TIMESTEPS
    hypers['FRAMES_SKIP'] = FRAMES_SKIP
    hypers['ACTIONS_PREV'] = ACTIONS_PREV
    hypers['AUX_INPUT_ON'] = AUX_INPUT_ON
    hypers['FRAMES_SKIP'] = FRAMES_SKIP
    hypers['mouse_x_possibles'] = mouse_x_possibles
    hypers['mouse_y_possibles'] = mouse_y_possibles
    hypers['n_keys'] = n_keys
    hypers['n_clicks'] = n_clicks
    hypers['n_extras'] = n_extras
    hypers['aux_input_length'] = aux_input_length

    model_path = os.path.join(save_dir, model_name+'.p')
    pickle.dump(hypers, open(model_path, 'wb'))
    print('saved hypers to ', model_path)

    if False:
        # this is easiest way to save BUT needs to know all
        # custom losses etc when loading
        # hence I swtiched to json and weights
        save_dir_2='F:\\02_saved_models_hdd' # seems to need it typed like this
        model_path = os.path.join(save_dir_2, model_name)
        model.save(model_path)
        print('Saved trained model at %s ' % model_path)

        model = keras.models.load_model(model_path)
        
    return



def get_highest_num(file_name_stub, folder_name):
    # for training data files, return highest number of that file name
    # must be named like 'blah_blah_number.extenstion'
    highest_num = 0
    for file in os.listdir(folder_name):
        if file_name_stub in file:
            num = int(file.split('.')[0].split('_')[-1])
            if num > highest_num: highest_num=num
    print(highest_num)
    return highest_num

# only use this function on a windows machine
# it will crash if try to import key_output on linux etc
import os
if os.name == 'nt': # if windows
    from key_output import HoldKey, ReleaseKey, n_char
    def wait_for_loop_end(loop_start_time, loop_fps, n_loops=0, is_clear_decals=True):
        # this is added to the end of each loop in game
        # allows to wait until the correct time before releasing
        # optionally can send an 'n' key to clear decals
        # seems like a good moment to do a non-essential thing if have spare time

        if is_clear_decals:
            # clear decals every x seconds
            if n_loops%(5*loop_fps) == 0:
                HoldKey(n_char)
                ReleaseKey(n_char)

        # if too slow, tell user
        if time.time() > loop_start_time + 1/loop_fps:
            pass
            # print('arrived later than wanted to :/, took ',round(time.time() - loop_start_time,4))
        else:
            # wait until end of time step
            while time.time() < loop_start_time + 1/loop_fps:
                time.sleep(0.001)
                pass
        return
