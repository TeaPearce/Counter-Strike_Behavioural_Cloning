import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # to force CPU
import time
import mss
import cv2
import socket
import sys
import struct
import math
import random
import win32api as wapi
import win32api
import win32gui
import win32process
import ctypes
from ctypes  import *
from pymem import *
import h5py

import requests
from http.server import BaseHTTPRequestHandler, HTTPServer
import json

import numpy as np

from key_input import key_check, mouse_check, mouse_l_click_check, mouse_r_click_check
from key_output import set_pos, HoldKey, ReleaseKey
from key_output import left_click, hold_left_click, release_left_click
from key_output import right_click, hold_right_click, release_right_click
from key_output import w_char, s_char, a_char, d_char, n_char, q_char, b_char
from key_output import ctrl_char, shift_char, space_char
from key_output import r_char, one_char, two_char, three_char, four_char
from key_output import p_char, e_char, c_char_, t_char, cons_char, ret_char
from key_output import m_char, u_char, under_char, g_char, esc_char
from key_output import i_char, v_char, o_char, g_char, k_char, seven_char, x_char, c_char2, y_char

from tensorflow import keras
import tensorflow as tf
import tensorflow.keras.backend as K

from screen_input import grab_window
from config import *

# this script applies a trained NN in a deathmatch environment

# select the game window
print('selecting the game window...')
hwin_orig = win32gui.GetForegroundWindow() # remember original window
# hwin_csgo = win32gui.FindWindow(None,'Counter-Strike: Global Offensive')
hwin_csgo = win32gui.FindWindow(None,'Counter-Strike: Global Offensive - Direct3d 9') # as of Feb 2022
win32gui.SetForegroundWindow(hwin_csgo)
time.sleep(1)

# get info about resolution and monitors
sct = mss.mss()
if len(sct.monitors) == 3:
    Wd, Hd = sct.monitors[2]["width"], sct.monitors[2]["height"]
else:
    Wd, Hd = sct.monitors[1]["width"], sct.monitors[1]["height"]

print('capturing mouse position...')
time.sleep(0.2)
mouse_x_mid, mouse_y_mid = mouse_check()

mins_per_iter = 10

# load model from
model_names = ['ak47_sub_55k_drop_d4_dmexpert_28'] # our best performing dm agent, pretrained and finetuned on expert dm data
# model_names = ['ak47_sub_55k_drop_d4'] # pretrained agent
# model_names = ['ak47_sub_55k_drop_d4_aimexpertv2_60'] # pretrained and finetuned on expert aim mode
# model_names = ['July_remoterun7_g9_4k_n32_recipe_ton96__e14'] # pretrained on full dataset
model_save_dir = os.path.join(os.getcwd(),'model')
model_save_dir_overflow = 'F:/2021/01_remotemodels_overflow' # could also be in here

# folder to save pickle about rewards etc
pickle_reward_folder = ''
pickle_reward_name = 'rewards_.p'

pickle_reward_path = os.path.join(pickle_reward_folder, pickle_reward_name)
pickle.dump([], open(pickle_reward_path, 'wb'))
print('saved pickled rewards',pickle_reward_path)

# which actions do you want to allow the agent to control?
IS_CLICKS=1 # allows only to play with left click
IS_RCLICK=0
IS_MOUSEMOVE=1
IS_WASD=1 # apply wsad only
IS_JUMP=1
IS_RELOAD=1
IS_KEYS=0 # remainder of keys -- 1, 2, 3, shift, ctrl

# should the agent choose best guess action or according to prob
IS_SPLIT_MOUSE=True # whether to do one mouse update per loop, or divide by 2
IS_PROBABILISTIC_ACTIONS = True # TODO, only set for left click atm
ENT_REG = 0.05 # entropy regularisation
N_FILES_RESTART = 500 # how many files to save (of 1000 frames) before map restart
SAVE_TRAIN_DATA=False # whether to actually save the files and do training
IS_DEMO=False # show agent vision with overlay
IS_GSI=False # whether to extract kill, death and aux info using GSI (must be set up on your machine)


if IS_GSI:
    from meta_utils import *
# else:
#     from meta_utils_noserver import *

def mp_restartgame():
    # is_esc lets send double e
    for c in [cons_char,m_char,p_char,under_char,r_char,e_char,s_char,t_char,a_char,r_char,t_char,g_char,a_char,m_char,e_char,space_char,one_char,ret_char,cons_char,esc_char,esc_char]:
        # type mp_restartgame 1
        if c == under_char:
            HoldKey(shift_char)
            HoldKey(under_char)
            ReleaseKey(under_char)
            ReleaseKey(shift_char)
        else:
            HoldKey(c)
            ReleaseKey(c)
        time.sleep(0.1)

    time.sleep(3)
    # should try to buy ak47
    for c in [cons_char,g_char,i_char,v_char,e_char, space_char, 
             w_char, e_char, a_char, p_char, o_char, n_char, under_char, a_char, k_char, four_char, seven_char,   ret_char,cons_char,esc_char,esc_char]:
        # type give weapon_ak47
        if c == under_char:
            HoldKey(shift_char)
            HoldKey(under_char)
            ReleaseKey(under_char)
            ReleaseKey(shift_char)
        else:
            HoldKey(c)
            ReleaseKey(c)
        time.sleep(0.1)

    return

def pause_game():
    # pause game
    for c in [cons_char,p_char,a_char,u_char,s_char,e_char,ret_char,cons_char]:
        # type pause
        time.sleep(0.1)
        HoldKey(c)
        ReleaseKey(c)
    return

n_iters_total = len(model_names)
for training_iter in range(n_iters_total):

    model_name = model_names[training_iter]

    print('\n\n training_iter',training_iter,', model_name ',model_name,'preparing buffer...')
    # need a small buffer of images and aux
    recent_imgs = []
    recent_actions = []
    recent_mouses = [] # store continuous values
    recent_health = []
    recent_ammo = []
    recent_team = []
    recent_val = []

    # do this every time incase clicked away
    win32gui.SetForegroundWindow(hwin_csgo)
    time.sleep(0.5)
    mp_restartgame()
    time.sleep(0.5)

    for i in range(0,16):
        loop_start_time = time.time()

        # image
        img_small = grab_window(hwin_csgo, game_resolution=csgo_game_res, SHOW_IMAGE=False)
        x_img = np.expand_dims(img_small, 0)
        x_img = x_img.astype('float16')
        x_img = x_img#/255
        recent_imgs.append(x_img)

        
        # actions
        dummy_action=np.zeros(int(aux_input_length))
        recent_mouses.append([0.,0.])
        dummy_action[0] = 1 # encourage to start moving
        recent_actions.append(dummy_action)
        recent_val.append(0)

        # aux extra info
        if IS_GSI:
            server.handle_request()
        recent_health.append(100)
        recent_ammo.append(30)
        recent_team.append('T')

        while time.time() < loop_start_time + 1/loop_fps:
            time.sleep(0.01)


    print('\n\n training_iter',training_iter,'starting loop...')

    # load stateful version of current model
    try:
        model_run = tp_load_model(model_save_dir, model_name+'_stateful')
    except:
        print('\n\nmodel not in main folder, trying overflow\n\n')
        model_run = tp_load_model(model_save_dir_overflow, model_name+'_stateful')

    n_loops = 0 # how many times loop through 
    keys_pressed_apply=[]
    keys_pressed=[]
    Lclicks=0
    Rclicks=0
    count_inaction=0 # count how many frames takes no action
    training_data=[]; hdf5_num=1
    iteration_deaths=0; iteration_kills=0;
    while n_loops<1000*(mins_per_iter + 0.02): # x minutes
        if IS_GSI:
            if 'map' not in server.data_all.keys() or 'player' not in server.data_all.keys():
                print('not running, map or player not in keys:', server.data_all.keys())
                time.sleep(5)
                continue

        loop_start_time = time.time()
        n_loops += 1
        keys_pressed_prev = keys_pressed.copy()
        Lclicks_prev = Lclicks
        Rclicks_prev = Rclicks

        # delete oldest elements (at head)
        del recent_imgs[0]
        del recent_actions[0]
        del recent_mouses[0]
        del recent_health[0]
        del recent_ammo[0]
        del recent_team[0]
        del recent_val[0]

        # grab screenshot and add to stack
        img_small = grab_window(hwin_csgo, game_resolution=csgo_game_res, SHOW_IMAGE=False)
        x_img = np.expand_dims(img_small, 0)
        x_img = x_img.astype('float16')
        x_img = x_img#/255
        recent_imgs.append(x_img)

        # package frames together for x_input_main
        x_input_main = np.zeros(input_shape_lstm_pred)
        x_input_main[0] = recent_imgs[-1]
        x_input_main = np.expand_dims(x_input_main,0)
    
        # run fwd pass through NN
        time_before_pass = time.time()
        y_preds = model_run.predict_on_batch(x_input_main)
        if n_loops <= 1:
            time_for_pass=0.1
        else:
            time_for_pass = 0.5*time_for_pass + 0.5*(time.time() - time_before_pass)

        [keys_pressed,mouse_x,mouse_y,Lclicks,Rclicks,val_pred] = onehot_to_actions(y_preds) 

        # manually inspect probabilities
        y_preds = y_preds.squeeze()
        keys_pred = y_preds[0:n_keys]
        clicks_pred = y_preds[n_keys:n_keys+n_clicks]
        mouse_x_pred = y_preds[n_keys+n_clicks:n_keys+n_clicks+n_mouse_x]
        mouse_y_pred = y_preds[n_keys+n_clicks+n_mouse_x:n_keys+n_clicks+n_mouse_x+n_mouse_y]
        val_pred2 = y_preds[n_keys+n_clicks+n_mouse_x+n_mouse_y:n_keys+n_clicks+n_mouse_x+n_mouse_y+1]

        # check for no movement, if none for n steps, then choose and action probabilistically
        if IS_WASD:
            if np.array([x in keys_pressed for x in ['w','s','a','d']]).sum()==0 and mouse_x==0 and mouse_y==0:
            # if np.array([x in keys_pressed for x in ['w','s','a','d']]).sum()==0:
                count_inaction+=1
            else:
                count_inaction=0
        else:
            if mouse_x==0 and mouse_y==0:
                count_inaction+=1
            else:
                count_inaction=0

        # overwrite best guess action with probabilistica;y selected action
        if IS_PROBABILISTIC_ACTIONS or count_inaction>8:
            Lclick_prob = clicks_pred[0]
            if np.random.rand()<Lclick_prob and IS_CLICKS:
                Lclicks=1
            else:
                Lclicks=0

            Rclick_prob = clicks_pred[1]
            if np.random.rand()<Rclick_prob and IS_CLICKS:
                Rclicks=1
            else:
                Rclicks=0

            if IS_JUMP:
                jump_prob = keys_pred[4:5][0]
                if 'expert' in model_name:
                    jump_prob*=5  # manually increase?
                if 'space' in keys_pressed:
                    keys_pressed.remove('space')
                if np.random.rand()>jump_prob:
                    pass
                else:
                    keys_pressed.append('space')

            if IS_RELOAD:
                reload_prob = keys_pred[n_keys-1:n_keys][0]
                if 'r' in keys_pressed:
                    keys_pressed.remove('r')
                if np.random.rand()>reload_prob:
                    pass
                else:
                    keys_pressed.append('r')

            if count_inaction>8 and IS_MOUSEMOVE: # using this one usually
                mouse_x = np.random.choice(mouse_x_possibles, size=1, p=mouse_x_pred)[0]
                mouse_y = np.random.choice(mouse_y_possibles, size=1, p=mouse_y_pred)[0]

            if count_inaction>96 and IS_MOUSEMOVE:
                print('\n\n choosing random mouse \n')
                mouse_x = np.random.choice(mouse_x_possibles, size=1)[0]
                mouse_y = np.random.choice(mouse_y_possibles, size=1)[0]
                model_run.reset_states()
                print('\n\n reset states')

            # clear lstm state w some prob in case bunged up
            if np.random.rand()>0.999: 
            # if n_loops % 256==0: 
                model_run.reset_states()
                print('\n\n reset states')

            if count_inaction>96 and IS_WASD:
                if np.random.rand()<np.maximum(keys_pred[0],0.05):
                    keys_pressed.append('w')
                elif np.random.rand()<np.maximum(keys_pred[1],0.05):
                    keys_pressed.append('a')
                elif np.random.rand()<np.maximum(keys_pred[2],0.05):
                    keys_pressed.append('s')
                elif np.random.rand()<np.maximum(keys_pred[3],0.05):
                    keys_pressed.append('d')


        # could apply some smoothing here
        # mouse_x_smooth = 0.9*mouse_x +  0.1*recent_mouses[-1][0]*mouse_x_lim[1] # + np.random.normal(0,scale = abs(mouse_x/5))
        # mouse_y_smooth = 0.9*mouse_y +  0.1*recent_mouses[-1][1]*mouse_y_lim[1] # + np.random.normal(0,scale = abs(mouse_y/5))
        mouse_x_smooth = mouse_x
        mouse_y_smooth = mouse_y

        mouse_x_smooth = np.clip(mouse_x_smooth,-300,300)


        if IS_MOUSEMOVE:
            if IS_SPLIT_MOUSE:
                set_pos(mouse_x_mid + mouse_x_smooth/2, mouse_y_mid + mouse_y_smooth/2,Wd, Hd)
            else:
                set_pos(mouse_x_mid + mouse_x_smooth/1, mouse_y_mid + mouse_y_smooth/1,Wd, Hd)

        if n_loops>2:
            if len(model_name)>15:
                model_name_print = model_name[-15:]
            else:
                model_name_print = model_name
            # if n_loops%4 == 0:
            print(model_name_print, 'it', training_iter, '/',n_iters_total, ', n', n_loops, ', ds',round(iteration_deaths*1000/n_loops,3), ', ks',round(iteration_kills*1000/n_loops,3), ', fwd',round(time_for_pass,3),', ms', round(time.time()-loop_start_time,3),', inaction',count_inaction,end='\r')
            # print('training_iter', training_iter, 'of',n_iters_total, ', n_loops', n_loops, ', iteration_deaths',round(iteration_deaths*1000/n_loops,3), ', iteration_kills',round(iteration_kills*1000/n_loops,3), ', val_pred',round(val_pred,4), ', time_for_pass',round(time_for_pass,4),', count_inaction',count_inaction,end='\r')

            if n_loops%1000==0:
                print('')

        # think we only needed this when feeding in aux info?
        keys_pressed_onehot,Lclicks_onehot,Rclicks_onehot,mouse_x_onehot,mouse_y_onehot = actions_to_onehot(keys_pressed, mouse_x, mouse_y, Lclicks, Rclicks)

        # append new element to buffer lists at tail
        recent_actions.append(np.concatenate([keys_pressed_onehot, Lclicks_onehot, Rclicks_onehot, mouse_x_onehot, mouse_y_onehot]))
        recent_mouses.append([mouse_x/mouse_x_lim[1],mouse_y/mouse_y_lim[1]])
        recent_val.append(val_pred)
        if IS_GSI:
            recent_health.append(server.data_all['player']['state']['health'])
            recent_team.append(server.data_all['player']['team'])
            gsi_weapons = server.data_all['player']['weapons']
        else:
            recent_health.append(-99)
            recent_team.append('No GSI')
            gsi_weapons = None
        recent_ammo.append(None) # could parse later if need

        helper_i = np.zeros(6) # this is now [0 kill, 1 death, 2 original_value_pred, 3 advantage (computed later), 4 reward_t (computed later), 5 value_t (computed later) ]
        helper_i[2] = val_pred
        curr_vars={}
        if IS_GSI:
            curr_vars['gsi_kills'] = server.data_all['player']['match_stats']['kills']
            curr_vars['gsi_deaths'] = server.data_all['player']['match_stats']['deaths']
        else:
            curr_vars['gsi_kills'] = -99
            curr_vars['gsi_deaths'] = -99
        if n_loops<2:
            pass
        else:
            if curr_vars['gsi_kills']==prev_vars['gsi_kills']+1:
                helper_i[0]=1 # got a kill
                iteration_kills+=1
                # print('killed a dude')
            if curr_vars['gsi_deaths']==prev_vars['gsi_deaths']+1:
                helper_i[1]=1 # died
                iteration_deaths+=1
                # print('RIP :/')
        prev_vars = curr_vars.copy()


        # implement actions
        # release actions held since last time
        if IS_WASD:
            if 'w' in keys_pressed_prev and 'w' not in keys_pressed:
                ReleaseKey(w_char)
            if 'a' in keys_pressed_prev and 'a' not in keys_pressed:
                ReleaseKey(a_char)
            if 's' in keys_pressed_prev and 's' not in keys_pressed:
                ReleaseKey(s_char)
            if 'd' in keys_pressed_prev and 'd' not in keys_pressed:
                ReleaseKey(d_char)
        if IS_JUMP:
            if 'space' in keys_pressed_prev and 'space' not in keys_pressed:
                ReleaseKey(space_char)
        if IS_RELOAD:
            if 'r' in keys_pressed_prev and 'r' not in keys_pressed:
                ReleaseKey(r_char)
        if IS_KEYS:
            if 'shift' in keys_pressed_prev and 'shift' not in keys_pressed:
                ReleaseKey(shift_char)
            if 'ctrl' in keys_pressed_prev and 'ctrl' not in keys_pressed:
                ReleaseKey(ctrl_char)
            if '1' in keys_pressed_prev and '1' not in keys_pressed:
                ReleaseKey(one_char)
            if '2' in keys_pressed_prev and '2' not in keys_pressed:
                ReleaseKey(two_char)
            if '3' in keys_pressed_prev and '3' not in keys_pressed:
                ReleaseKey(three_char)
        if IS_CLICKS:
            if Lclicks==0 and Lclicks_prev==1:
                release_left_click()
        if IS_RCLICK:
            if Rclicks==0 and Rclicks_prev==1:
                release_right_click()

        # press keys 
        if IS_WASD:
            if 'w' in keys_pressed:
                HoldKey(w_char)
            if 'a' in keys_pressed:
                HoldKey(a_char)
            if 's' in keys_pressed:
                HoldKey(s_char)
            if 'd' in keys_pressed:
                HoldKey(d_char)
        if IS_JUMP:
            if 'space' in keys_pressed:
                HoldKey(space_char)
        if IS_RELOAD:
            if 'r' in keys_pressed:
                HoldKey(r_char)
        if IS_KEYS:
            if 'shift' in keys_pressed:
                HoldKey(shift_char)
            if 'ctrl' in keys_pressed:
                HoldKey(ctrl_char)
            if '1' in keys_pressed:
                HoldKey(one_char)
            if '2' in keys_pressed:
                HoldKey(two_char)
            if '3' in keys_pressed:
                HoldKey(three_char)
        if IS_CLICKS:
            if Lclicks==1:
                hold_left_click()
        if IS_RCLICK:
            if Rclicks==1:
                hold_right_click()


        # refresh GSI - I care less about this being up to date, so do last
        if IS_GSI:
            server.handle_request()

        keys_pressed_tp = key_check()
        if 'Q' in keys_pressed_tp:
            # exit loop
            print('exiting...')
            if IS_GSI:
                server.server_close()
            break

        # mouse movement
        if IS_SPLIT_MOUSE and IS_MOUSEMOVE:
            # wait here for about half the time to smooth mouse
            while time.time() < loop_start_time + 0.5/loop_fps:
                time.sleep(0.001)
                pass
            set_pos(mouse_x_mid + mouse_x_smooth/2, mouse_y_mid + mouse_y_smooth/2,Wd, Hd)


        if IS_DEMO:
            contrast = 1
            img_small = np.clip(128 + contrast * img_small - contrast * 128, 0, 255).astype(int)
            
            bright = 1. # 0.7 to 1.3
            img_small = np.clip(img_small*bright,0, 255).astype(int)

            # or normalise?
            if False:
                img_small = (img_small - np.mean(img_small.flatten())) / np.std(img_small.flatten())
                img_small = (img_small*64+128)
                img_small = np.clip(img_small,0, 255).astype(int)

            img_small = np.array(img_small, dtype='uint8')

            target_width = 800
            scale = target_width / img_small.shape[1] # how much to magnify
            dim = (target_width,int(img_small.shape[0] * scale))
            resized = cv2.resize(img_small, dim, interpolation = cv2.INTER_AREA)



            font = cv2.FONT_HERSHEY_SIMPLEX
            text_loc_1 = (50,50) # x, y
            text_loc_2 = (50,90)
            # text_loc_2a = (50,130)
            # text_loc_2b = (50,170)
            text_loc_3 = (50,130)
            text_loc_4 = (50,170)
            fontScale = 0.8
            fontColor_1 = (20,255,0)  # BGR
            fontColor_2 = (0,0,255) 
            fontColor_3 = (200,200,100) 
            fontColor_4 = (150,20,150) 
            lineType = 2

            text_show_1 = 'mouse_x ' + str(int(mouse_x)) + (5-len(str(int(mouse_x))))*' '
            text_show_1 += ', mouse_y ' + str(int(mouse_y))    + (5-len(str(int(mouse_y))))*' '
            text_show_2 = 'fire ' + str(round(clicks_pred[0],3))
            text_show_2a = 'jump ' + str(round(keys_pred[4:5][0],3))
            text_show_2b = 'reload ' + str(round(keys_pred[n_keys-1:n_keys][0],3))
            text_show_3 = 'keys '
            for char in keys_pressed:
                if char not in ['shift','1','2','3','ctrl']:
                    text_show_3 += char + ' '
            text_show_4 = 'value fn ' + str(round(val_pred,3))

            SHOW_VAL=False
            if SHOW_VAL:
                # trace of value fn
                val_trace_len = np.minimum(len(recent_val),64) # length to use
                val_trace_dist = 100
                (x1, y1) = (300, 180)
                (x2, y2) = (int(x1+val_trace_dist/val_trace_len), int(180 - recent_val[-val_trace_len + i]*30))
                for i in range(val_trace_len):
                    (x1, y1) = (x2, y2)
                    (x2, y2) = (int(x1+val_trace_dist/val_trace_len), int(180 - recent_val[-val_trace_len + i]*30))
                    cv2.line(resized, (x1, y1), (x2, y2), fontColor_4, thickness=2)
                cv2.line(resized, (300, 180), (300, 150), fontColor_4, thickness=4)


            # arrow showing mouse movement
            # (x1, y1) = (int(target_width/2), int(220/1000*target_width))
            (x1, y1) = (int(target_width/2), int(target_width/csgo_img_dimension[1]*csgo_img_dimension[0]/2))
            (x2, y2) = (x1+int(mouse_x_smooth/2), int(y1+mouse_y_smooth/2))
            if Lclicks>0: # change colour if fire
                cv2.arrowedLine(resized, (x1, y1), (x2, y2), (50, 0, 255), thickness=3)
            else:
                cv2.arrowedLine(resized, (x1, y1), (x2, y2), (20, 200, 10), thickness=3)

            # bar showing firing prob
            (x1, y1) = (220, 70)
            (x2, y2) = (int(clicks_pred[0]*100)+220, 90)
            (x2_outline, y2_outline) = (1*100+220, 90)
            cv2.rectangle(resized, (x1, y1), (x2, y2), fontColor_2, -1)
            cv2.rectangle(resized, (x1, y1), (x2_outline, y2_outline), fontColor_2, 2)

            key_offset=0
            if SHOW_VAL:
                key_offset+=20

            # boxes showing key pushes
            (x1, y1) = (70, 170+30-key_offset)
            (x2, y2) = (96, 196+30-key_offset)
            if 'w' in keys_pressed:
                cv2.rectangle(resized, (x1, y1), (x2, y2), (200,200,100)  , 4)
            else:
                cv2.rectangle(resized, (x1, y1), (x2, y2), (0,0,255) , 2)
            (x1, y1) = (40, 200+30-key_offset)
            (x2, y2) = (66, 226+30-key_offset)
            if 'a' in keys_pressed:
                cv2.rectangle(resized, (x1, y1), (x2, y2), (200,200,100)  , 4)
            else:
                cv2.rectangle(resized, (x1, y1), (x2, y2), (0,0,255) , 2)
            (x1, y1) = (70, 200+30-key_offset)
            (x2, y2) = (96, 226+30-key_offset)
            if 's' in keys_pressed:
                cv2.rectangle(resized, (x1, y1), (x2, y2), (200,200,100)  , 4)
            else:
                cv2.rectangle(resized, (x1, y1), (x2, y2), (0,0,255) , 2)
            (x1, y1) = (100, 200+30-key_offset)
            (x2, y2) = (126, 226+30-key_offset)
            if 'd' in keys_pressed:
                cv2.rectangle(resized, (x1, y1), (x2, y2), (200,200,100), 4)
            else:
                cv2.rectangle(resized, (x1, y1), (x2, y2), (0,0,255) , 2)
            
            # add text
            cv2.putText(resized,text_show_1, text_loc_1, font, fontScale,fontColor_1,lineType)
            cv2.putText(resized,text_show_2, text_loc_2, font, fontScale,fontColor_2,lineType)
            # cv2.putText(resized,text_show_2a, text_loc_2a, font, fontScale,fontColor_2,lineType)
            # cv2.putText(resized,text_show_2b, text_loc_2b, font, fontScale,fontColor_2,lineType)
            cv2.putText(resized,text_show_3, text_loc_3, font, fontScale,fontColor_3,lineType)
            if SHOW_VAL:
                cv2.putText(resized,text_show_4, text_loc_4, font, fontScale,fontColor_4,lineType)

            cv2.imshow('resized',resized) 
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

            keys_pressed_tp = key_check()
            if 'Q' in keys_pressed_tp:
                # exit loop
                print('exiting...')
                cv2.destroyAllWindows()
                if IS_GSI:
                    server.server_close()
                break
            if n_loops==1:
                print('\n\npausing align windows')
                time.sleep(15)



        # print('will wait for loop end, ',time.time()-loop_start_time)
        wait_for_loop_end(loop_start_time, loop_fps, n_loops, is_clear_decals=True)
        # time.sleep(0.01)

    print('\n\n -- finished collecting data for iteration', training_iter)
    ReleaseKey(w_char)
    ReleaseKey(a_char)
    ReleaseKey(s_char)
    ReleaseKey(d_char)
    ReleaseKey(r_char)
    ReleaseKey(space_char)
    release_left_click()
    release_right_click()
    time.sleep(0.1)        
    

    pause_game()


    reward_list = pickle.load(open(pickle_reward_path, 'rb'))
    reward_list.append([model_name,training_iter,n_loops,iteration_kills*1000/n_loops,iteration_deaths*1000/n_loops, iteration_kills,iteration_deaths,iteration_kills/np.maximum(iteration_deaths,1)])
    pickle.dump(reward_list, open(pickle_reward_path, 'wb'))
    print('saved pickled rewards',pickle_reward_path)





