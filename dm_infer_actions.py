import os
import sys
import time
import datetime
import struct
import math
import random

import numpy as np
from matplotlib import pyplot as plt

from config import *

# this script takes a .npy file saved by dm_record_data.py
# and tries to reverse engineer the actions (key presses and mouse movements and mouse clicks)
# that would give rise to the metadata we recorded
# this is added to the .npy file in the format
# [img_small, curr_vars, infer_actions]
# we also set it up in a way that this file can be rerun on the same file
# and will overwrite the old infer_actions
# img_small and curr_vars are never modified here

# also pulls out some checks and high level stats about data collected
# checks eg does
# stats eg weapon usage, team

# also has a helper function at the bottom to help renumber files if needed
# (probably want to delete some due to the per file checks)


file_name_stub = 'dm_sample_' 
folder_name = 'G:/2021/csgo_bot_train_july2021/'

starting_value = 1
highest_num = get_highest_num(file_name_stub, folder_name)
# highest_num = 4020
summary_stats=[]

# for each file of interest
for file_num in range(starting_value,highest_num+1):
    file_name = folder_name+file_name_stub + str(file_num) + '.npy'

    # try to find file
    is_found=False
    for file in os.listdir(folder_name):
        if file_name_stub + str(file_num) + '.npy' == file:
            is_found=True
            break
    if not is_found:
        print('\ncouldnt find file, skipping ', file_name, ' ...')
        continue

    print('\nload train data from ', file_name, ' ...')

    training_data = np.load(file_name, allow_pickle=True)

    # actions will be of format
    # [keys_pressed, mouse_x, mouse_y, press_mouse_l, press_mouse_r]
    # keys_pressed is a list of all keys, e.g. ['w','a','ctrl']
    # mouse_x, y are movements for mouse, floats
    # press_mouse_l, r is whether click left or right mouse, binary
    keys_pressed = []
    mouse_x = 0.0
    mouse_y = 0.0
    press_mouse_l = 0
    press_mouse_r = 0

    # first make an action list assuming nothing will happen
    infer_actions=[]
    for i in range(0,len(training_data)):
        infer_actions.append([[],mouse_x,mouse_y,press_mouse_l,press_mouse_r])
        # had to be careful how create this so can overwrite single elements of it later

    # now loop through data and work out which actions applied
    dead_skipped, skipped = 0,0 # keep track of how many dead
    ram_gsi_disagree_hp, ram_gsi_disagree_ammo = 0,0 # track how many times GSI didn't sync with RAM info
    obs_count = 0 # track how man time obs mode was off
    mouse_static = 0 # how many times didn't move mouse
    vel_static = 0 # how many times didn't move
    for i in range(1,len(training_data)):
        # print(i,end='\r')
        curr_vars = training_data[i][1]
        prev_vars = training_data[i-1][1]


        # first do some checks to see if data is good

        # don't choose any actions if was dead or loading etc
        if True:
            # if curr_vars['obs_health']==0 or prev_vars['obs_health']==0 or not curr_vars['found_active'] or not prev_vars['found_active']:
            if curr_vars['gsi_health']==0 or prev_vars['gsi_health']==0 or not curr_vars['found_active'] or not prev_vars['found_active']:
                if curr_vars['obs_health']==0:
                    dead_skipped+=1
                else:
                    skipped+=1
                continue

        # look for falling out of sync, allowing for one frame of slippage as GSI is slightly delayed often
        if curr_vars['obs_health'] != curr_vars['gsi_health'] and prev_vars['obs_health'] != curr_vars['gsi_health']:
            # print('hp ram',curr_vars['obs_health'] , 'gsi', curr_vars['gsi_health'])
            ram_gsi_disagree_hp+=1

        if curr_vars['ammo_active'] != curr_vars['gsi_ammo'] and prev_vars['ammo_active'] != curr_vars['gsi_ammo']:
        # if np.abs(curr_vars['ammo_active'] - curr_vars['gsi_ammo'])>1:
            # print('ammo ram',curr_vars['ammo_active'], 'gsi', curr_vars['gsi_ammo'] )
            ram_gsi_disagree_ammo+=1

        if curr_vars['obs_mode']!=4:
            obs_count+=1 # count how many times were observing from 3rd person or something -- won't want these

        if np.abs(curr_vars['xy_rad']-prev_vars['xy_rad'])<0.001:
            mouse_static+=1

        if curr_vars['vel_mag']<0.01: 
            vel_static+=1

        # see if we have clean actions, which happens if I recorded it
        if 'tp_wasd' in curr_vars.keys():
            is_clean_wasd=True
        else:
            is_clean_wasd=False

        # see if we have clean actions, which happens if I recorded it
        if 'tp_lclick' in curr_vars.keys():
            is_clean_lclick=True
        else:
            is_clean_lclick=False

        # check for jump
        if is_clean_wasd:
            if 'space' in curr_vars['tp_wasd']:
                infer_actions[i-0][0].append('space')
        else:
            if curr_vars['vel_3'] > prev_vars['vel_3']+150 and not (curr_vars['vel_3'] > -10.0 and curr_vars['vel_3']<10.0):
                infer_actions[i-0][0].append('space')

        # check for crouch
        if  curr_vars['height'] < 50 and curr_vars['height']>0.1: # a weird glitch where at times always zero
            if i>1: 
                infer_actions[i-2][0].append('ctrl')

        # check for weapon switch
        is_switch=False
        if 'type' in curr_vars['gsi_weap_active'].keys() and 'type' in prev_vars['gsi_weap_active'].keys(): # avoid taser
            if curr_vars['gsi_weap_active']['type'] != prev_vars['gsi_weap_active']['type']:
                is_switch=True
                if curr_vars['gsi_weap_active']['type'] == 'Knife':
                    infer_actions[i-1][0].append('3')
                elif curr_vars['gsi_weap_active']['type'] == 'Pistol':
                    infer_actions[i-1][0].append('2')
                else:
                    infer_actions[i-1][0].append('1')

        # weapon switch - this is a bit more reliable than the GSI switch, but gives less info
        is_switch_weap_ram=False
        if curr_vars['itemdef'] != prev_vars['itemdef']: 
            is_switch_weap_ram=True

        # check for scope
        if curr_vars['obs_fov'] != prev_vars['obs_fov'] and not is_switch_weap_ram:
            infer_actions[i-1][4] = 1

        # check for fire
        if is_clean_lclick:
            infer_actions[i-0][3] = curr_vars['tp_lclick']
        else:
            if curr_vars['ammo_active'] < prev_vars['ammo_active'] and not is_switch_weap_ram:
                infer_actions[i-0][3] = 1

        # check for reload
        if is_clean_wasd:
            if 'r' in curr_vars['tp_wasd']:
                infer_actions[i-0][0].append('r')
        else:
            if curr_vars['ammo_active'] > prev_vars['ammo_active'] and not is_switch_weap_ram:
                # this condition only happens some time delta after actually pressed 'r'
                # delta varies per weapon
                # here we use a simple fixed delta, erring on the side of being too early
                # also if in firing, it overrides reload, so check for earliest time when wasnt firing
                if i>32:
                    not_fired=0
                    for j in range(0,10):
                        if infer_actions[i-32+j][3] == 0: # if didn't fire
                            not_fired+=1
                        else:
                            not_fired=0
                        if not_fired > 4: # if didn't fire x times in a row, should be able to reload
                            infer_actions[i-32+j][0].append('r')
                            break

        # predict mouse movement
        zvert_delta = curr_vars['zvert_rads']-prev_vars['zvert_rads']
        xy_delta = curr_vars['xy_rad']-prev_vars['xy_rad']
        if xy_delta < -np.pi: # adjust in case passed true north
            xy_delta = (2*np.pi) + xy_delta
        if xy_delta > np.pi: # adjust in case passed true north other way
            xy_delta = xy_delta - (2*np.pi)
        const_1 = 0.00096 # change in theta angles for one unit of (my) mouse input, w 250 sensitivity
        fov_use = curr_vars['obs_fov']
        if fov_use == 0: fov_use=90
        mouse_x_pred = xy_delta*(90/fov_use)/const_1 
        mouse_y_pred = -zvert_delta*(90/fov_use)/const_1
        infer_actions[i-0][1] = mouse_x_pred
        infer_actions[i-0][2] = mouse_y_pred

        # check for wsad and walk
        # 1) first get velocity relative angle to facing direction
        # player does seems to drift a bit, doesn't directly follow current angle facing
        # slightly smoother if use current velocity relative to previous facing angle
        vel_theta_rel = (2*np.pi - prev_vars['xy_rad'])+curr_vars['vel_theta_abs']
        if vel_theta_rel > 2*np.pi:
            vel_theta_rel = vel_theta_rel - 2*np.pi
        curr_vars['vel_theta_rel'] = vel_theta_rel 
        # this is 0 to 2pi, with 0 meaning velocity is in direction facing (fwds), and pi meaning vel is opposite direction (bwds)

        # 2) get general classification of orientation of vel relative to angle facing
        if (curr_vars['vel_theta_rel']>(8/4)*np.pi-(2*np.pi/16) or curr_vars['vel_theta_rel']<=(0/4)*np.pi+(2*np.pi/16)):
            orient = 'w' # e.g. vel_theta_rel is either close to 2pi or 0, so must be moving forward 
        elif (curr_vars['vel_theta_rel']>(1/4)*np.pi-(2*np.pi/16) and curr_vars['vel_theta_rel']<=(1/4)*np.pi+(2*np.pi/16)):
            orient = 'wd'
        elif (curr_vars['vel_theta_rel']>(2/4)*np.pi-(2*np.pi/16) and curr_vars['vel_theta_rel']<=(2/4)*np.pi+(2*np.pi/16)):
            orient = 'd'
        elif (curr_vars['vel_theta_rel']>(3/4)*np.pi-(2*np.pi/16) and curr_vars['vel_theta_rel']<=(3/4)*np.pi+(2*np.pi/16)):
            orient = 'sd'
        elif (curr_vars['vel_theta_rel']>(4/4)*np.pi-(2*np.pi/16) and curr_vars['vel_theta_rel']<=(4/4)*np.pi+(2*np.pi/16)):
            orient = 's'
        elif (curr_vars['vel_theta_rel']>(5/4)*np.pi-(2*np.pi/16) and curr_vars['vel_theta_rel']<=(5/4)*np.pi+(2*np.pi/16)):
            orient = 'as'
        elif (curr_vars['vel_theta_rel']>(6/4)*np.pi-(2*np.pi/16) and curr_vars['vel_theta_rel']<=(6/4)*np.pi+(2*np.pi/16)):
            orient = 'a'
        elif (curr_vars['vel_theta_rel']>(7/4)*np.pi-(2*np.pi/16) and curr_vars['vel_theta_rel']<=(7/4)*np.pi+(2*np.pi/16)):
            orient = 'wa'
        else:
            orient=''

        # 3) now make classification based on acceleration (vel_mag_diff), absolute vel, and orientation
        # there are lots of quirks with this, and I haven't coded up all edge cases, but prob good enough
        vel_mag_diff = curr_vars['vel_mag'] - prev_vars['vel_mag']

        if is_clean_wasd:
            if 'w' in curr_vars['tp_wasd']:
                infer_actions[i-0][0].append('w')
            if 'a' in curr_vars['tp_wasd']:
                infer_actions[i-0][0].append('a')
            if 's' in curr_vars['tp_wasd']:
                infer_actions[i-0][0].append('s')
            if 'd' in curr_vars['tp_wasd']:
                infer_actions[i-0][0].append('d')
        else:
            # case 1) motionless
            if curr_vars['vel_mag']<0.1: 
                pass
            # case 2) stable running at const speed
            elif np.abs(vel_mag_diff)<7: # was 0.1
                if 'w' in orient:
                    infer_actions[i-0][0].append('w')
                if 's' in orient:
                    infer_actions[i-0][0].append('s')
                if 'a' in orient:
                    infer_actions[i-0][0].append('a')
                if 'd' in orient:
                    infer_actions[i-0][0].append('d')

                # think about whether shifted or not
                # if crouched, shift has no effect, so don't check
                if 'ctrl' not in infer_actions[i-0][0]:
                    # moves slowly when scoped with awp, even if not holding shift
                    if curr_vars['gsi_weap_active']['name'] == 'weapon_awp' and curr_vars['obs_scope']==1:
                        if curr_vars['vel_mag'] < 60:
                            infer_actions[i-0][0].append('shift')
                    elif curr_vars['vel_mag'] < 140: # 140 a rough threshold for walking, works with most weapons
                        infer_actions[i-0][0].append('shift')
                        # this is not perfect, doesn't capture when accelerating walking

            # case 3) accelerating
            elif curr_vars['vel_mag']>prev_vars['vel_mag']: 
                if 'w' in orient:
                    infer_actions[i-0][0].append('w')
                if 's' in orient:
                    infer_actions[i-0][0].append('s')
                if 'a' in orient:
                    infer_actions[i-0][0].append('a')
                if 'd' in orient:
                    infer_actions[i-0][0].append('d')

            # case 4) decelerating     
            elif curr_vars['vel_mag']<prev_vars['vel_mag']: 
                # not perfect, sometimes says counterstrafe when didnt but I don't think it's a big problem
                # anyway I'd rather add in extra counterstrafes than too few 
                # also can have large decel when run into walls, objects etc
                if vel_mag_diff < -70: # to acheive this mag of decel, need to be hitting opposite direction
                    if 'w' in orient:
                        infer_actions[i-0][0].append('s')
                    if 's' in orient:
                        infer_actions[i-0][0].append('w')
                    if 'a' in orient:
                        infer_actions[i-0][0].append('d')
                    if 'd' in orient:
                        infer_actions[i-0][0].append('a')
                else:
                    # slowing via natural friction
                    pass

    # can ignore this bit if only running for analytics
    if True:
        # do a quick clean up on wasd, since sometimes gaps appear when they should be pushed
        for i in range(1,len(training_data)-3):
            for c in ['w','s','a','d']:
                if c not in infer_actions[i][0] and c in infer_actions[i-1][0] and c in infer_actions[i+2][0]:
                    infer_actions[i][0].append(c)
                

        # all actions have been inferred
        # repackage into correct format and save new training data
        new_training_data = []
        for i, data in enumerate(training_data):
            img_small = data[0]
            curr_vars = data[1]
            infer_a = infer_actions[i]

            if False: # paint over a cross hair -- used for one chunk of data I recorded with incorrect crosshair setting
                cross_rgb = [46,250,42]
                img_small[71,141,:] = cross_rgb
                img_small[77,141,:] = cross_rgb
                img_small[74,137:139,:] = cross_rgb
                img_small[74,144,:] = cross_rgb

            new_training_data.append([img_small,curr_vars,infer_a])
            # new_training_data.append([[],curr_vars,infer_a]) # use this for tracking
        np.save(file_name,new_training_data)
        print('SAVED', file_name)

    print('dead_skipped',dead_skipped)
    print('skipped',skipped)
    print('ram_gsi_disagree_ammo',ram_gsi_disagree_ammo) # can be in the 10s
    print('ram_gsi_disagree_hp',ram_gsi_disagree_hp) # can be in the 10s
    print('obs_count',obs_count) # should be zero really
    print('mouse_static',mouse_static) # 500, 600's can be normal
    print('vel_static',vel_static) 
    if dead_skipped>200 or ram_gsi_disagree_ammo>20 or ram_gsi_disagree_hp>20 or obs_count>0 or mouse_static>650 or vel_static>350:
        print('-- bad data file? --\n\n\n\n')
        # could automate it's deletion here?

    summary_stats.append([file_num,dead_skipped,skipped, ram_gsi_disagree_ammo,ram_gsi_disagree_hp,obs_count,mouse_static,vel_static ])

summary_stats_np = np.array(summary_stats)

fig, ax = plt.subplots()
ax.plot(summary_stats_np[:,0], summary_stats_np[:,1]/1000,label='dead_skipped')
ax.plot(summary_stats_np[:,0], summary_stats_np[:,3]/1000,label='ram_gsi_disagree_ammo')
ax.plot(summary_stats_np[:,0], summary_stats_np[:,5],label='obs_count')
ax.plot(summary_stats_np[:,0], summary_stats_np[:,6]/1000,label='mouse_static')
ax.plot(summary_stats_np[:,0], summary_stats_np[:,7]/1000,label='vel_static')
ax.legend()
ax.set_title('summary stats')
fig.show()



# because I delete some of these files, it's useful to automate a renaming process
# so I end up with a consecutive set of files
if False:
    from pathlib import Path
    file_name_stub = 'dm_july2021_' 
    folder_name = 'G:/2021/csgo_bot_train_july2021/02_unprocessed/'
    # folder_name = 'G:/2021/csgo_bot_train_july2021/03_scraped/'
    highest_num = get_highest_num(file_name_stub, folder_name)
    for file_num in range(5001,highest_num+1):
    # for file_num in range(2612,highest_num+1):
        file_name = folder_name+file_name_stub + str(file_num) + '.npy'

        # try to find file
        is_found=False
        file_check = Path(file_name)
        if file_check.exists():
            is_found=True

        if not is_found:
            # going to try to find next highest number and relabel to this
            for file_num2 in range(file_num,highest_num+1):
                file_name2 = folder_name+file_name_stub + str(file_num2) + '.npy'

                # should be a quicker command to find if a file exists?
                is_found=False
                file_check = Path(file_name2)
                if file_check.exists():
                    is_found=True

                if is_found:
                    # rename this file to earlier one
                    os.rename(file_name2,file_name)
                    print('\nrenaming', file_name2, 'to', file_name)
                    break
            continue


if False:
    # to shift all file numbers by a fixed offset
    from pathlib import Path
    file_name_stub = 'dm_july2021_' 
    folder_name = 'G:/2021/csgo_bot_train_july2021/03_scraped/01_rename/'
    highest_num = get_highest_num(file_name_stub, folder_name)
    for file_num in range(2675,highest_num+1):
        file_name = folder_name+file_name_stub + str(file_num) + '.npy'
        file_name2 = folder_name+file_name_stub + str(file_num-2675+3143) + '.npy'

        is_found=False
        file_check = Path(file_name)
        if file_check.exists():
            is_found=True

        if is_found:
            # rename this file to earlier one
            os.rename(file_name,file_name2)
            print('\nrenaming', file_name, 'to', file_name2)











