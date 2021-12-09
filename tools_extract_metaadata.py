# just scrape meta part from numpy files
# after running infer actions 
# get [img_small,curr_vars,infer_a
# and pretrain_preprocess,
# get [img_small,curr_vars,infer_a,y_train_i,x_aux_i,helper_arr_i]
# and  infer_a is format [keys_pressed, mouse_x, mouse_y, press_mouse_l, press_mouse_r] i
# while curr_vars is a general dictionary

import os
import numpy as np
from pathlib import Path

# file_name_stub = 'agentj22_capture' 
# file_name_stub = 'agentj22_dmexpert20_capture' 
# file_name_stub = 'bot_capture_' 
# file_name_stub = 'dm_july2021_' 
# file_name_stub = 'dm_july2021_expert_' 
# file_name_stub = 'aim_july2021_expert_' 
# file_name_stub = 'dm_inferno_expert_' 
# file_name_stub = 'dm_nuke_expert_' 
file_name_stub = 'dm_mirage_expert_' 

# folder_name = 'G:/2021/csgo_bot_train_july2021/'
# folder_name = '/Volumes/My Passport/2021/csgo_bot_train_july2021/'
# folder_name = '/Volumes/My Passport/2021/csgo_bot_train_july2021/03_dm_expert/'
# folder_name = '/Volumes/My Passport/2021/csgo_bot_train_july2021/04_aim_expert/'
folder_name = '/Volumes/My Passport/2021/csgo_bot_train_july2021/06_othermaps/'
# folder_name = '/Volumes/My Passport/2021/csgo_bot_train_july2021/05_trackings/'
# highest_num = get_highest_num(file_name_stub, folder_name)

# for file_chunk in range(0,40):
for file_chunk in range(0,1):
    save_dict={}
    n_filer_per_chunk=100
    for file_num in range(file_chunk*n_filer_per_chunk+1,(file_chunk+1)*n_filer_per_chunk+1):
        print('file_num',file_num,end='\r')
        file_name = folder_name+file_name_stub + str(file_num) + '.npy'

        # try to find file
        is_found=False
        file_check = Path(file_name)
        if file_check.exists():
            is_found=True
        else:
            print('\ncouldnt find file, skipping ', file_name, ' ...')
            continue

        training_data = np.load(file_name, allow_pickle=True)

        for i in range(0,len(training_data)):
            curr_vars = training_data[i][1]
            infer_a = training_data[i][2]
            helper_arr_i = training_data[i][-1] # kill, death flag
            save_dict['file_num'+str(file_num)+'_frame_'+str(i)] = [curr_vars, infer_a, helper_arr_i]


    np.save(folder_name+'currvarsv2_'+file_name_stub+str(file_chunk*n_filer_per_chunk+1)+'_to_'+str((file_chunk+1)*n_filer_per_chunk)+'.npy', save_dict)
    print('SAVED')


