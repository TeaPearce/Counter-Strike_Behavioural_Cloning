import os
import time
import cv2
import h5py

import numpy as np

# opens up sections of training data, displaying image and optionally actions
# can just view or also save the images to disk
# select window and press 's' to go one frame forward, 'q' to quit

# list of frames to view or save
file_frame_ids = {}
file_frame_ids['10'] = list(range(14,43))  
file_frame_ids['13'] = list(range(1,300))  
file_frame_ids['16'] = list(range(10,32))  


save_path = '/Users/timpearce/Google Drive/Google Drive all/05. misc geeky/03_csgo_bot/01_writeup/images/01_train_egs_hi_res/'

hdf5_file_stub = '/Volumes/My Passport/2021/csgo_bot_train_july2021/dataset_dm_scraped_dust2/hdf5_dm_july2021_'
metadata_file_stub = '/Volumes/My Passport/2021/csgo_bot_train_july2021/dataset_metadata/currvarsv2_dm_july2021_'

IS_OVERLAY=True # whether to print actions over top
is_save=False
is_view=True


for file_id in file_frame_ids.keys():
    file_name = hdf5_file_stub + file_id + '.hdf5'
    # training_data = np.load(file_name, allow_pickle=True)
    h5file = h5py.File(file_name, 'r')

    if IS_OVERLAY:
        # open .npy metadata file
        # file_id=201
        n_filer_per_chunk=100
        meta_file_name = metadata_file_stub+str(int(math.floor((int(file_id)-1)/n_filer_per_chunk))*n_filer_per_chunk+1)+'_to_'+str(int(math.floor((int(file_id)-1)/n_filer_per_chunk)+1)*n_filer_per_chunk)+'.npy'
        dict_chunk = np.load(meta_file_name,allow_pickle=True)
        dict_chunk = dict_chunk.item()

    n_loops=0
    for frame_id in file_frame_ids[file_id]:
        frame_id = int(frame_id)
        img = h5file['frame_'+str(frame_id)+'_x'][:]


        target_width = 1000
        scale = target_width / img.shape[1] # how much to magnify
        dim = (target_width,int(img.shape[0] * scale))
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

        if IS_OVERLAY:

            dict_i = dict_chunk['file_num'+str(file_id)+'_frame_'+str(frame_id)]

            curr_vars = dict_i[0]
            [keys_pressed,mouse_x,mouse_y,press_mouse_l,press_mouse_r] = dict_i[1]
            helper_i = dict_i[2]

            print(frame_id, 'ram',curr_vars['obs_health'], 'gsi',curr_vars['gsi_health'], 
                           'ram ammo',curr_vars['ammo_active'],'gsi ammo',curr_vars['gsi_ammo'] if 'gsi_ammo' in curr_vars.keys() else 0,
                           'kill',curr_vars['gsi_kills'],
                           'death',curr_vars['gsi_deaths'],
                           helper_i,
                           keys_pressed)

            # add action as text overlay
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_loc = (20,50) # x, y
            text_loc_pred = (40,100) # x, y coords of NN predictions 
            fontScale = 0.8
            fontColor = (20,255,0)  # BGR
            fontColor_pred = (0,0,255)  # BGR
            lineType = 2

            text_show = 'TRUE: mousex ' + str(int(mouse_x)) + (5-len(str(int(mouse_x))))*' '
            text_show += ', mousey ' + str(int(mouse_y))    + (5-len(str(int(mouse_y))))*' '
            text_show += ', l click ' + str(int(press_mouse_l)) + (2-len(str(int(press_mouse_l))))*' '
            text_show += ', r click ' + str(int(press_mouse_r)) + (2-len(str(int(press_mouse_r))))*' '
            text_show += ', keys: '
            for char in keys_pressed:
                text_show += char + ' '

            cv2.putText(resized,text_show, text_loc, 
                font, fontScale,fontColor,lineType)

        if is_save:
            save_name = save_path + 'train_data_file' + file_id + '_frame' + str(frame_id) +'.png'
            
            contrast = 1.2
            brightness = 5
            img = cv2.addWeighted(img, contrast, img, 0, brightness)
            cv2.imwrite(save_name,img) # show expanded so can see stuff...
            # cv2.imwrite(save_name,resized) # show original resolution
            print('saved',save_name)

        if is_view:
            cv2.imshow('resized',resized)

            is_exit_prog=False
            while True:
                loop_ctrl_key = cv2.waitKey(0)
                # go next time step if press 's'
                if loop_ctrl_key & 0xFF == ord('s'):
                    n_loops += 1 # 1
                    break
                # press 'q' to exit
                elif loop_ctrl_key & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    is_exit_prog=True
                    break
                time.sleep(0.0625)
            if is_exit_prog: break

    cv2.destroyAllWindows()

