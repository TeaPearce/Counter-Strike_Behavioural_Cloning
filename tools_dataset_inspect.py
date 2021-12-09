# this script provides skeleton code for opening up different datasets
# both hdf5 image files and the accompanying meta data

import numpy as np
import matplotlib.pyplot as plt
import h5py

folder_name = '/Volumes/My Passport/2021/csgo_bot_train_july2021/'
n_filer_per_chunk=100
meta_folder = 'dataset_metadata/'

for file_name_stub in ['dm_july2021_', 'dm_july2021_expert_', 'dm_inferno_expert_', 'dm_nuke_expert_', 'dm_mirage_expert_', 'aim_july2021_expert_']:

    for file_chunk in range(0,1):
        dict_chunk = np.load(folder_name+meta_folder+'currvarsv2_'+file_name_stub+str(file_chunk*n_filer_per_chunk+1)+'_to_'+str((file_chunk+1)*n_filer_per_chunk)+'.npy',allow_pickle=True)
        dict_chunk = dict_chunk.item()
    print(dict_chunk['file_num5_frame_200'])


    if file_name_stub == 'dm_july2021_':
        hdf5_folder = 'dataset_dm_scraped_dust2/'
    elif file_name_stub == 'dm_july2021_expert_':
        hdf5_folder = 'dataset_dm_expert_dust2/'
    elif file_name_stub == 'aim_july2021_expert_':
        hdf5_folder = 'dataset_aim_expert/'
    else:
        hdf5_folder = 'dataset_dm_expert_othermaps/'

    file_num = 5
    file_name = folder_name + hdf5_folder + 'hdf5_'+file_name_stub + str(file_num) + '.hdf5'
    h5file = h5py.File(file_name, 'r')

    frame_num = 200
    n_frames = 5
    fig, axs = plt.subplots(n_frames)
    for j in range(n_frames):
        x = h5file['frame_'+str(frame_num+j)+'_x'][:]
        xaux = h5file['frame_'+str(frame_num+j)+'_xaux'][:]
        y = h5file['frame_'+str(frame_num+j)+'_y'][:]
        help_i = h5file['frame_'+str(frame_num+j)+'_helperarr'][:]
        axs[j].imshow(np.flip(x,axis=-1))
    fig.show()


