import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 16})

save_path = '/Users/timpearce/Google Drive/Google Drive all/05. misc geeky/03_csgo_bot/01_writeup/images/aaaishots/'
# figsize_in = (5,2.5) # x, y width

is_save=False

file_name_stub = 'dm_july2021_' 
# folder_name = 'G:/2021/csgo_bot_train_july2021/'
# folder_name = '/Volumes/My Passport/2021/csgo_bot_train_july2021/'
folder_name = '/Volumes/My Passport/2021/csgo_bot_train_july2021/dataset_metadata/'
n_filer_per_chunk=100
info_array = []
weap_arr=[]
weap_type_arr=[]
for file_chunk in range(0,3):
# for file_chunk in range(0,55):
    print('file_chunk',file_chunk,end='\r')
    dict_chunk = np.load(folder_name+'currvarsv2_'+file_name_stub+str(file_chunk*n_filer_per_chunk+1)+'_to_'+str((file_chunk+1)*n_filer_per_chunk)+'.npy',allow_pickle=True)
    dict_chunk = dict_chunk.item()

    # file_num1000_frame_0
    # file_num1000_frame_999
    # use this to make sure go through in order
    prev_mouse = (0,0)
    for file_i in range(file_chunk*n_filer_per_chunk+1,(file_chunk+1)*n_filer_per_chunk+1):
        for frame_i in range(0,1000):
            key = 'file_num' + str(file_i) +'_frame_' + str(frame_i)

            dict_key = dict_chunk[key]
            mousex = dict_key[1][1]
            mousey = dict_key[1][2]
            pos1 = dict_key[0]['localpos1'] 
            pos2 = dict_key[0]['localpos2'] 
            pos3 = dict_key[0]['localpos3'] 
            kill_flag = dict_key[2][0]  
            death_flag = dict_key[2][1]  

            if 'gsi_weap_active' in dict_key[0].keys():
                weap_arr.append(dict_key[0]['gsi_weap_active']['name'])
                if 'taser' not in dict_key[0]['gsi_weap_active']['name']:
                    weap_type_arr.append(dict_key[0]['gsi_weap_active']['type'])
                else:
                    weap_type_arr.append('taser')
            else:
                weap_arr.append('none found')
                weap_type_arr.append('none found')
            if 'ak47' in weap_arr[-1]:
                ak_flag=1
            else:
                ak_flag=0
            # if (mousex,mousey) == prev_mouse and mousex != 0:
            if mousex < prev_mouse[0]*1.1 and mousex > prev_mouse[0]*0.9 and mousex != 0:
                same_mouse_flag = 1
            else:
                same_mouse_flag = 0
            info_array.append([pos1,pos2,pos3,mousex,mousey,kill_flag,death_flag,ak_flag,same_mouse_flag])
            prev_mouse = (mousex,mousey)

info_array = np.array(info_array)

kill_arr = info_array[info_array[:,5]==1]
death_arr = info_array[info_array[:,6]==1]
ak_arr = info_array[info_array[:,7]==1]
not_ak_arr = info_array[info_array[:,7]!=1]
same_mouse_arr = info_array[info_array[:,8]==1]

# info_array = ak_arr # could overwrie
# info_array = not_ak_arr 

print('total frames',info_array.shape[0])
print('total kills',info_array[:,5].sum())
print('total deaths',info_array[:,6].sum())

print('total ak frames',info_array[:,7].sum())
print('total ak kills',ak_arr[:,5].sum())
print('total ak deaths',ak_arr[:,6].sum())
print('mean ak kills',ak_arr[:,5].mean())
print('mean ak deaths',ak_arr[:,6].mean())

weap_count_dict = {}
for w in weap_arr:
    if w in weap_count_dict.keys():
        weap_count_dict[w] +=1
    else:
        weap_count_dict[w] = 1

type_count_dict = {}
for w in weap_type_arr:
    if w in type_count_dict.keys():
        type_count_dict[w] +=1
    else:
        type_count_dict[w] = 1

knife_total=0
for w in weap_count_dict.keys():
    if 'knife' in w:
        knife_total+=weap_count_dict[w]
        weap_count_dict[w]=0
weap_count_dict['weapon_knives'] = knife_total

total=0
for w in weap_count_dict.keys():
    if 'm4a1' in w:
        total+=weap_count_dict[w]
        weap_count_dict[w]=0
weap_count_dict['weapon_m4a1'] = total


weap_count_arr = []
weap_count_name = []
for w in weap_count_dict.keys():
    if w == 'none found':
        pass
    else:
        weap_count_name.append(w[7:])
        weap_count_arr.append(weap_count_dict[w])
weap_count_arr = np.array(weap_count_arr)
weap_count_name = np.array(weap_count_name)
ids_order = np.argsort(weap_count_arr)      
weap_count_arr = weap_count_arr[ids_order]
weap_count_name = weap_count_name[ids_order]
weap_count_arr = weap_count_arr/weap_count_arr.sum()





fig, ax = plt.subplots(1,1,figsize=(8,8))
# ax.scatter(info_array[:100000,0],info_array[:100000,1],alpha=0.02,s=10,lw=0.,color='magenta')
# ax.scatter(info_array[:,0],info_array[:,1],alpha=0.005,s=5,lw=0.,color='magenta')
ax.scatter(info_array[:2000000,0],info_array[:2000000,1],alpha=0.005,s=4,lw=0.,color='magenta')
# ax.scatter(death_arr[:,0],death_arr[:,1],alpha=0.1,s=40,lw=0.,color='red')
# ax.scatter(kill_arr[:,0],kill_arr[:,1],alpha=0.1,s=40,lw=0.,color='blue')
ax.set_xlabel('Player coords x')
ax.set_ylabel('Player coords y')
ax.set_xticks([])
ax.set_yticks([])
fig.show()
name = save_path + 'dataset_trajs'
if is_save:
    fig.savefig(name+'.png', format='png', dpi=500, bbox_inches='tight')


fig, ax = plt.subplots(1,1)
h = ax.hist2d(info_array[:100000,0],info_array[:100000,1],bins=30,cmap='jet',norm=mpl.colors.LogNorm())
fig.show()
fig.colorbar(h[3], ax=ax)


fig, ax = plt.subplots(1,1,figsize=(7,3))
ax.hist(np.clip(info_array[:,3],-380,380),alpha=1,bins=200, edgecolor=None,color='deepskyblue',density=True,histtype='bar',zorder=2)
ax.hist(np.clip(info_array[:,3],-380,380),alpha=1,bins=200, edgecolor=None,color='k',density=True,histtype='step',lw=1,zorder=3)
for x in [ -300.0, -200.0, -100.0, -60.0, -30.0, -20.0, -10.0, -4.0, -2.0, -0.0, 2.0, 4.0, 10.0, 20.0, 30.0, 60.0, 100.0, 200.0, 300.0]:
    ax.axvline(x,zorder=1,color='k',lw=0.5,ls='--')
ax.set_xlabel('Mouse x')
ax.set_ylabel('Density')
ax.set_xlim((-350,350))
ax.set_yticks([])
fig.show()
name = save_path + 'dataset_mousex'
if is_save:
    fig.savefig(name+'.pdf', format='pdf', dpi=1000, bbox_inches='tight')

fig, ax = plt.subplots(1,1,figsize=(7,3))
ax.hist(np.clip(info_array[:,4],-380,380),alpha=1,bins=200, edgecolor=None,color='deepskyblue',density=True,histtype='bar',zorder=2)
ax.hist(np.clip(info_array[:,4],-380,380),alpha=1,bins=200, edgecolor=None,color='k',density=True,histtype='step',lw=1,zorder=3)
for x in [-100.0, -50.0, -20.0, -10.0, -4.0, -2.0, -0.0, 2.0, 4.0, 10.0, 20.0, 50.0, 100.0]:
    ax.axvline(x,zorder=1,color='k',lw=0.5,ls='--')
ax.set_xlabel('Mouse y')
ax.set_ylabel('Density')
ax.set_xlim((-350,350))
ax.set_yticks([])
fig.show()
name = save_path + 'dataset_mousey'
if is_save:
    fig.savefig(name+'.pdf', format='pdf', dpi=1000, bbox_inches='tight')



nshow=25
weap_count_arr[-nshow] = weap_count_arr[:-nshow].sum()
weap_count_name[-nshow] = 'other'
fig, ax = plt.subplots(1,1,figsize=(8,8))
ax.grid(which='both',color='k', linestyle='--', linewidth=1,alpha=0.2,axis='x',markevery=0.05,zorder=1)
ax.barh(np.arange(len(weap_count_arr[-nshow:])),  weap_count_arr[-nshow:], align='center', edgecolor='black',color='deepskyblue',zorder=2)
ax.set_yticks(np.arange(len(weap_count_arr[-nshow:])))
ax.set_yticklabels(weap_count_name[-nshow:])
ax.set_xlabel('Proportion of time equipped')
fig.show()
name = save_path + 'dataset_equip'
if is_save:
    fig.savefig(name+'.pdf', format='pdf', dpi=1000, bbox_inches='tight')


