import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import matplotlib.patches as mpatches
import math

profile_key_index_map = {
    'Write RDMA init': (0,1),
    'Write RDMA transfer': (1,2),
    'Read RDMA init': (7,8),
    'Read RDMA transfer': (8,9),
}



hatch_list =  ['---', '...', '***', "///", '---', 'ooo']

store_keys =  ['Write RDMA init', 'Write RDMA transfer' ]
transfer_keys = [ 'Read RDMA init', 'Read RDMA transfer']

color_list_store = ['yellow', 'cyan' ]
color_list_transfer = ['brown', 'violet'  ]

color_dict_store = dict(zip(store_keys, color_list_store))
color_dict_transfer = dict(zip(transfer_keys, color_list_transfer))

def generate_patches_for_legend(color_dict_store, color_dict_transfer, hatch_dict, case_label_dict):
    patches_list_store = []
    for key, val in color_dict_store.items():
        patches_list_store.append(mpatches.Patch(color=val, label=key))

    patches_list_transfer = []
    for key, val in color_dict_transfer.items():
        patches_list_transfer.append(mpatches.Patch(color=val, label=key))

    patches_list_setting = []
    for key, val in hatch_dict.items():
        patch = mpatches.Patch(hatch=val, label=case_label_dict[key])
        patch.set_facecolor('white')
        patches_list_setting.append(patch)
    
    return patches_list_store,  patches_list_transfer,  patches_list_setting


def read_populate_main_bar_dict(percent_layers, cases,  action):
    bar_main_dict = {}
    for case_iter, item in enumerate(cases):
        bar_dict = {}
        for i in keys:
            bar_dict[i] = []
        dir_path = os.path.join(head_dir, item)
        for perc_iter, percent in enumerate(percent_layers):
            filename = action + '_' + tensor_size + '_' + str(percent) + '.npy'
            file_path = os.path.join(dir_path, filename)
            loaded = np.load(file_path, 'r')
            mean = loaded[0]
            std = loaded[1]
            siz = len(mean)
            for key in keys:
                start_pt = profile_key_index_map[key][0]
                end_pt = profile_key_index_map[key][1]
                bar_dict[key].append(mean[end_pt] - mean[start_pt])
        bar_main_dict[item] = bar_dict

    return bar_main_dict



plt.rcParams["figure.figsize"] = (8,2)


parser = argparse.ArgumentParser()
parser.add_argument("--action", type=str, default='store')
parser.add_argument("--size", type=str, default='small')
args = parser.parse_args()
action = args.action
tensor_size = args.size


head_dir = '/Users/admin/datastates'

cases = ['pdsw_res']
case_labels = ['pdsw_res']
#cases = ['final_memtable', 'final_default', 'final_devshm']#, 'rocksdb_defaultbuffer']
#case_labels = ['Large Memtable', 'Default: PFS', 'Default: In Memory']#, 'large blockcache']

case_label_dict = dict(zip(cases, case_labels))
percent_layers = [25, 50, 75, 100]
x_main = [0, 1, 2, 3]


hatch_dict = {}
for case_iter, item in enumerate(cases):
    hatch_dict[item] = hatch_list[case_iter]

fig,ax = plt.subplots(1, 4)
sizes = ['small', 'large']
gb_size = {'small': '128MB', 'large':'1GB'}
actions = ['store', 'transfer']
action_labels = {'store':'Write', 'transfer':'Read'}


main_iter=0
for size_iter, tensor_size in enumerate(sizes):
    for action_iter, action in enumerate(actions):
        action_label = action_labels[action]
        keys = []
        actioned = ''
        color_dict = {}
        if action == 'store':
            keys = store_keys
            actioned = 'written'
            color_dict = color_dict_store
        else:
            keys = transfer_keys
            actioned = 'read'
            color_dict = color_dict_transfer
        
        bar_main_dict = read_populate_main_bar_dict(percent_layers, cases, action)

        width = 0.2
        jiter=0
        plot_handles = {}
        num_keys = len(keys)
        for case_iter, item in enumerate(cases):
            bar_dict = bar_main_dict[item]
            x = [a + jiter for a in x_main]
            jiter+=0.2
            x_len = len(x)
            curr_height = [0]*x_len
            for i in range(num_keys): 
                ax[main_iter].bar(
                    x, 
                    bar_dict[keys[i]],  
                    bottom=curr_height, width=width, 
                    color=color_dict[keys[i]], align='center', 
                    data=None, hatch=hatch_dict[item], 
                    edgecolor='black'
                )
                for j in range(x_len):
                    curr_height[j] += bar_dict[keys[i]][j] 


        xlabels =[str(percent) + '%' for percent in percent_layers]
        ax[main_iter].set_xticks(x_main)
        ax[main_iter].set_xticklabels(xlabels)
        ax[main_iter].tick_params(axis='both', which='minor', labelsize=7)
        ax[main_iter].tick_params(axis='both', which='minor', labelsize=7)
        ax[main_iter].tick_params(axis='both', which='major', labelsize=7)
        ax[main_iter].tick_params(axis='both', which='major', labelsize=7)
        ax[main_iter].set_title(action_label + ' '+ gb_size[tensor_size]+' Model' , fontsize=8)
        main_iter+=1


fig.supylabel('Time(ms)', fontsize=7)
patches = generate_patches_for_legend(color_dict_store, color_dict_transfer, hatch_dict, case_label_dict)
fig.legend(handles=patches[0],  fontsize=6,  title='Write',   title_fontsize='x-small',   bbox_to_anchor=(0.95, 0.996))
fig.legend(handles=patches[1],  fontsize=6, title='Read',ncol=1,   title_fontsize='x-small',  bbox_to_anchor=(0.98, 0.68))
fig.legend(handles=patches[2],  fontsize=6,  title_fontsize='x-small', bbox_to_anchor=(0.95, 0.24))
fig.supxlabel("Fraction of layers read/written", fontsize=7)
plt.subplots_adjust(left=0.056, hspace=0, wspace=0.26, bottom=0.2, right=0.8)
plt.savefig('all_timings.pdf')  
plt.show()
