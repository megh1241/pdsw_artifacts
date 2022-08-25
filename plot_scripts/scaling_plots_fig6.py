import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import matplotlib.patches as mpatches

plt.rcParams["figure.figsize"] = (4, 2)
plt.rcParams.update({'font.size': 7})
x = [1, 2, 4, 8, 16]
titles = ['In-memory DB\n Default params', 'PFS\n Default params', 'PFS\n Large memtable']
time_1_main = [
    [ 516.3406505584717,345.4877519607544,281.26208782196045,254.37904024124146,208.8128218650818 ], 
    [143.2812306880951,105.38065361976624, 68.22132682, 50.36773705482483, 44.69189190864563], 
    [181.79233264923096,113.37270379066467, 85.9153938293457, 70.68430471420288 , 59.97581720352173]
]


time_2_main = [
        [516.3406505584717,  353.06793904304504, 221.75356125831604, 188.1627447605133, 174.32543444633484], 
        [148.1076180934906, 82.44990873336792, 47.74658250808716,36.13772916793823, 35.46236848831177], 
        [186.12420964241028, 121.42337083816528, 88.88848066329956, 77.68923830986023,  65.88510656356812]
]
fig, ax = plt.subplots(nrows=1, ncols=3)

iter1=0
for time_1, time_2 in zip(time_1_main, time_2_main):
    t00 = time_1[0]
    t01 = time_2[0]
    speedup_one_provider = [t00/i for i in time_1]
    speedup_multi_provider = [t01/i for i in time_2]

    ax[iter1].plot(x, speedup_one_provider, linestyle='--', marker='o', color='b', label = "Single Provider")
    ax[iter1].plot(x, speedup_multi_provider, linestyle='--', marker='o', color='r', label = "One provider per client")

    ax[iter1].set_xlabel("# of Clients")
    ax[iter1].set_ylabel("Speedup")
    ax[iter1].set_title(titles[iter1])
    iter1+=1
plt.subplots_adjust(left=0.1, wspace=0.45, hspace=0.45, bottom=0.2, right=0.99, top=0.68)
plt.legend( bbox_to_anchor=(0.1, 1.57), ncol=2)
plt.savefig('scalability.pdf')

