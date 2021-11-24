# Sample Running Command:
'''Example:
You have a file ABC.ms1 at /DeepIsoV1/rawdata/ABC.ms1
You have run makeDictionary file on this file. Now you want to run this and see the log in output_makeMS1.log file. 
Then the command should be:
nohup python -u DeepIso_v1_preprocess_makeMS1.py ABC > output_makeMS1.log &
'''

from __future__ import division
from __future__ import print_function
from time import time
import pickle
import numpy as np
from collections import defaultdict
import sys
import os
########## file run parameters #################################
current_path=os.system("pwd")
datapath=current_path+'/DeepIsoV1/data/'  
file_name=sys.arg[1]

##############################################################
RT_unit=0.01
mz_unit=0.01
mz_resolution=2

print('scanning test ms: '+file_name)

####################################################################
f=open(datapath+file_name+'_ms1_record', 'rb')
RT_mz_I_dict, maxI=pickle.load(f)
f.close()   
print('ms1 record load done')

###########################
#scan ms1 and record the cnn outputs in list_dict[z]: hash table based on m/z
#for each m/z
RT_list = np.sort(list(RT_mz_I_dict.keys()))
max_RT=RT_list[len(RT_list)-1]
min_RT=10    

sorted_mz_list=[]
for i in range(0, RT_list.shape[0]):
    sorted_mz_list.append(sorted(RT_mz_I_dict[RT_list[i]]))    
    
max_mz=0
min_mz=1000
for i in range (0, len(sorted_mz_list)):
    mz_I_list=sorted_mz_list[i]
    mz=mz_I_list[len(mz_I_list)-1][0]
    if mz>max_mz:
        max_mz=mz
    mz=mz_I_list[0][0]
    if mz<min_mz:
        min_mz=mz


rt_search_index=0
while(RT_list[rt_search_index]<=min_RT):
    if RT_list[rt_search_index]==min_RT:
        break
    rt_search_index=rt_search_index+1 

total_mz=int(round((max_mz-min_mz+mz_unit)/mz_unit, mz_resolution))
total_RT=len(RT_list)-rt_search_index
ms1=np.zeros((total_RT, total_mz))


start_time=time()
y=0
for rt_index in range (rt_search_index, len(RT_list)):
#        print(rt_index)
    mz_start=min_mz
    mz_end=max_mz
    j=0
    while(j<len(sorted_mz_list[rt_index]) and sorted_mz_list[rt_index][j][0]<=mz_start):
        if sorted_mz_list[rt_index][j][0]==mz_start:
            break
        j=j+1

    temp_dict=defaultdict(list)
    while(j<len(sorted_mz_list[rt_index]) and sorted_mz_list[rt_index][j][0]<=mz_end):
        temp_dict[sorted_mz_list[rt_index][j][0]].append(sorted_mz_list[rt_index][j][1])
        j=j+1
    
    temp_dict_keys=list(temp_dict.keys())
#    print('len of mz range %d'%len(temp_dict))
    for k in range (0, len(temp_dict)):
        temp_dict[temp_dict_keys[k]]=np.max(temp_dict[temp_dict_keys[k]])
        x=int(round((temp_dict_keys[k]-min_mz)/mz_unit, mz_resolution))
        ms1[y, x]=temp_dict[temp_dict_keys[k]]
        
    y=y+1

min_I=0
ms1=((ms1-min_I)/(maxI-min_I))*255
time_elapsed=time()-start_time
print(time_elapsed)
    
#try:
#    print('write ms1')
#    f=open(modelpath+dataname[test_index]+'_consecutive_scan_MS1', 'wb') 
#    pickle.dump(ms1, f, protocol=2)
#    print('write done')
#except: 
#    f.close()
#    print('problem in writing ms1')
f=open(datapath+file_name+'_consecutive_scan_MS1_1', 'wb') 
pickle.dump(ms1[:, 0:ms1.shape[1]//2], f, protocol=2)
f.close()

f=open(datapath+file_name+'_consecutive_scan_MS1_2', 'wb') 
pickle.dump(ms1[:, ms1.shape[1]//2:], f, protocol=2)
f.close()    
print('data write done')    


