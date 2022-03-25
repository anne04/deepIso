# Sample Running Command:
'''Example:
You have a raw data file ABC.ms1 at /DeepIsoV1/rawdata/ABC.ms1
You want to run this over that file and see the log in output_makeDictionary.log file. 
Then the command should be:
nohup python -u DeepIso_v1_preprocess_makeDictionary.py ABC > output_makeDictionary.log &
'''

from __future__ import division
from __future__ import print_function
import pickle
import sys
from collections import defaultdict
import os
########## file run parameters #################################
current_path=os.system("pwd")
datapath=current_path+'/DeepIsoV1/data/' 
rawdata=current_path+'/DeepIsoV1/rawdata/'
modelpath=current_path+'/DeepIsoV1/model/'
file_name=sys.argv[1]

#################################################################
print('reading raw data for %s'%file_name)
f = open(rawdata+file_name+'.ms1', 'r') 
line=f.readline()
RT_mz_I_dict=defaultdict(list)
i=0
j=0
maxI=0
max_mz=0
mz_resolution=3
while line!='':
    if line.find('RTime')>=0:
        temp=line.split('\t')
        temp=temp[len(temp)-1]
        temp=temp.split('\n')
        temp=temp[len(temp)-2]
        RT_value=round(float(temp), 2)       #?
        line=f.readline()  
        line=f.readline()    
        line=f.readline()  
        line=f.readline() 
        j=0
#            print('found RT')
        while line!='' and line.find('S')<0:
            temp=line.split(' ')
            #print(temp[0])
            mz_value=round(float(temp[0]), mz_resolution) 
            temp=temp[1].split('\n')
            temp=temp[len(temp)-2]
            intensity_value=round(float(temp), 4)            
            if maxI<intensity_value:
               maxI=intensity_value 
            RT_mz_I_dict[RT_value].append((mz_value, intensity_value))
            line=f.readline()  
            j=j+1
        i=i+1
    if line!='':   
        line=f.readline()    
f.close()


print('trying to save MS1 record')
f=open(datapath+file_name+'_ms1_record', 'wb')
pickle.dump([RT_mz_I_dict, maxI], f, protocol=2)
f.close()   
print('done!')
