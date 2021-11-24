# Sample Running Command: nohup python -u DeepIso_v1_reportFeature_isoGroup.py <rawDataFileName without extension> <batch_size> > <logfile> &
'''Example: 
You have a file ABC.ms1 at /DeepIsoV1/rawdata/ABC.ms1
You want to run IsoGrouping on this file. 
You allow batch size 500 to be processed at a time by this module. 
You want to see the log in output_isoGrouping.log file. 
Then you should put the command as below:
nohup python -u DeepIso_v1_reportFeature_isoGroup.py ABC 500 > output_isoGrouping.log &
'''

from __future__ import division
from __future__ import print_function
from time import time
import pickle
import numpy as np
from collections import defaultdict
import sys
import math
import model_generation
import os
########## file run parameters #################################
current_path=os.system("pwd")
datapath=current_path+'/DeepIsoV1/data/'  
modelpath=current_path+'/DeepIsoV1/model/'
file_name=sys.arg[1]
batch_size=int(sys.arg[2]) #500

####################### scannnig parameter ##########
isotope_gap=np.zeros((10))
isotope_gap[0]=0.01
isotope_gap[1]=1.00
isotope_gap[2]=0.50
isotope_gap[3]=0.33
isotope_gap[4]=0.25
isotope_gap[5]=0.20
isotope_gap[6]=0.17
isotope_gap[7]=0.14
isotope_gap[8]=0.13
isotope_gap[9]=0.11
RT_window=15
mz_window=11
frame_width=11


################### deep learning parameter #########

truncated_backprop_length = 5
fc_size = 128
num_epochs= 200
total_frames_hor=truncated_backprop_length
total_hops_horizontal= total_frames_hor//truncated_backprop_length 
num_class=total_frames_hor # number of isotopes to report
drop_out_k=0.5

mz_unit=0.01
RT_unit=0.01



learn_rate=[]
state_size =[]
model_no=[]
fc_size=[]
#
model_no.append('cnn_rnn_isoGrouping_attention_v2_lrp07_run1')  # 95.53 -
learn_rate.append(0.07)
state_size.append(8)
fc_size.append(128)

#
model_no.append('cnn_rnn_isoGrouping_attention_temp_lrp07_run2')  #95.4164 -
learn_rate.append(0.08)
state_size.append(10)
fc_size.append(128)

#
model_no.append('cnn_rnn_isoGrouping_attention_temp3_lrp08_run1')  #95.36
learn_rate.append(0.08)
state_size.append(10)
fc_size.append(80)

#
model_no.append('cnn_rnn_isoGrouping_attention_v8_lrp08_run1') #95.47  -
learn_rate.append(0.08)
state_size.append(6)
fc_size.append(128)


#model_no.append('cnn_rnn_isoGrouping_attention_v2_lrp08_run1')  #95.36 -
#learn_rate.append(0.08)
#state_size.append(8)
#fc_size.append(128)

#model_no.append('cnn_rnn_isoGrouping_attention_v2_lrp08_run2') #95.42 -
#learn_rate.append(0.08)
#state_size.append(8)
#fc_size.append(128)

#model_no.append('cnn_rnn_isoGrouping_attention_v2_lrp08_run3') #  95.42
#learn_rate.append(0.08)
#state_size.append(8)
#fc_size.append(128)

#model_no.append('cnn_rnn_isoGrouping_attention_temp_lrp08_run1')  #95.30 -
#learn_rate.append(0.08)
#state_size.append(10)
#fc_size.append(128)


#model_no.append('cnn_rnn_isoGrouping_attention_temp2_lrp08_run1')  #95.14
#learn_rate.append(0.08)
#state_size.append(10)
#fc_size.append(100)

#model_no.append('cnn_rnn_isoGrouping_attention_v2_lrp09_run1') #95.19 -
#learn_rate.append(0.09)
#state_size.append(8)
#fc_size.append(128)

#model_no.append('cnn_rnn_isoGrouping_attention_v2_lrp09_run2') #95.14
#learn_rate.append(0.09)
#state_size.append(8)
#fc_size.append(128)

total_models=len(model_no)


my_models=[]
for model_idx in range (0, total_models):
    my_models.append(model_generation. isoGrouping_model(state_size[model_idx], fc_size[model_idx], learn_rate[model_idx], model_no[model_idx]))

for model_idx in range(0, total_models):
    print(model_no[model_idx])



#print('model restore done')

####################################################################

print('scanning test ms: '+file_name)
print('reading dictionary record from disk, you will get a message after its done')
f=open(datapath+file_name+'_ms1_record', 'rb')
RT_mz_I_dict, maxI=pickle.load(f)
f.close()   
print('disk read done')
print('reading ms1 record from disk, you will get a message after its done')
f=open(datapath+file_name+'_consecutive_scan_MS1_1', 'rb') 
ms1=pickle.load(f)
f.close()
f=open(datapath+file_name+'_consecutive_scan_MS1_2', 'rb') 
ms1_next=pickle.load(f)
f.close()    
print('disk read done')

ms1=np.concatenate((ms1, np.copy(ms1_next)), axis=1)
temp_ms1=np.zeros((ms1.shape[0]+RT_window, ms1.shape[1]+211))
temp_ms1[0:ms1.shape[0], 0:ms1.shape[1]]=np.copy(ms1[:, :])
ms1=np.copy(temp_ms1)
temp_ms1=0

mz_resolution=2
RT_list = np.sort(list(RT_mz_I_dict.keys()))
max_RT=RT_list[len(RT_list)-1]
min_RT=10    

sorted_mz_list=[]
RT_index=dict()
for i in range(0, len(RT_list)):
    RT_index[round(RT_list[i], 2)]=i
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
while(RT_list[rt_search_index]<min_RT):
    rt_search_index=rt_search_index+1
#############################
f=open(datapath+file_name+'_clusters', 'rb')  
isotope_cluster, max_num_iso,  total_clusters=pickle.load(f)
f.close()


cluster_length=np.zeros((total_clusters))
count=0
mz_list=sorted(isotope_cluster.keys())
cluster_list=[]
for i in range (0, len(mz_list)): #len(mz_list)
    ftr_list=isotope_cluster[mz_list[i]]
    for j in range (0, len(ftr_list)):
        ftr=ftr_list[j]        
        cluster_list.append(ftr)
        cluster_length[count]=len(ftr)-1                   
        count=count+1

isotope_cluster=0
start_iso=np.zeros((total_clusters))
current_iso=np.zeros((total_clusters))
feature_table=defaultdict(list)

total_batch=math.ceil(total_clusters/batch_size)
print('total batch %d. starting processing.'%total_batch)
DEBUG=0
total_feature=0
cluster_count=0
case_count=0
start_time=time()
for batch_idx in range (0, total_batch):
#        print(batch_idx)
    start_cluster=batch_idx*batch_size
    end_cluster=min(start_cluster+batch_size, total_clusters)
    cluster_count=cluster_count+end_cluster-start_cluster
    cluster_left=1

    while(cluster_left):
        # for each cluster, assign frames from start_iso to total_frames_hor, to the cut_block
        # make the batch
        count=0
        cut_block=np.zeros((batch_size, RT_window,frame_width*total_frames_hor))   
        ftr_z=np.zeros((batch_size, 1))
        for c in range (start_cluster, end_cluster):
            if cluster_length[c]<=0:
                continue
            ftr=cluster_list[c]    
            cluster_z=int(ftr[len(ftr)-1][0])
            space=int(round((isotope_gap[int(cluster_z)]/mz_unit), mz_resolution))    
            RT_peak=round(ftr[int(current_iso[c])][1][0], 2)        
            # 7 step before, peak, 7 step after
            RT_s=max(RT_index[RT_peak]-7-rt_search_index, 0)
            RT_e=min(RT_s+RT_window, len(RT_list)) #ex
        
            fr_mz=round(ftr[int(current_iso[c])][0], mz_resolution)
            mz_poz=max(int(round((fr_mz-min_mz)/mz_unit, mz_resolution))-5, 0)
            
            for fr in range (0, total_frames_hor): # all isotopes
                temp=np.copy(ms1[RT_s:RT_e, mz_poz:mz_poz+frame_width])                   
                cut_block[count, 0:temp.shape[0], fr*frame_width:fr*frame_width+temp.shape[1]]=np.copy(temp)
                mz_poz=mz_poz+space                

            ftr_z[count, 0]=cluster_z
            count=count+1
        # one batch made
        
        if count==0:
            break
    
        # now run the model
        current_batch_size=count
        _single_prediction=np.zeros((total_models, current_batch_size))
        _decision_array=np.zeros((total_models, current_batch_size,  num_class))

        for model_idx in range (0, total_models):
            _current_state = np.zeros((current_batch_size, state_size[model_idx]))
            _single_prediction[model_idx, :], _decision_array[model_idx, :, :]= my_models[model_idx].sess.run(
                [my_models[model_idx].prediction, my_models[model_idx].decision_array],
                feed_dict={
                    my_models[model_idx].batchX_placeholder:cut_block[0:current_batch_size],
                    my_models[model_idx].batchZ_placeholder:ftr_z[0:current_batch_size] ,
                    my_models[model_idx].init_state:_current_state[0:current_batch_size],
                    my_models[model_idx].keep_prob:1.0
                })          

        
        # now ck the prediction for each cluster and set the start iso for next run accordingly
        count=-1
        for c in range (start_cluster, end_cluster):
            if cluster_length[c]<=0:
                continue
            count=count+1
            decision_sum=_decision_array[0, count, :] # 1st model
            hard_voting=np.zeros((num_class)) 
            for model_idx in range (1, total_models):
                decision_sum=decision_sum+_decision_array[model_idx, count, :]
                hard_voting[int(_single_prediction[model_idx, count])]=hard_voting[int(_single_prediction[model_idx, count])]+1

            _prediction= np.argmax(decision_sum/total_models) #int(np.argmax(hard_voting)) #np.argmax(decision_sum/total_models) #_single_prediction[0, count] #
            
            if _prediction>0:
                if _prediction==total_frames_hor-1 and cluster_length[c]>total_frames_hor:
                    current_iso[c]=current_iso[c]+_prediction
                    cluster_length[c]=cluster_length[c]-_prediction
#                    _current_state = np.zeros((batch_size, state_size))
                else:
                    if _prediction<cluster_length[c]:
                        end_iso=int(current_iso[c]+_prediction+1) #(ex)
                    else:
                        end_iso=int(current_iso[c]+cluster_length[c]) #(ex)
                    
                    new_ftr=[]
                    ftr=cluster_list[c]
                    for isotope in range (int(start_iso[c]) ,end_iso):
                        new_ftr.append(ftr[isotope])
                        
                    new_ftr.append([ftr[len(ftr)-1][0]]) # charge
                    feature_table[round(new_ftr[0][0], mz_resolution)].append(new_ftr)
                    total_feature=total_feature+1                    

                    start_iso[c]=end_iso                        
                    current_iso[c]=start_iso[c]
                    cluster_length[c]=cluster_length[c]-_prediction-1
#                        _current_state = np.zeros((batch_size, state_size))
            else:
                end_iso=int(current_iso[c]+_prediction+1)
                if current_iso[c]!=start_iso[c]: # it was continuing 
                    new_ftr=[]
                    ftr=cluster_list[c]
                    for isotope in range (int(start_iso[c]),end_iso):
                        new_ftr.append(ftr[isotope])
                        
                    new_ftr.append([ftr[len(ftr)-1][0]]) # charge
                    feature_table[round(new_ftr[0][0], mz_resolution)].append(new_ftr)
                    total_feature=total_feature+1                    

                start_iso[c]=end_iso                        
                current_iso[c]=start_iso[c]
                cluster_length[c]=cluster_length[c]-_prediction-1
#                    _current_state = np.zeros((batch_size, state_size))
    
        
        
    #print('%d, cluster count %d, feature_count %d'%(i, cluster_count, total_feature))
print('elapsed time: %g seconds'%(time()-start_time))


#######################################

print('writing feature table')     
f=open(datapath+file_name+'_featureTable', 'wb')
pickle.dump(feature_table, f, protocol=2)
f.close()   
print('data write done')


################ Iteration over features ##########
'''
count=0
for key in feature_table:
    feature_list=feature_table[key]
    print("list of features starting at m/z: %g"%key)
    for i in range (0, len(feature_list)): # iterate over the features
        ftr=feature_list[i]
        print("Feature %d: "%count)
        for j in range (0, len(ftr)-1): # iterate over the isotopes of that feature
            print("isotope %d: m/z:%g, RT start:%g, RT end:%g, RT peak:%g "%(j, ftr[j][0], ftr[j][1][1], ftr[j][1][2], ftr[j][1][0]))
        print("charge z: %d"%ftr[len(ftr)-1][0]) # charge of that feature
        
        count=count+1
        
print('total feature %d'%count)
'''







