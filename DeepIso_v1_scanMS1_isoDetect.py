# Sample Running Command: nohup python -u DeepIso_v1_scanMS1_isoDetect.py <rawDataFileName without extension> <batch_size> <parallel_section> <segment> <gpu_index> > <logfile> &
'''Example:
You have a file ABC.ms1 at /DeepIsoV1/rawdata/ABC.ms1
You want to run IsoDetecting on this file. 
You allow batch size 5000 to be processed at a time by this module.
You allow 3 parallel processing of the MS1 file. That means, you slice the MS1 file into 3 separate slices and process them on parallel.
You want to run slice 0 of those three slices.
You are using GPU 3.
You want to see the log in output_isoDetecting_0.log file. 
Then you should put the command as below:
nohup python -u DeepIso_v1_scanMS1_isoDetect.py ABC 5000 3 0 3 > output_isoGrouping_0.log &
If your GPU allows processing of other two slices as well then you should also provide following two lines right away:
nohup python -u DeepIso_v1_scanMS1_isoDetect.py ABC 5000 3 1 3 > output_isoGrouping_1.log &
nohup python -u DeepIso_v1_scanMS1_isoDetect.py ABC 5000 3 2 3 > output_isoGrouping_2.log &
'''
from __future__ import division
from __future__ import print_function
import tensorflow as tf
#import math
from time import time
import pickle
import numpy as np
from collections import deque
import gc
#from collections import defaultdict
#import copy
#import scipy.misc
import sys
import os
########## file run parameters #################################
current_path=os.system("pwd")
datapath=current_path+'/DeepIsoV1/data/'  
modelpath=current_path+'/DeepIsoV1/model/'
file_name=sys.arg[1]
batch_size=int(sys.arg[2]) #5000
parallel_section=int(sys.arg[3]) # total number of slices to process on parallel
segment= int(sys.argv[4]) # if the previous parameter is set to 3, then this parameter should be something 0/1/2 
gpu_index=sys.argv[5]

############# scanning parameters #########################################################
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
mz_window=211
total_class=10
RT_unit=0.01
mz_unit=0.01


###########deep learning parameters###########################################################
fc_size=4
num_class=10
state_size = fc_size
num_neurons= num_class #mz_window*RT_window
print("model building start")
def weight_variable(shape, variable_name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=variable_name)

def bias_variable(shape, variable_name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=variable_name)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#15 x 211
with tf.device('/gpu:'+gpu_index): 
    batchX_placeholder = tf.placeholder(tf.float32, [None, RT_window, mz_window]) #image block to consider for one run of training by back propagat$
    sample_weight = tf.placeholder(tf.float32, [None]) #image block to consider for one run of training by back propagation
    keep_prob = tf.placeholder(tf.float32)

    W_conv0 = weight_variable([8, 10 , 1, 8], 'W_conv0')#v10: 23x 202
    b_conv0 = bias_variable([8], 'b_conv0') #for each of feature maps

    W_conv1 = weight_variable([4, 10 , 8, 16], 'W_conv1')# #20x193
    b_conv1 = bias_variable([16], 'b_conv1') #for each of feature maps

    W_conv2 = weight_variable([4, 8, 16, 32],  'W_conv2')  #18x186
    b_conv2 = bias_variable([32], 'b_conv2') 

    W_fc1 = weight_variable([2 * 186 * 32, 264], 'W_fc1')
    b_fc1 = bias_variable([264], 'b_fc1')

    W_out = weight_variable([264, fc_size], 'W_out')
    b_out = bias_variable([fc_size], 'b_out')


    batchY_placeholder = tf.placeholder(tf.int32, [None])
    init_state = tf.placeholder(tf.float32, [None, state_size])

    W = tf.Variable(np.random.rand(state_size, state_size), dtype=tf.float32) 

    W2 = tf.Variable(np.random.rand(state_size, num_class),dtype=tf.float32) #final output
    b2 = tf.Variable(np.zeros((1,num_class)), dtype=tf.float32) #final output


    # Forward pass
    current_state = init_state
    x_image = tf.reshape(batchX_placeholder[:, :, :], [-1, RT_window, mz_window, 1]) #

    h_conv0 = tf.tanh(conv2d(x_image, W_conv0) + b_conv0) # now the input is: (15-8+1) x (211-22+1) x 16 = 8 x 190 x 16
    h_conv1 = tf.tanh(conv2d(h_conv0, W_conv1) + b_conv1) # now the input is: (8-4+1) x (190-6+1) x 16 = 5 x 185 x 16
    h_conv2 = tf.tanh(conv2d(h_conv1, W_conv2) + b_conv2) # now the input is: (5-3+1) x (185-4+1) x 8 = 3  x 182  x 8
    #    h_conv3 = tf.tanh(conv2d(h_conv2, W_conv3) + b_conv3) #3-3+1 x 182-3+1 x 8 = 1 x 180 x 8
    h_conv2_flat = tf.reshape(h_conv2, [-1, 2 * 186  * 32])
    h_conv2_flat_drop = tf.nn.dropout(h_conv2_flat, keep_prob)
    h_fc1 = tf.tanh(tf.matmul(h_conv2_flat_drop, W_fc1) + b_fc1) # finally giving the output
    h_fc1_dropout=tf.nn.dropout(h_fc1, keep_prob)
    #        h_fc2 = tf.tanh(tf.matmul(h_fc1, W_fc2) + b_fc2) # finally giving the output
    #        h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)
    y_conv = tf.tanh(tf.matmul(h_fc1_dropout, W_out) + b_out) # finally giving the output
    ##############################
    current_FC  = y_conv # tf.nn.dropout(y_conv, keep_prob) # [batch_size, fc_size])
    weighted_state = tf.matmul(current_state, W) # Broadcasted addition #shape?? # EDIT
    next_state = tf.tanh(weighted_state + current_FC)  # Broadcasted addition #shape?? # EDIT
        

    logit = tf.matmul(next_state, W2) + b2  #Broadcasted addition
    #predictions_series = [tf.nn.softmax(logits) for logits in logits_series]
    predictions_series = tf.argmax(tf.nn.softmax(logit), 1) 

    loss=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=batchY_placeholder[:]) # [batch_size,loss]  
        
    considered_loss=tf.multiply(sample_weight, loss)    
    total_loss=tf.reduce_sum(considered_loss) / tf.to_float(tf.reduce_sum(sample_weight))
    train_step = tf.train.AdagradOptimizer(.01).minimize(total_loss)

config=tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
saver = tf.train.Saver()
print("model building done")
print("state restore start")
saver.restore(sess, modelpath+'fcrnn_isoDetect_v2_lrp05_fold2_run6_best_sen_model.ckpt') # 
print('state restore done')
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
temp_ms1=np.zeros((ms1.shape[0]+RT_window, ms1.shape[1]+mz_window))
temp_ms1[0:ms1.shape[0], 0:ms1.shape[1]]=np.copy(ms1[:, :])
ms1=np.copy(temp_ms1)
temp_ms1=0
ms1_next=0
gc.collect()


###########################
#scan ms1_block and record the cnn outputs in list_dict[z]: hash table based on m/z
#for each m/z
mz_resolution=2
RT_list = np.sort(list(RT_mz_I_dict.keys()))
max_RT=RT_list[len(RT_list)-1]
min_RT=RT_list[0] #10    

sorted_mz_list=[]
RT_index=dict()
for i in range(0, len(RT_list)):
    RT_index[round(RT_list[i], 2)]=i
    sorted_mz_list.append(sorted(RT_mz_I_dict[RT_list[i]]))   
    
mz_exist=[]
for i in range (0, len(sorted_mz_list)):
    temp_mz_list=[]
    temp_mz_list.append(sorted_mz_list[i][0][0])
    count=0
    for j in range (1, len(sorted_mz_list[i])):
        if temp_mz_list[count]==sorted_mz_list[i][j][0]:
            continue
        else:
            temp_mz_list.append(sorted_mz_list[i][j][0])
            count=count+1

    sorted_mz_list[i]=temp_mz_list
    mz_exist.append(dict())
    for j in range (0, len(temp_mz_list)):
        mz_exist[i][temp_mz_list[j]]=1
    
max_mz=0
min_mz=1000
for i in range (0, len(sorted_mz_list)):
    mz_I_list=sorted_mz_list[i]
    mz=mz_I_list[len(mz_I_list)-1]
    if mz>max_mz:
        max_mz=mz
    mz=mz_I_list[0]
    if mz<min_mz:
        min_mz=mz
        

rt_search_index=0
while(RT_list[rt_search_index]<=min_RT):
    if RT_list[rt_search_index]==min_RT:
        break
    rt_search_index=rt_search_index+1 



total_mz=int(round((max_mz-min_mz+mz_unit)/mz_unit, mz_resolution)) 
total_RT=len(RT_list)-rt_search_index

#############################

mz_used_before=np.zeros((total_class))    
pred_RT=np.zeros((total_class))
pred_start=np.zeros((total_class))
list_dict=[]
for i in range (0, total_class):
    list_dict.append(dict())



total_stripe=int(total_mz/batch_size)
print(total_stripe)


stripe_per_section=total_stripe//parallel_section

start_stripe=segment*stripe_per_section
if segment==parallel_section-1:
    end_stripe=total_stripe
else:
    end_stripe=start_stripe+stripe_per_section
    
start_mz=round(min_mz+(start_stripe*batch_size)*mz_unit, mz_resolution)


start_time=time()
max_len=0
for stripe_index in range (start_stripe, end_stripe): # total_stripe):
    print(stripe_index)
    #start_time=time()
    # do one stripe
    start_col=stripe_index*batch_size
    current_mz=round(min_mz+start_col*mz_unit, mz_resolution)
    
    output_list=np.zeros((total_RT, batch_size))
    holder_current_state= np.zeros((batch_size, state_size)) # for one stripe

    for y in range (0, total_RT):
#        print(y)
        # creating one batch        
        batch_input=np.zeros((batch_size, RT_window,  mz_window))
        _current_state= np.zeros((batch_size, state_size))
        count=0
        batch_index=0
        kept_batch_index=[]
        for col in range (start_col, start_col+batch_size):
            mz_col=round(col*mz_unit + min_mz, mz_resolution)
            after_10ppm=round(mz_col+(mz_col*10.0)/10**6, mz_resolution)   
            flag=0
            col_difference=int(round((after_10ppm-mz_col)/mz_unit, mz_resolution))
            if np.sum(ms1[y:y+4, col:col+col_difference+1]!=0)>=1:
                flag=1
            
            if flag==1 or (y>0 and flag==0 and output_list[y-1, batch_index]!=0):
                batch_input[count, :, :]=np.copy(ms1[y:y+RT_window, col:col+mz_window])
                _current_state[count, :]=holder_current_state[batch_index, :]
                kept_batch_index.append(batch_index)
                count=count+1
            
            batch_index=batch_index+1
        
        if count==0:
            continue
            
        batch_input=np.copy(batch_input[0:count][:, :])
        _current_state=np.copy(_current_state[0:count][:, :])
        if max_len<count:
            max_len=count

        #one batch is made
        _current_state, _predictions_series = sess.run(
            [next_state, predictions_series],
            feed_dict={
                batchX_placeholder:batch_input,
                init_state:_current_state, 
                keep_prob:1.0
            })    
        #one batch is done
        count=0
        for batch_index in kept_batch_index:
            output_list[y, batch_index]=_predictions_series[count]
            holder_current_state[batch_index, :]=_current_state[count, :]
            count=count+1 

    #one stripe having dimension [total_RT, batch_size] is searched
    #fill out the list_dict for each col in that stripe
#       print('fcrnn processing done')
    # how many non zero cells in output_list?        
    for batch_index in range (0, batch_size):
        # current_mz will be found from mz_list
        after_10ppm=(current_mz*10.0)/10**6 
        mz_poz=round(current_mz+after_10ppm, mz_resolution)
        mz_used_before[:]=0
        pred_RT[:]=0
        pred_start[:]=0
        not_exist=1
        for i in range (0, total_RT): #0.02
            RT_poz=round(RT_list[rt_search_index+i], 2) # i1=int((RT_poz1-min_RT)/RT_unit)     i2=int((RT_poz2-min_RT)/RT_unit)  # step = int((mz_poz-min_mz)/mz_unit-5)
            z=int(output_list[i, batch_index]) #int(output_list[i, batch_index])
            if z!=0:
                # add (m/z,RT) to the dict
                if mz_used_before[z]==1:  #list_dict[p_ion[i]].has_key(mz_poz):
                    # append the new number to the existing array at this slot
        #                if RT_poz not in list_dict[p_ion[i]][mz_poz]:
                    if RT_index[RT_poz]-RT_index[pred_RT[z]]==1: #continuation of same isotope
                        pred_RT[z]=RT_poz
                    elif pred_start[z]==pred_RT[z]:  
                        list_dict[z][mz_poz].append(-1) # insert a separating marker
                        list_dict[z][mz_poz].append(RT_poz) # insert the starting RT of new isotope
                        pred_start[z]=RT_poz #keep track of starting RT
                        pred_RT[z]=RT_poz
                        
                    else: 
                        list_dict[z][mz_poz].append(round(pred_RT[z], 2))
                        list_dict[z][mz_poz].append(-1) # insert a separating marker --> CHECK
                        list_dict[z][mz_poz].append(RT_poz) # insert the starting RT of new isotope
                        pred_start[z]=RT_poz #keep track of starting RT
                        pred_RT[z]=RT_poz
                else:
                    # create a new array in this slot
                    list_dict[z][mz_poz] = deque()#[RT_poz]
                    list_dict[z][mz_poz].append(RT_poz)
                    mz_used_before[z]=1
                    pred_start[z]=RT_poz
                    pred_RT[z]=RT_poz
                    
            
        for i in range (1, total_class):
            if mz_used_before[i]==1: 
                if pred_start[i]==pred_RT[i]:  
                    list_dict[i][mz_poz].pop()
                else: 
                    list_dict[i][mz_poz].append(round(pred_RT[i], 2))
                    list_dict[i][mz_poz].append(-1)
                    
        # all rt done for one mz
        current_mz=round(current_mz+mz_unit, mz_resolution)
    #one stripe done
# followings are tabbed back
time_elapsed=time()-start_time 
print(time_elapsed) 
print('writing the scanning results')
f=open(datapath+file_name+'_scanning_result'+'_seg_'+str(segment), 'wb') 
pickle.dump([list_dict,stripe_index], f, protocol=2) #all mz_done
f.close()
print('done')
list_dict=0





