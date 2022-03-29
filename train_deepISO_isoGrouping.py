# consecutive scan along RT axis
# The original script was developed for Tensorflow Version 1
# So if you need to retrain this model, then you may need to change the script according to the newer versions
from __future__ import print_function, division
import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt
import pickle
import math
import copy
from time import time
#import sys
#import scipy.misc
import csv
from collections import defaultdict
isotope_gap=np.zeros((10))
isotope_gap[0]=0.01
isotope_gap[1]=100
isotope_gap[2]=50
isotope_gap[3]=33
isotope_gap[4]=25
isotope_gap[5]=20
isotope_gap[6]=17
isotope_gap[7]=14
isotope_gap[8]=13
isotope_gap[9]=11
#
truncated_backprop_length = 5
state_size = 8 #
fc_size = 128
num_epochs= 200
learn_rate= 0.07 # 0.08 -- gave best so far
log_no='cnn_rnn_isoGrouping_attention_v2_lrp07_run2'
batch_size=128
print('%s, learn rate %g'%(log_no,learn_rate))
#better_start='fcrnn_consecutiveScan_combISO_REtrain_scratch_pool'
take_zero=1
#log_no_old='fcrnn_consecutiveScan_combISO_z_run2'
activation_func=2
#val_start=100
total_frames_hor=truncated_backprop_length
total_hops_horizontal= total_frames_hor//truncated_backprop_length 
drop_out_k=0.5


RT_window=15
mz_window=11

validation_index=9
############## load data #################
modelpath='ENTER the path to save the model'
mappath='ENTER the path to load the hash tables holding ms1 data'
datapath='ENTER the path to load the training data from'     
dataname=['130124_dilA_8_01','130124_dilA_8_02', '130124_dilA_8_03', '130124_dilA_8_04', '130124_dilA_9_01','130124_dilA_9_02','130124_dilA_9_03','130124_dilA_9_04','130124_dilA_10_01','130124_dilA_10_02', '130124_dilA_10_03', '130124_dilA_10_04', '130124_dilA_11_01', '130124_dilA_11_02', '130124_dilA_11_03', '130124_dilA_11_04', '130124_dilA_12_01', '130124_dilA_12_02', '130124_dilA_12_03', '130124_dilA_12_04'] 

#######################################################################
num_class=total_frames_hor

num_neurons= num_class #mz_window*RT_window


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

batchX_placeholder = tf.placeholder(tf.float32, [None, RT_window, mz_window*truncated_backprop_length]) #image block to consider for one run of training by back propagation
keep_prob = tf.placeholder(tf.float32)

# each image is 15 x 11
W_conv0 = weight_variable([2, 2 , 1, 8], 'W_conv0')#v10: 
b_conv0 = bias_variable([8], 'b_conv0') #15-2+1=14,11-2+1=10
# pool - 7, 5

W_conv1 = weight_variable([2, 2 , 8, 16], 'W_conv1')#v10: 12-4+1=9, 7-4+1=4 # 6,4
b_conv1 = bias_variable([16], 'b_conv1') #for each of feature maps
# pool - 3, 2

W_conv2 = weight_variable([2, 2 , 16, 32], 'W_conv2')#v10: 12-4+1=9, 7-4+1=4 # 2, 1
b_conv2 = bias_variable([32], 'b_conv2') #for each of feature maps


W_conv3 = weight_variable([2, 1, 32, 64], 'W_conv3')  # 1, 1
b_conv3 = bias_variable([64], 'b_conv3') 


#2 x 1
W_fc1 = weight_variable([1 * 1 * 64, 64], 'W_fc1') #
b_fc1 = bias_variable([64], 'b_fc1')

#
#W_fc2 = weight_variable([128, 256], 'W_fc2') #
#b_fc2 = bias_variable([256], 'b_fc2')

W_out = weight_variable([64+1, fc_size], 'W_out')
b_out = bias_variable([fc_size], 'b_out')

#param_loader = tf.train.Saver({'W_conv0': W_conv0, 'W_conv1': W_conv1, 'W_conv2': W_conv2, 'W_conv3': W_conv3, 'W_fc1':W_fc1, 'W_out':W_out, 'b_conv0':b_conv0, 'b_conv1':b_conv1, 'b_conv2':b_conv2, 'b_conv3':b_conv3, 'b_fc1':b_fc1, 'b_out':b_out})

batchY_placeholder = tf.placeholder(tf.float32, [None, num_class])
batchZ_placeholder = tf.placeholder(tf.float32, [None, 1])
batchAUC_placeholder = tf.placeholder(tf.float32, [None, num_class, 1])

init_state = tf.placeholder(tf.float32, [None, state_size])

W = tf.Variable(np.random.rand(fc_size+state_size, state_size), dtype=tf.float32) 
b = tf.Variable(np.zeros((1,state_size)), dtype=tf.float32) # 2D RNN

W_attention = tf.Variable(np.random.rand(state_size, state_size), dtype=tf.float32) 
b_attention = tf.Variable(np.zeros((1,state_size)), dtype=tf.float32) #final output


W2 = tf.Variable(np.random.rand(state_size, num_class),dtype=tf.float32) #final output
b2 = tf.Variable(np.zeros((1,num_class)), dtype=tf.float32) #final output

# Unpack columns
#labels_series = tf.unpack(batchY_placeholder, axis=1)

# Forward pass
current_state = init_state
states_series = []
for j in range (0, truncated_backprop_length):
    ##############################
    x_image = tf.reshape(batchX_placeholder[:, : , mz_window*j : mz_window* (j+1)], [-1, RT_window, mz_window, 1]) #flatten to 2d: row: RT, column: mz

    h_conv0 = tf.tanh(conv2d(x_image, W_conv0) + b_conv0) # now the input is: (15-8+1) x (211-22+1) x 16 = 8 x 190 x 16           
    h_pool0 = max_pool_2x2(h_conv0)    

    h_conv1 = tf.tanh(conv2d(h_pool0, W_conv1) + b_conv1) # now the input is: (8-4+1) x (190-6+1) x 16 = 5 x 185 x 16
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.tanh(conv2d(h_pool1, W_conv2) + b_conv2) # now the input is: (8-4+1) x (190-6+1) x 16 = 5 x 185 x 16
#    h_pool2 = max_pool_2x2(h_conv2)


    h_conv3 = tf.tanh(conv2d(h_conv2, W_conv3) + b_conv3) # now the input is: (5-3+1) x (185-4+1) x 8 = 3  x 182  x 8
    h_conv3_flat = tf.reshape(h_conv3, [-1, 1 * 1  * 64])

#    h_conv3_flat_drop = tf.nn.dropout(h_conv3_flat, keep_prob)

    h_fc1 = tf.tanh(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)
    h_fc1_dropout=tf.nn.dropout(h_fc1, keep_prob)
#
#    h_fc2 = tf.tanh(tf.matmul(h_fc1_dropout, W_fc2) + b_fc2)
#    h_fc2_dropout=tf.nn.dropout(h_fc2, keep_prob)
# 
    h_fc1_dropout_z = tf.concat([h_fc1_dropout, batchZ_placeholder], 1)
#        h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)
    h_fc2= tf.tanh(tf.matmul(h_fc1_dropout_z, W_out) + b_out) # finally this will connect with RNN
    ##############################
    current_FC  = tf.nn.dropout(h_fc2, keep_prob) # [batch_size, fc_size])
    
    FC_and_state_concatenated = tf.concat([current_FC, current_state], 1) # row --> batch
    weighted_FC_state = tf.matmul(FC_and_state_concatenated , W) + b # 
    
    cand_next_state = tf.tanh(weighted_FC_state)  # ht
    at= tf.sigmoid(tf.matmul(cand_next_state, W_attention) + b_attention) # attention of ht 
    next_state=  tf.multiply(current_state, (1-at))+ tf.multiply(cand_next_state, at) 
        
    states_series.append(next_state)
    current_state = next_state

logit = tf.matmul(current_state, W2) + b2 
#predictions_series = [tf.nn.softmax(logits) for logits in logits_series]
prediction = tf.argmax(tf.nn.softmax(logit), 1)

loss=tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=batchY_placeholder)
#    loss_series.append(tf.nn.softmax_cross_entropy_with_logits(logits=tf.multiply(logits_series[col], class_weight), labels=batchY_placeholder[:, col, :]))
    
total_loss = tf.reduce_mean(loss)

train_step = tf.train.AdagradOptimizer(learn_rate).minimize(total_loss)

#sess = tf.Session()
#sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
config=tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
#saver.restore(sess, modelpath+'init-model_'+log_no+'.ckpt')
saver.save(sess, modelpath+'init-model_'+log_no+'.ckpt')
#saver.restore(sess, modelpath+'trained-model_'+better_start+'_best.ckpt')
#saver.restore(sess, modelpath+'trained-model_'+log_no+'_best.ckpt')
#######################################

data_index=4
f=open(datapath+'/feature_list/consecutiveScan_'+dataname[data_index]+'_combineIsotope_dataset', 'rb')
cut_ms1, auc_info, feature_info = pickle.load(f)
f.close() 
feature_info=np.copy(feature_info[0:len(cut_ms1), :])

for data_index in range (5,  8):
    f=open(datapath+'/feature_list/consecutiveScan_'+dataname[data_index]+'_combineIsotope_dataset', 'rb')
    cut_ms1_next,auc_info,  feature_info_next = pickle.load(f)
    f.close()   
    for i in range (0, len(cut_ms1_next)):
#	if feature_info_next[i,1]==2 or feature_info_next[i,1]==3:
#		continue
        cut_ms1.append(cut_ms1_next[i])    
    
    feature_info=np.concatenate([feature_info, np.copy(feature_info_next[0:len(cut_ms1_next), :])], axis=0)    

# 8, 9, 10, 11 --> for val
for data_index in range (12, len(dataname)):
    f=open(datapath+'/feature_list/consecutiveScan_'+dataname[data_index]+'_combineIsotope_dataset', 'rb')
    cut_ms1_next,auc_info,  feature_info_next = pickle.load(f)
    f.close()   
    for i in range (0, len(cut_ms1_next)):
#	if feature_info_next[i,1]==2 or feature_info_next[i,1]==3:
#		continue
        cut_ms1.append(cut_ms1_next[i])    
    
    feature_info=np.concatenate([feature_info, np.copy(feature_info_next[0:len(cut_ms1_next), :])], axis=0)    


#
#cut_ms1_val= copy.deepcopy(cut_ms1[int(len(cut_ms1)*(4/5)):])
#feature_info_val= np.copy(feature_info[int(len(cut_ms1)*(4/5)):, :])        
#
#cut_ms1=copy.deepcopy(cut_ms1[0:int(len(cut_ms1)*(4/5))])
#feature_info= np.copy(feature_info[0:int(feature_info.shape[0]*(4/5)), :])        


f=open(datapath+'/feature_list/consecutiveScan_'+dataname[10]+'_combineIsotope_dataset', 'rb')
cut_ms1_val, auc_info, feature_info_val = pickle.load(f)
f.close()   
feature_info_val=np.copy(feature_info_val[0:len(cut_ms1_val), :])


#f=open(datapath+'/feature_list/consecutiveScan_'+dataname[9]+'_combineIsotope_dataset', 'rb')
#cut_ms1_next_val, auc_info, feature_info_next_val = pickle.load(f)
#f.close()   
#for i in range (0, len(cut_ms1_next_val)):
#    cut_ms1_val.append(cut_ms1_next_val[i])    
#
#feature_info_val=np.concatenate([feature_info_val, np.copy(feature_info_next_val[0:len(cut_ms1_next_val), :])], axis=0)        
    
#for data_index in range (10,  len(dataname)):
#    f=open(datapath+'/feature_list/consecutiveScan_'+dataname[data_index]+'_combineIsotope_dataset', 'rb')
#    cut_ms1_next, auc_info, feature_info_next = pickle.load(f)
#    f.close()   
#    for i in range (0, len(cut_ms1_next)):
#        cut_ms1.append(cut_ms1_next[i])    
#    
#    feature_info=np.concatenate([feature_info, np.copy(feature_info_next[0:len(cut_ms1_next), :])], axis=0)    
#    

################    
#feature_info=np.copy(feature_info[:, 0:3])
#feature_info_val=np.copy(feature_info_val[:, 0:3])
amount_nonzero_data=len(cut_ms1)
print('****** non zero data for training %d **********'%amount_nonzero_data)

if take_zero==1:
    data_index=12 
    f=open(datapath+'cut_features/'+dataname[data_index]+'_isoCombine_zerodata_poz', 'rb')
    cut_ms1_zero, auc_info,  feature_info_zero = pickle.load(f)
    f.close()       
    feature_info_zero=np.copy(feature_info_zero[0:len(cut_ms1_zero), :])
    logfile=open(datapath+'feature_list/'+dataname[data_index]+'_combineIsotopes_featureList.csv', 'rb')
    peptide_feature=np.loadtxt(logfile, delimiter=',')
    logfile.close()    
    
    for i in range (0, len(cut_ms1_zero)):
        feature_info_zero[i, 1]=peptide_feature[int(feature_info_zero[i, 0]), 3] #charge    
        feature_info_zero[i, 2]=0 # feature_width    
    
    
    for data_index in range (13,16): #len(dataname)): #9
        f=open(datapath+'cut_features/'+dataname[data_index]+'_isoCombine_zerodata_poz', 'rb')
        cut_ms1_zero_next, auc_info, feature_info_zero_next = pickle.load(f)
        f.close()
        
        logfile=open(datapath+'feature_list/'+dataname[data_index]+'_combineIsotopes_featureList.csv', 'rb')
        peptide_feature=np.loadtxt(logfile, delimiter=',')
        logfile.close()
     
        for i in range (0, len(cut_ms1_zero_next)):
            cut_ms1_zero.append(cut_ms1_zero_next[i])
            feature_info_zero_next[i, 1]=peptide_feature[int(feature_info_zero_next[i, 0]), 3] #charge    
            feature_info_zero_next[i, 2]=0 # feature_width
            
        feature_info_zero=np.concatenate([feature_info_zero, np.copy(feature_info_zero_next[0:len(cut_ms1_zero_next), :])], axis=0)
        

        
        
    cut_ms1.extend(copy.deepcopy(cut_ms1_zero[0:int(len(cut_ms1_zero)*(4/5))]))
    feature_info=np.concatenate([feature_info, np.copy(feature_info_zero[0:int(len(cut_ms1_zero)*(4/5)), :])], axis=0)        
        
    cut_ms1_val.extend(copy.deepcopy(cut_ms1_zero[int(len(cut_ms1_zero)*(4/5)):]))
    feature_info_val=np.concatenate([feature_info_val, np.copy(feature_info_zero[int(len(cut_ms1_zero)*(4/5)):, :])], axis=0)        
###############blank frames before############################################################
    data_index=12
    f=open(datapath+'cut_features/'+dataname[data_index]+'_isoCombine_zerodata_bz', 'rb')
    cut_ms1_zero, feature_info_zero = pickle.load(f)
    f.close()       
    feature_info_zero=np.copy(feature_info_zero[0:len(cut_ms1_zero), :])
#    temp_block=255-np.copy(cut_ms1_zero[1])
#    scipy.misc.imsave(datapath+'blank_frame.jpg', temp_block) 
 
    for data_index in range (13, 16): #len(dataname)): #9
        f=open(datapath+'cut_features/'+dataname[data_index]+'_isoCombine_zerodata_bz', 'rb')
        cut_ms1_zero_next, feature_info_zero_next = pickle.load(f)
        f.close()
     
        for i in range (0, len(cut_ms1_zero_next)):
            cut_ms1_zero.append(cut_ms1_zero_next[i])
            
        feature_info_zero=np.concatenate([feature_info_zero, np.copy(feature_info_zero_next[0:len(cut_ms1_zero_next), :])], axis=0)
        
    feature_info=np.copy(feature_info[:, 0:3])
    feature_info_val=np.copy(feature_info_val[:, 0:3])
    cut_ms1.extend(copy.deepcopy(cut_ms1_zero[0:int(len(cut_ms1_zero)*(4/5))]))
    feature_info=np.concatenate([feature_info, np.copy(feature_info_zero[0:int(len(cut_ms1_zero)*(4/5)), :])], axis=0)        
        
    cut_ms1_val.extend(copy.deepcopy(cut_ms1_zero[int(len(cut_ms1_zero)*(4/5)):]))
    feature_info_val=np.concatenate([feature_info_val, np.copy(feature_info_zero[int(len(cut_ms1_zero)*(4/5)):, :])], axis=0)        


#####################Noisy#############################################################

    f=open(datapath+'cut_features/'+'isoCombine_zeroNnoisy_data', 'rb')
    cut_ms1_retrain, feature_info_retrain = pickle.load(f)
    f.close() 
    feature_info_retrain=np.copy(feature_info_retrain[0:len(cut_ms1_retrain), 0:3])

    cut_ms1.extend(copy.deepcopy(cut_ms1_retrain[0:int(len(cut_ms1_retrain)*(4/5))]))
    feature_info=np.concatenate([feature_info, np.copy(feature_info_retrain[0:int(len(cut_ms1_retrain)*(4/5)), :])], axis=0)   

    cut_ms1_val.extend(copy.deepcopy(cut_ms1_retrain[int(len(cut_ms1_retrain)*(4/5)):]))
    feature_info_val=np.concatenate([feature_info_val, np.copy(feature_info_retrain[int(len(cut_ms1_retrain)*(4/5)):, :])], axis=0)   
    
 ##################################################################################   
    # adjacent feature problem: based on peptide feature match of lots of lc-ms map
    f=open(datapath+'cut_features/'+'_combineIso_AUC_zerodata_1', 'rb')
    cut_ms1_retrain, auc_info, feature_info_retrain = pickle.load(f)
    f.close() 
    feature_info_retrain=np.copy(feature_info_retrain[0:len(cut_ms1_retrain), 0:3])
#    feature_info_retrain=np.copy(feature_info_retrain[:, 0:3])
    for i in range (0, 1):    
        cut_ms1.extend(copy.deepcopy(cut_ms1_retrain[0:int(len(cut_ms1_retrain)*(4/5))]))
        feature_info=np.concatenate([feature_info, np.copy(feature_info_retrain[0:int(len(cut_ms1_retrain)*(4/5)), :])], axis=0)   

    cut_ms1_val.extend(copy.deepcopy(cut_ms1_retrain[int(len(cut_ms1_retrain)*(4/5)):]))
    feature_info_val=np.concatenate([feature_info_val, np.copy(feature_info_retrain[int(len(cut_ms1_retrain)*(4/5)):, :])], axis=0) 



#    # adjacent feature problem:  based on augmented case of above
#    f=open(datapath+'cut_features/'+'_combineIso_AUC_zerodata_2', 'rb')
#    cut_ms1_retrain, auc_info, feature_info_retrain = pickle.load(f)
#    f.close() 
#    feature_info_retrain=np.copy(feature_info_retrain[0:len(cut_ms1_retrain), 0:3])
##    feature_info_retrain=np.copy(feature_info_retrain[:, 0:3])
#    for i in range (0, 1):    
#        cut_ms1.extend(copy.deepcopy(cut_ms1_retrain[0:int(len(cut_ms1_retrain)*(4/5))]))
#        feature_info=np.concatenate([feature_info, np.copy(feature_info_retrain[0:int(len(cut_ms1_retrain)*(4/5)), :])], axis=0)   
#
#    cut_ms1_val.extend(copy.deepcopy(cut_ms1_retrain[int(len(cut_ms1_retrain)*(4/5)):]))
#    feature_info_val=np.concatenate([feature_info_val, np.copy(feature_info_retrain[int(len(cut_ms1_retrain)*(4/5)):, :])], axis=0) 
#
#

    # based on peptife feature match
    f=open(datapath+'cut_features/'+'isoCombine_retrainData_2', 'rb')
    cut_ms1_retrain, feature_info_retrain = pickle.load(f)
    f.close() 
    feature_info_retrain=np.copy(feature_info_retrain[0:len(cut_ms1_retrain), :])
    feature_info_retrain=np.copy(feature_info_retrain[:, 0:3])
    for i in range (0, 5):    
        cut_ms1.extend(copy.deepcopy(cut_ms1_retrain[  0:int(len(cut_ms1_retrain)*(4/5))  ]))
        feature_info=np.concatenate([feature_info, np.copy(feature_info_retrain[   0:int(len(cut_ms1_retrain)*(4/5)), :   ])], axis=0)   

    cut_ms1_val.extend(copy.deepcopy(cut_ms1_retrain[   int(len(cut_ms1_retrain)*(4/5)):   ]))
    feature_info_val=np.concatenate([feature_info_val, np.copy(feature_info_retrain[   int(len(cut_ms1_retrain)*(4/5)):, :   ])], axis=0)   

    # adjacent feature problem:  based on ms2 id match
    f=open(datapath+'cut_features/'+'isoCombine_retrainData', 'rb')
    cut_ms1_retrain, feature_info_retrain = pickle.load(f)
    f.close() 
    feature_info_retrain=np.copy(feature_info_retrain[0:len(cut_ms1_retrain), 0:3])
#    feature_info_roetrain=np.copy(feature_info_retrain[:, 0:3])
    for i in range (0, 10):    
        cut_ms1.extend(copy.deepcopy(cut_ms1_retrain[0:int(len(cut_ms1_retrain)*(4/5))]))
        feature_info=np.concatenate([feature_info, np.copy(feature_info_retrain[0:int(len(cut_ms1_retrain)*(4/5)), :])], axis=0)   

    cut_ms1_val.extend(copy.deepcopy(cut_ms1_retrain[int(len(cut_ms1_retrain)*(4/5)):]))
    feature_info_val=np.concatenate([feature_info_val, np.copy(feature_info_retrain[int(len(cut_ms1_retrain)*(4/5)):, :])], axis=0)   

total_feature=len(cut_ms1)
print('****** zero data for training %d **********'%(total_feature-amount_nonzero_data))

print('total feature %d'%total_feature)
#batch_size=total_feature
number_of_batch=total_feature//batch_size



#########Create Log##############################################################
logfile=open(modelpath+'deepISO_performance_'+log_no+'.csv', 'wb')
logfile.close()

#logfile=open(modelpath+'deepISO_class_loss_'+log_no+'.csv', 'wb')
#logfile.close()
#################### report feature module#####################################

f=open(datapath+'feature_list/'+dataname[validation_index]+'_ms1_record', 'rb')
RT_mz_I_dict, maxI=pickle.load(f)
f.close()   

f=open(mappath+dataname[validation_index]+'_consecutive_scan_MS1_1', 'rb') 
ms1=pickle.load(f)
f.close()
f=open(mappath+dataname[validation_index]+'_consecutive_scan_MS1_2', 'rb') 
ms1_next=pickle.load(f)
f.close()    
ms1=np.concatenate((ms1, np.copy(ms1_next)), axis=1)
temp_ms1=np.zeros((ms1.shape[0]+RT_window, ms1.shape[1]+mz_window))
temp_ms1[0:ms1.shape[0], 0:ms1.shape[1]]=np.copy(ms1[:, :])
ms1=np.copy(temp_ms1)
temp_ms1=0
print('data restore done')
#scan ms1_block and record the cnn outputs in list_dict[z]: hash table based on m/z
#for each m/z
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

print('preprocess done')
#############################
f=open(mappath+dataname[validation_index]+'_db_matched_cluster_v2_sum', 'rb') # 94.41 
isotope_cluster, max_num_iso, total_clusters=pickle.load(f)
f.close()
print('making cluster list')
mz_list=sorted(isotope_cluster.keys())
cluster_list=[]
for i in range (0, len(mz_list)): #len(mz_list)
    ftr_list=isotope_cluster[mz_list[i]]
    for j in range (0, len(ftr_list)):
        ftr=ftr_list[j]        
        cluster_list.append(ftr)
print('making done')

total_clusters=len(cluster_list)

filename = datapath+'feature_list/'+dataname[validation_index]+'_peptide.csv' 
rows = []
csvfile=open(filename, 'r')
csvreader = csv.reader(csvfile)     
for row in csvreader:
    rows.append(row)
csvfile.close() 

RT_tolerance=0.2 # dino is same

mz_unit=0.01
frame_width=11

####################################################################################

prob=0
with sess.as_default():    
    print('starting validation')
    count_batch_val=0
    avg_loss_val=0
    accuracy_measure=np.zeros((1, num_class+1))
    confusion_matrix=np.zeros((num_class, num_class))
    real_class=np.zeros((num_class))                                        
        
        
    total_feature_val=len(cut_ms1_val)
    batch_size_val=1000
    number_of_batch_val=total_feature_val//batch_size_val
    count_batch_val=count_batch_val+number_of_batch_val
    ftr_val=-1
    for batch_idx_val in range (0,  number_of_batch_val):
        batch_ms1_val=np.zeros((batch_size_val, RT_window,mz_window*total_frames_hor))
        batch_label_val=np.zeros((batch_size_val, 1, num_class))    
        batch_z_val=np.zeros((batch_size_val, 1))    
        batch_predictions_val=np.zeros((batch_size_val, 1)) 
        count_val=0
        while count_val!=batch_size_val:
            ftr_val=ftr_val+1
            charge=int(feature_info_val[ftr_val, 1])
            feature_width=int(feature_info_val[ftr_val, 2]) # number of isotopes
#                                feature_width=int(MQ_feature_val[int(peptide_feature_val[int(feature_info_val[ftr_val, 0]), 15]), 2])
            if feature_width==0:
                batch_label_val[count_val, 0, 0]=1
            elif feature_width<total_frames_hor:
                batch_label_val[count_val, 0, feature_width-1]=1
            else:
                batch_label_val[count_val, 0, total_frames_hor-1]=1
                
            batch_z_val[count_val, 0]=charge                                                        
            frame_count=total_frames_hor #min(feature_width, total_frames_hor)
            mz_start=0
            for i in range (0, frame_count):
                batch_ms1_val[count_val,:, (i)*mz_window:(i+1)*mz_window]=np.copy(cut_ms1_val[ftr_val][:,mz_start:mz_start+mz_window])                                                
                mz_start=int(mz_start+isotope_gap[charge])

            count_val=count_val+1

        # one batch_val is formed
#                print('batch %d is formed'%batch_idx)
#                class_loss=np.zeros((1, num_class))
        _current_state = np.zeros((batch_size_val, state_size))               
        for col_idx in range(0,total_hops_horizontal): # total_hops_horizontal=87 in each hop, 6 windows are considered as truncated backprop length is 6
            start_col=col_idx * truncated_backprop_length * mz_window # 
            end_col= start_col + truncated_backprop_length * mz_window # 
            
            batchX = batch_ms1_val[:,:, start_col:end_col]
            
            label_start_column = col_idx * truncated_backprop_length
            label_end_column = label_start_column + truncated_backprop_length
            batchY = batch_label_val[:,0, label_start_column:label_end_column]
            batchZ = batch_z_val[:]
            _total_loss, _current_state, _predictions_series = sess.run(
                [total_loss, current_state, prediction],
                feed_dict={
                    batchX_placeholder:batchX,
                    batchY_placeholder:batchY ,
                    batchZ_placeholder:batchZ ,
                    init_state:_current_state,
                    keep_prob:1.0
                })                                        
#                print("hello")
#                print(sess.run(W))
            avg_loss_val=avg_loss_val+_total_loss
            for b in range (0, batch_size_val):
                batch_predictions_val[b,0]=int(_predictions_series[b])                    
    
        
        for b in range (0, batch_size_val):
            real_charge=int(np.argmax(batch_label_val[b, 0, :]))
            pred_charge=int(batch_predictions_val[b, 0])
#                                print(pred_charge)
            real_class[real_charge]=real_class[real_charge]+1
            confusion_matrix[real_charge, pred_charge]=confusion_matrix[real_charge, pred_charge]+1   
                                            
    
    avg_loss_val=avg_loss_val/(count_batch_val)    
    print('avg loss %g'%(avg_loss_val) )    
    for i in range (0, num_class):
        print("avg accuracy for z=%d is %g, amount %d"%(i, confusion_matrix[i, i]/real_class[i], real_class[i]))

min_loss=avg_loss_val
##################                
print('starting validation by report feature module')
cluster_length=np.zeros((total_clusters))
count=0
for i in range (0, len(cluster_list)): #len(mz_list)
        ftr=copy.deepcopy(cluster_list[i])        
        cluster_length[count]=len(ftr)-1                   
        count=count+1

start_iso=np.zeros((total_clusters))
current_iso=np.zeros((total_clusters))
feature_table=defaultdict(list)
batch_size_val=total_clusters
total_batch_val=math.ceil(total_clusters/batch_size_val)
DEBUG=0
total_feature=0
cluster_count=0
case_count=0
for batch_idx in range (0, total_batch_val):
    print(batch_idx)
    start_cluster=batch_idx*batch_size_val
    end_cluster=min(start_cluster+batch_size_val, total_clusters)
    cluster_count=cluster_count+end_cluster-start_cluster
    cluster_left=1
    _current_state = np.zeros((batch_size_val, state_size))    
    while(cluster_left):
        # for each cluster, assign frames from start_iso to total_frames_hor, to the cut_block
        # make the batch
        count=0
        cut_block=np.zeros((batch_size_val, RT_window,frame_width*total_frames_hor))   
        ftr_z=np.zeros((batch_size_val, 1))
        for c in range (start_cluster, end_cluster):
            if cluster_length[c]<=0:
                continue
            ftr=cluster_list[c]    
            cluster_z=int(ftr[len(ftr)-1][0])
            space=int(isotope_gap[cluster_z])    
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
        print('current_batch_size %d'%current_batch_size)
        _prediction_batch=np.zeros((current_batch_size))
        _current_state = np.zeros((batch_size_val, state_size))

        _prediction_batch= sess.run(
            [prediction],
            feed_dict={
                batchX_placeholder:cut_block[0:current_batch_size],
                batchZ_placeholder:ftr_z[0:current_batch_size] ,
                init_state:_current_state[0:current_batch_size],
                keep_prob:1.0
            })          
        
        # now ck the prediction for each cluster and set the start iso for next run accordingly
        count=-1
        for c in range (start_cluster, end_cluster):
            if cluster_length[c]<=0:
                continue
            count=count+1            
            _prediction= _prediction_batch[0][count] 
            
            if _prediction>0:
                if _prediction==total_frames_hor-1 and cluster_length[c]>total_frames_hor:
                    current_iso[c]=current_iso[c]+_prediction
                    cluster_length[c]=cluster_length[c]-_prediction
                    _current_state = np.zeros((batch_size_val, state_size))
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
                    _current_state = np.zeros((batch_size_val, state_size))
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
                _current_state = np.zeros((batch_size_val, state_size))

############## match feature module ###########################################
found_ftr=0
total_feature=0
for i in range (1, len(rows)):
    found=0
    mz_exact=round(float(rows[i][5]), mz_resolution)        
    total_feature=total_feature+1           

    mz_range=[]
    mz_range.append(mz_exact)    
    tolerance_mz=0.01# dinosaur = 0.005 
    mz_range.append(round(mz_exact-tolerance_mz, mz_resolution))
    mz_range.append(round(mz_exact+tolerance_mz, mz_resolution))
    for j in range (0, len(mz_range)):
        mz=mz_range[j]
        if mz in feature_table:
            ftr_list=feature_table[mz]
            for k in range (0, len(ftr_list)):
                ftr=ftr_list[k]
                peak_RT=ftr[0][1][0] 
                if (round(float(rows[i][6])-RT_tolerance, 2) <= peak_RT) and (peak_RT<=round(float(rows[i][6])+RT_tolerance, 2)):
                    found=1
                    found_ftr=found_ftr+1
                    break

            if found==1:
                break

val_acc=found_ftr/total_feature
print(val_acc)
max_acc=val_acc
################ training ############################################################
#saver.restore(sess, modelpath+'trained-model_'+better_start+'_epoch.ckpt')
#saver.restore(sess, modelpath+'trained-model_'+log_no+'_epoch.ckpt')

val_start=94
with sess.as_default():    
    for epoch_idx in range(0, 200):
        # go to each feature
        start_time=time()
        print("epoch", epoch_idx)
        count_batch=0
        avg_loss=0
        count_batch=count_batch+number_of_batch
        random_pick=np.random.permutation(len(cut_ms1))
        confusion_matrix_train=np.zeros((num_class, num_class))
        real_class_train=np.zeros((num_class))   
        r=-1
        for batch_idx in range (0,  number_of_batch):
            batch_ms1=np.zeros((batch_size, RT_window,mz_window*total_frames_hor))
            batch_label=np.zeros((batch_size, 1, num_class))
            batch_z=np.zeros((batch_size, 1))
            batch_predictions=np.zeros((batch_size, 1))
            count=0
            while count!=batch_size :
                r=r+1
                ftr=random_pick[r]

                charge=int(feature_info[ftr, 1])
                feature_width=int(feature_info[ftr, 2]) # number of isotopes
                if feature_width==0:
                    batch_label[count, 0, 0]=1 # zero data
                elif feature_width<total_frames_hor:
                    batch_label[count, 0, feature_width-1]=1
                else:
                    batch_label[count, 0, total_frames_hor-1]=1
                    
                batch_z[count, 0]=charge
                frame_count=total_frames_hor #min(feature_width, total_frames_hor)
                mz_start=0
                for i in range (0, frame_count):
                    batch_ms1[count,:, (i)*mz_window:(i+1)*mz_window]=np.copy(cut_ms1[ftr][:,mz_start:mz_start+mz_window])                                                
                    mz_start=int(mz_start+isotope_gap[charge])

                count=count+1

            # one batch is formed
            _current_state = np.zeros((batch_size, state_size))               
            for col_idx in range(0,total_hops_horizontal): # total_hops_horizontal=87 in each hop, 6 windows are considered as truncated backprop length is 6
                start_col=col_idx * truncated_backprop_length * mz_window # 
                end_col= start_col + truncated_backprop_length * mz_window # 
                
                batchX = batch_ms1[:,:, start_col:end_col]
                
                label_start_column = col_idx * truncated_backprop_length
                label_end_column = label_start_column + truncated_backprop_length
                batchY = batch_label[:,0, :]
                batchZ = batch_z[:]
                _total_loss, _train_step, _current_state, _predictions_series = sess.run(
                    [total_loss, train_step, current_state, prediction],
                    feed_dict={
                        batchX_placeholder:batchX,
                        batchY_placeholder:batchY ,
                        batchZ_placeholder:batchZ ,
                        init_state:_current_state, 
                        keep_prob:drop_out_k
                    })                                                        
                avg_loss=avg_loss+_total_loss


################################################################################                
            if (epoch_idx>=val_start and batch_idx%5==0) or (batch_idx==number_of_batch-1 and epoch_idx%10==0):
#                print('starting validation')
                count_batch_val=0
                avg_loss_val=0
                accuracy_measure=np.zeros((1, num_class+1))
                confusion_matrix=np.zeros((num_class, num_class))
                real_class=np.zeros((num_class))                                        
                    
                total_feature_val=len(cut_ms1_val)
                batch_size_val=1000
                number_of_batch_val=total_feature_val//batch_size_val
                count_batch_val=count_batch_val+number_of_batch_val
                ftr_val=-1
                for batch_idx_val in range (0,  number_of_batch_val):
                    batch_ms1_val=np.zeros((batch_size_val, RT_window,mz_window*total_frames_hor))
                    batch_label_val=np.zeros((batch_size_val, 1, num_class))    
                    batch_z_val=np.zeros((batch_size_val, 1))    
                    batch_predictions_val=np.zeros((batch_size_val, 1)) 
                    count_val=0
                    while count_val!=batch_size_val:
                        ftr_val=ftr_val+1
                        charge=int(feature_info_val[ftr_val, 1])
                        feature_width=int(feature_info_val[ftr_val, 2]) # number of isotopes
#                                feature_width=int(MQ_feature_val[int(peptide_feature_val[int(feature_info_val[ftr_val, 0]), 15]), 2])
                        if feature_width==0:
                            batch_label_val[count_val, 0, 0]=1
                        elif feature_width<total_frames_hor:
                            batch_label_val[count_val, 0, feature_width-1]=1
                        else:
                            batch_label_val[count_val, 0, total_frames_hor-1]=1
                            
                        batch_z_val[count_val, 0]=charge                                                        
                        frame_count=total_frames_hor #min(feature_width, total_frames_hor)
                        mz_start=0
                        for i in range (0, frame_count):
                            batch_ms1_val[count_val,:, (i)*mz_window:(i+1)*mz_window]=np.copy(cut_ms1_val[ftr_val][:,mz_start:mz_start+mz_window])                                                
                            mz_start=int(mz_start+isotope_gap[charge])

                        count_val=count_val+1

                    # one batch_val is formed
    #                print('batch %d is formed'%batch_idx)
    #                class_loss=np.zeros((1, num_class))
                    _current_state = np.zeros((batch_size_val, state_size))               
                    for col_idx in range(0,total_hops_horizontal): # total_hops_horizontal=87 in each hop, 6 windows are considered as truncated backprop length is 6
                        start_col=col_idx * truncated_backprop_length * mz_window # 
                        end_col= start_col + truncated_backprop_length * mz_window # 
                        
                        batchX = batch_ms1_val[:,:, start_col:end_col]
                        
                        label_start_column = col_idx * truncated_backprop_length
                        label_end_column = label_start_column + truncated_backprop_length
                        batchY = batch_label_val[:,0, label_start_column:label_end_column]
                        batchZ = batch_z_val[:]
                        _total_loss, _current_state, _predictions_series = sess.run(
                            [total_loss, current_state, prediction],
                            feed_dict={
                                batchX_placeholder:batchX,
                                batchY_placeholder:batchY ,
                                batchZ_placeholder:batchZ ,
                                init_state:_current_state,
                                keep_prob:1.0
                            })                                        
        #                print("hello")
        #                print(sess.run(W))
                        avg_loss_val=avg_loss_val+_total_loss
                        for b in range (0, batch_size_val):
                            batch_predictions_val[b,0]=int(_predictions_series[b])                    
                
                    
                    for b in range (0, batch_size_val):
                        real_charge=int(np.argmax(batch_label_val[b, 0, :]))
                        pred_charge=int(batch_predictions_val[b, 0])
#                                print(pred_charge)
                        real_class[real_charge]=real_class[real_charge]+1
                        confusion_matrix[real_charge, pred_charge]=confusion_matrix[real_charge, pred_charge]+1   
                                                        
                
                avg_loss_val=avg_loss_val/(count_batch_val)    
                for i in range (0, num_class):
#                    print("avg accuracy for z=%d is %g, amount %d"%(i, confusion_matrix[i, i]/real_class[i], real_class[i]))
                    accuracy_measure[0, i]=round(confusion_matrix[i, i]/real_class[i], 2)
                    
                accuracy_measure[0, num_class]=avg_loss_val
                print('for epoch %d, batch %d, avg loss %g'%(epoch_idx,batch_idx, avg_loss_val) )    
                logfile=open(modelpath+'deepISO_performance_'+log_no+'.csv', 'ab')
                np.savetxt(logfile,accuracy_measure, delimiter=',')
                logfile.close() 
                
##################                
                print('starting validation by report feature module')
                cluster_length=np.zeros((total_clusters))
                count=0
                for i in range (0, len(cluster_list)): #len(mz_list)
                        ftr=copy.deepcopy(cluster_list[i])        
                        cluster_length[count]=len(ftr)-1                   
                        count=count+1

                start_iso=np.zeros((total_clusters))
                current_iso=np.zeros((total_clusters))
                feature_table=defaultdict(list)
                batch_size_val=total_clusters
                total_batch_val=math.ceil(total_clusters/batch_size_val)
#                print('total batch %d'%total_batch)
                DEBUG=0
                total_feature=0
                cluster_count=0
                case_count=0
                for batch_idx_val in range (0, total_batch_val):
#                    print(batch_idx)
                    start_cluster=batch_idx_val*batch_size_val
                    end_cluster=min(start_cluster+batch_size_val, total_clusters)
                    cluster_count=cluster_count+end_cluster-start_cluster
                    cluster_left=1
                    _current_state = np.zeros((batch_size_val, state_size))    
                    while(cluster_left):
                        # for each cluster, assign frames from start_iso to total_frames_hor, to the cut_block
                        # make the batch
                        count=0
                        cut_block=np.zeros((batch_size_val, RT_window,frame_width*total_frames_hor))   
                        ftr_z=np.zeros((batch_size_val, 1))
                        for c in range (start_cluster, end_cluster):
                            if cluster_length[c]<=0:
                                continue
                            ftr=cluster_list[c]    
                            cluster_z=int(ftr[len(ftr)-1][0])
                            space=int(isotope_gap[cluster_z])    
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
#                        print('current_batch_size %d'%current_batch_size)
                        _prediction_batch=np.zeros((current_batch_size))
                        _current_state = np.zeros((batch_size_val, state_size))

                        _prediction_batch= sess.run(
                            [prediction],
                            feed_dict={
                                batchX_placeholder:cut_block[0:current_batch_size],
                                batchZ_placeholder:ftr_z[0:current_batch_size] ,
                                init_state:_current_state[0:current_batch_size],
                                keep_prob:1.0
                            })          
                        
                        # now ck the prediction for each cluster and set the start iso for next run accordingly
                        count=-1
                        for c in range (start_cluster, end_cluster):
                            if cluster_length[c]<=0:
                                continue
                            count=count+1            
                            _prediction= _prediction_batch[0][count] 
                            
                            if _prediction>0:
                                if _prediction==total_frames_hor-1 and cluster_length[c]>total_frames_hor:
                                    current_iso[c]=current_iso[c]+_prediction
                                    cluster_length[c]=cluster_length[c]-_prediction
                                    _current_state = np.zeros((batch_size_val, state_size))
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
                                    _current_state = np.zeros((batch_size_val, state_size))
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
                                _current_state = np.zeros((batch_size_val, state_size))
 
############## match feature module ###########################################
                found_ftr=0
                total_feature=0
                for i in range (1, len(rows)):
                    found=0
                    mz_exact=round(float(rows[i][5]), mz_resolution)        
                    total_feature=total_feature+1           

                    mz_range=[]
                    mz_range.append(mz_exact)    
                    tolerance_mz=0.01# dinosaur = 0.005 
                    mz_range.append(round(mz_exact-tolerance_mz, mz_resolution))
                    mz_range.append(round(mz_exact+tolerance_mz, mz_resolution))
                    for j in range (0, len(mz_range)):
                        mz=mz_range[j]
                        if mz in feature_table:
                            ftr_list=feature_table[mz]
                            for k in range (0, len(ftr_list)):
                                ftr=ftr_list[k]
                                peak_RT=ftr[0][1][0] 
                                if (round(float(rows[i][6])-RT_tolerance, 2) <= peak_RT) and (peak_RT<=round(float(rows[i][6])+RT_tolerance, 2)):
                                    found=1
                                    found_ftr=found_ftr+1
                                    break

                            if found==1:
                                break

                val_acc=found_ftr/total_feature

##################
                print('val_acc:%g'%val_acc)
                for i in range (0, num_class):
                    print("avg accuracy for z=%d is %g, fn: %g, amount %d"%(i, confusion_matrix[i, i]/real_class[i], confusion_matrix[i, 0]/real_class[i], real_class[i]))

                if  val_acc>=max_acc: #avg_loss_val<=min_loss:
                    max_acc= val_acc #min_loss=avg_loss_val
                    #save the model
                    saver.save(sess, modelpath+'trained-model_'+log_no+'_best.ckpt')
                    print('BEST found with val acc %g'% val_acc)

                    
        elapsed_time=time()-start_time
        print('elapsed time:%g, total_batch:%d, avg_loss %g'%(elapsed_time,  count_batch, avg_loss/count_batch))
        saver.save(sess, modelpath+'trained-model_'+log_no+'_epoch.ckpt')
        


#

############

#epoch 51
#avg loss 0.522766
#avg accuracy for z=0 is 0.981728, amount 19155
#avg accuracy for z=1 is 0.571073, amount 3616
#avg accuracy for z=2 is 0.644633, amount 6475
#avg accuracy for z=3 is 0.669578, amount 5953
#avg accuracy for z=4 is 0.536499, amount 2548
#avg accuracy for z=5 is 0, amount 253

#avg accuracy for z=0 is 0.987097, amount 155
#f=open(datapath+'cut_features/'+'isoCombine_retrainData', 'rb')
#cut_ms1_retrain, feature_info_retrain = pickle.load(f)
#f.close() 










#v4_run2_90 was achived first at epoch 30.
#for epoch 90, batch 3180, avg loss 0.405239
#starting validation by report feature module
#val_acc:0.904972
#avg accuracy for z=0 is 0.967636, fn: 0.967636, amount 41281
#avg accuracy for z=1 is 0.479714, fn: 0.110743, amount 3919
#avg accuracy for z=2 is 0.659351, fn: 0.0262556, amount 7427
#avg accuracy for z=3 is 0.647643, fn: 0.0147424, amount 6851
#avg accuracy for z=4 is 0.717617, fn: 0.0163883, amount 3173
#avg accuracy for z=5 is 0.00286533, fn: 0.00859599, amount 349
#BEST found with val acc 0.904972












