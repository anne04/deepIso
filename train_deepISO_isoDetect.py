# consecutive scan along RT axis
# The original script was developed for Tensorflow Version 1
# So if you need to retrain this model, then you may need to change the script according to the newer versions

from __future__ import print_function, division
import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt
import pickle
#import math
from time import time
#import sys
import copy

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


num_epochs= 55
learn_rate= 0.05 #run2 had learn_rate=0.02
batch_size=128
log_no='fcrnn_isoDetect_v2_lrp05_run4' #'fcrnn_isoDetect_v2_lrp05_run2'
  #run2 had learn_rate=0.02
activation_func=2
val_start=0
#
#truncated_backprop_length = int(sys.argv[1]) #3
#fc_size= int(sys.argv[2]) #3 
#num_epochs= int(sys.argv[3])
#learn_rate= float(sys.argv[4])
#batch_size=int(sys.argv[5]) #128
#log_no=sys.argv[6] #128
#activation_func=int(sys.argv[7])
#val_start=int(sys.argv[8])

total_frames_var=20
RT_window=15
mz_window=211
num_class=10
############## load data #################
modelpath='ENTER the path to save the model'
datapath='ENTER the path to load the training data from'    

dataname=['130124_dilA_10_01', '130124_dilA_10_02', '130124_dilA_10_03', '130124_dilA_10_04', '130124_dilA_9_01', '130124_dilA_9_02', '130124_dilA_9_03', '130124_dilA_9_04', '130124_dilA_11_01', '130124_dilA_11_02', '130124_dilA_11_03', '130124_dilA_11_04', '130124_dilA_12_01', '130124_dilA_12_02', '130124_dilA_12_03', '130124_dilA_12_04'] 


f=open(datapath+'/cut_features/'+dataname[4]+'_scanMS1_stripe_dataset', 'rb')
feature_set, label_set, sequence_length, map_to_peaks, real_class_train=pickle.load(f)
f.close()   

for data_index in range (5, len(dataname)):
    f=open(datapath+'/cut_features/'+dataname[data_index]+'_scanMS1_stripe_dataset', 'rb')
    feature_set_next, label_set_next, sequence_length_next, map_to_peaks_next, real_class_next=pickle.load(f)
    f.close()   

    feature_set.extend(copy.deepcopy(feature_set_next))
    label_set.extend(copy.deepcopy(label_set_next))
    sequence_length.extend(copy.deepcopy(sequence_length_next))
    real_class_train=real_class_train+real_class_next

#for data_index in range (8, 10):
#    f=open(datapath+'/cut_features/'+dataname[data_index]+'_scanMS1_stripe_dataset', 'rb')
#    feature_set_next, label_set_next, sequence_length_next, map_to_peaks_next, real_class_next=pickle.load(f)
#    f.close()   
#    for i in range (0, len(feature_set_next)):
#        charge=np.max(label_set_next[i])
#        if charge==2 or charge==3:
#            continue
#        else:            
#            feature_set.append(np.copy(feature_set_next[i]))
#            label_set.append(np.copy(label_set_next[i]))
#            sequence_length.append(np.copy(sequence_length_next[i]))
#            for j in range (0, label_set_next[i].shape[0]):
#                real_class_train[int(label_set_next[i][j])]=real_class_train[int(label_set_next[i][j])]+1

print('+ve data:%d'%len(feature_set))

poz_data=len(feature_set)

f=open(datapath+'/cut_features/'+dataname[2]+'_scanMS1_stripe_dataset', 'rb')
feature_set_val, label_set_val, sequence_length_val, map_to_peaks_val, real_class_val=pickle.load(f)
f.close()   
f=open(datapath+'/cut_features/'+dataname[3]+'_scanMS1_stripe_dataset', 'rb')
feature_set_val_next, label_set_val_next,  sequence_length_val_next, map_to_peaks_val_next, real_class_val_next=pickle.load(f)
f.close()   

feature_set_val.extend(copy.deepcopy(feature_set_val_next))
label_set_val.extend(copy.deepcopy(label_set_val_next))
sequence_length_val.extend(copy.deepcopy(sequence_length_val_next))
real_class_train_val=real_class_train+real_class_val_next


################################################################

f=open(datapath+'cut_features/'+dataname[8]+'_zerodata_stripe', 'rb')
feature_set_next, label_set_next, sequence_length_next, map_to_peaks_next, real_class_next = pickle.load(f)
f.close()
total_zero=len(feature_set_next) #100000 #10000
num_zero_stripe_val=int((total_zero*20)/100)
num_zero_stripe=int((total_zero*80)/100)
feature_set.extend(copy.deepcopy(feature_set_next[0:num_zero_stripe]))
label_set.extend(copy.deepcopy(label_set_next[0:num_zero_stripe]))
sequence_length.extend(copy.deepcopy(sequence_length_next[0:num_zero_stripe]))
for i in range (0, num_zero_stripe):
    for j in range (0, label_set_next[i].shape[0]):
        real_class_train[int(label_set_next[i][j])]=real_class_train[int(label_set_next[i][j])]+1


feature_set_val.extend(copy.deepcopy(feature_set_next[num_zero_stripe:num_zero_stripe+num_zero_stripe_val]))
label_set_val.extend(copy.deepcopy(label_set_next[num_zero_stripe:num_zero_stripe+num_zero_stripe_val]))
sequence_length_val.extend(copy.deepcopy(sequence_length_next[num_zero_stripe:num_zero_stripe+num_zero_stripe_val]))
######### Negative data: add noisy data############################################################
print('Noisy data')
f=open(datapath+'cut_features/'+dataname[8]+'_zerodata_TN_stripe', 'rb')
feature_set_next, label_set_next, sequence_length_next, map_to_peaks_next, real_class_next=pickle.load(f)
f.close()
# repeat 5 times to give 5x weight
for i in range (0, 15):    
    feature_set.extend(copy.deepcopy(feature_set_next[0:int(len(feature_set_next)*0.8)]))
    label_set.extend(copy.deepcopy(label_set_next[0:int(len(feature_set_next)*0.8)]))
    sequence_length.extend(copy.deepcopy(sequence_length_next[0:int(len(feature_set_next)*0.8)]))
    for i in range (0, int(len(feature_set_next)*0.8)):
        for j in range (0, label_set_next[i].shape[0]):
            real_class_train[int(label_set_next[i][j])]=real_class_train[int(label_set_next[i][j])]+1


feature_set_val.extend(copy.deepcopy(feature_set_next[int(len(feature_set_next)*0.8): ]))
label_set_val.extend(copy.deepcopy(label_set_next[int(len(feature_set_next)*0.8): ]))
sequence_length_val.extend(copy.deepcopy(sequence_length_next[int(len(feature_set_next)*0.8): ]))
######### Negative data: add data with long gap before ############################################################
f=open(datapath+'cut_features/'+dataname[8]+'_zerodata_SB_stripe', 'rb')
feature_set_next, label_set_next, sequence_length_next, map_to_peaks_next, real_class_next=pickle.load(f)
f.close()
# repeat 5 times to give 5x weight
for i in range (0, 15):    
    feature_set.extend(copy.deepcopy(feature_set_next[0:int(len(feature_set_next)*0.8)]))
    label_set.extend(copy.deepcopy(label_set_next[0:int(len(feature_set_next)*0.8)]))
    sequence_length.extend(copy.deepcopy(sequence_length_next[0:int(len(feature_set_next)*0.8)]))
    for i in range (0, int(len(feature_set_next)*0.8)):
        for j in range (0, label_set_next[i].shape[0]):
            real_class_train[int(label_set_next[i][j])]=real_class_train[int(label_set_next[i][j])]+1

feature_set_val.extend(copy.deepcopy(feature_set_next[int(len(feature_set_next)*0.8): ]))
label_set_val.extend(copy.deepcopy(label_set_next[int(len(feature_set_next)*0.8): ]))
sequence_length_val.extend(copy.deepcopy(sequence_length_next[int(len(feature_set_next)*0.8): ]))
########Negative data: add data with long gap after#####################################
f=open(datapath+'cut_features/'+dataname[8]+'_zerodata_EB_stripe', 'rb')
feature_set_next, label_set_next, sequence_length_next, map_to_peaks_next, real_class_next=pickle.load(f)
f.close()
# repeat 5 times to give 5x weight
for i in range (0, 15):    
    feature_set.extend(copy.deepcopy(feature_set_next[0:int(len(feature_set_next)*0.8)]))
    label_set.extend(copy.deepcopy(label_set_next[0:int(len(feature_set_next)*0.8)]))
    sequence_length.extend(copy.deepcopy(sequence_length_next[0:int(len(feature_set_next)*0.8)]))
    for i in range (0, int(len(feature_set_next)*0.8)):
        for j in range (0, label_set_next[i].shape[0]):
            real_class_train[int(label_set_next[i][j])]=real_class_train[int(label_set_next[i][j])]+1

feature_set_val.extend(copy.deepcopy(feature_set_next[int(len(feature_set_next)*0.8): ]))
label_set_val.extend(copy.deepcopy(label_set_next[int(len(feature_set_next)*0.8): ]))
sequence_length_val.extend(copy.deepcopy(sequence_length_next[int(len(feature_set_next)*0.8): ]))
#####################Total Noise###########################################
f=open(datapath+'cut_features/'+dataname[8]+'_zerodata_TB_stripe', 'rb')
feature_set_next, label_set_next, sequence_length_next, map_to_peaks_next, real_class_next=pickle.load(f)
f.close()
# repeat 5 times to give 5x weight
for i in range (0, 15):    
    feature_set.extend(copy.deepcopy(feature_set_next[0:int(len(feature_set_next)*0.8)]))
    label_set.extend(copy.deepcopy(label_set_next[0:int(len(feature_set_next)*0.8)]))
    sequence_length.extend(copy.deepcopy(sequence_length_next[0:int(len(feature_set_next)*0.8)]))
    for i in range (0, int(len(feature_set_next)*0.8)):
        for j in range (0, label_set_next[i].shape[0]):
            real_class_train[int(label_set_next[i][j])]=real_class_train[int(label_set_next[i][j])]+1

feature_set_val.extend(copy.deepcopy(feature_set_next[int(len(feature_set_next)*0.8): ]))
label_set_val.extend(copy.deepcopy(label_set_next[int(len(feature_set_next)*0.8): ]))
sequence_length_val.extend(copy.deepcopy(sequence_length_next[int(len(feature_set_next)*0.8): ]))
############################
print('real class amounts')
for i in range(0, real_class_train.shape[0]):
    print('z=%d is %d'%(i, real_class_train[i]))
    
############################
print('-ve data:%d'% (len(feature_set)-poz_data))
#########Create Log##############################################################
logfile=open(modelpath+log_no+'.csv', 'wb')
logfile.close()
#######################################################################
fc_size=4
num_class=10
state_size = fc_size
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

#15 x 211
batchX_placeholder = tf.placeholder(tf.float32, [None, RT_window, mz_window]) #image block to consider for one run of training by back propagation
sample_weight = tf.placeholder(tf.float32, [None]) #image block to consider for one run of training by back propagation
keep_prob = tf.placeholder(tf.float32)

W_conv0 = weight_variable([8, 10 , 1, 8], 'W_conv0')#v10: 23x 202
b_conv0 = bias_variable([8], 'b_conv0') #for each of feature maps

W_conv1 = weight_variable([4, 10 , 8, 16], 'W_conv1')# #20x193
b_conv1 = bias_variable([16], 'b_conv1') #for each of feature maps

W_conv2 = weight_variable([4, 8, 16, 32],  'W_conv2')  #18x186
b_conv2 = bias_variable([32], 'b_conv2') 

#W_conv3 = weight_variable([2, 4, 32, 32], 'W_conv3')  #16x183
#b_conv3 = bias_variable([32], 'b_conv3')

#W_fc1 = weight_variable([1 * 183 * 32, 264], 'W_fc1')
#b_fc1 = bias_variable([264], 'b_fc1')

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

##############################
x_image = tf.reshape(batchX_placeholder[:, :, :], [-1, RT_window, mz_window, 1]) #
        
if (activation_func==1):       
    h_conv0 = tf.nn.relu(conv2d(x_image, W_conv0) + b_conv0) # now the input is: (15-8+1) x (211-22+1) x 16 = 8 x 190 x 16
    h_conv1 = tf.nn.relu(conv2d(h_conv0, W_conv1) + b_conv1) # now the input is: (8-4+1) x (190-6+1) x 16 = 5 x 185 x 16
    h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2) # now the input is: (5-3+1) x (185-4+1) x 8 = 3  x 182  x 8
#    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3) #3-3+1 x 182-3+1 x 8 = 1 x 180 x 8
    h_conv2_flat = tf.reshape(h_conv2, [-1, 2 * 186  * 32])
    h_conv2_flat_drop = tf.nn.dropout(h_conv2_flat, keep_prob)
    h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat_drop, W_fc1) + b_fc1) # finally giving the output
#        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2) # finally giving the output
#        h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)
    y_conv = tf.nn.relu(tf.matmul(h_fc1, W_out) + b_out) # finally giving the output
    ##############################
    current_FC  =  tf.nn.dropout(y_conv, keep_prob) # [batch_size, fc_size])
    weighted_state = tf.matmul(current_state, W) # Broadcasted addition #shape?? # EDIT
    next_state = tf.nn.relu(weighted_state + current_FC)  # Broadcasted addition #shape?? # EDIT
else:   
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
    current_FC  = y_conv #tf.nn.dropout(y_conv, keep_prob) # [batch_size, fc_size])
    weighted_state = tf.matmul(current_state, W) # Broadcasted addition #shape?? # EDIT
    next_state = tf.tanh(weighted_state + current_FC)  # Broadcasted addition #shape?? # EDIT
    


logit = tf.matmul(next_state, W2) + b2  #Broadcasted addition
#predictions_series = [tf.nn.softmax(logits) for logits in logits_series]
predictions_series = tf.argmax(tf.nn.softmax(logit), 1) 

loss=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=batchY_placeholder[:]) # [batch_size,loss]  
    
considered_loss=tf.multiply(sample_weight, loss)    
total_loss=tf.reduce_sum(considered_loss) / tf.to_float(tf.reduce_sum(sample_weight))
train_step = tf.train.AdagradOptimizer(.01).minimize(total_loss)


####################################################################################
#sess = tf.Session()
#sess.run(tf.global_variables_initializer())

config=tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
#sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
#saver.save(sess, modelpath+log_no+'_init.ckpt')
saver.restore(sess, modelpath+log_no+'_best_sen_model.ckpt')
########################################
print('starting validation')
accuracy_measure=np.zeros((1, num_class+1))
confusion_matrix=np.zeros((num_class, num_class))
real_class=np.zeros((num_class))                    
batch_size_val=10000 #len(feature_set_val) #
count_batch_val=0
avg_loss_val=0

total_feature_val=len(feature_set_val)
number_of_batch_val=total_feature_val//batch_size_val
count_batch_val=count_batch_val+number_of_batch_val
_current_state_val = np.zeros((batch_size_val, state_size))
ftr=0
for batch_idx_val in range (0,  number_of_batch_val):
    start_ftr=ftr
    batch_ms1_val=np.zeros((batch_size_val,total_frames_var, RT_window,mz_window))
    batch_label_val=np.zeros((batch_size_val, total_frames_var)) 
    batch_prediction_val=np.zeros((batch_size_val, total_frames_var)) 
    sequence_length_mask_val=np.zeros((batch_size_val, total_frames_var))  
    count_val=0
    while count_val!=batch_size_val:
        for i in range (0, sequence_length_val[ftr]):
            batch_ms1_val[count_val, i, :, :]=np.copy(feature_set_val[ftr][i:i+RT_window, :])
        
        batch_label_val[count_val, :]=np.copy(label_set_val[ftr])
        sequence_length_mask_val[count_val, 0:sequence_length_val[ftr]]=1 #sequence_length[ftr]=[1-total_frames_var]
        count_val=count_val+1
        ftr=ftr+1
        
    # one batch is formed
#                    _current_state_val = np.zeros((batch_size_val, state_size))               
    for row_idx in range(0, total_frames_var): # total_hops_horizontal=87 in each hop, 6 windows are considered as truncated backprop length is 6
        batchX = batch_ms1_val[:,row_idx,:,:]                    
        batchY = batch_label_val[:,row_idx]                    
        batch_weight=sequence_length_mask_val[:, row_idx]
        _total_loss=0
        if np.sum(batch_weight)==0:
            break
        
        _total_loss, _current_state_val, _predictions_series = sess.run(
            [total_loss, next_state, predictions_series],
            feed_dict={
                batchX_placeholder:batchX,
                batchY_placeholder:batchY ,
                init_state:_current_state_val, 
                sample_weight:batch_weight, 
                keep_prob:1.0
            })                                        
        
        avg_loss_val=avg_loss_val+_total_loss
        batch_prediction_val[:, row_idx]=_predictions_series[:]
        
    avg_loss_val=avg_loss_val/total_frames_var
    for b in range (0, batch_size_val):
        for row_idx in range (0, sequence_length_val[start_ftr+b]):
            real_charge=int(batch_label_val[b, row_idx])
            pred_charge=int(batch_prediction_val[b, row_idx])
            real_class[real_charge]=real_class[real_charge]+1
            confusion_matrix[real_charge, pred_charge]=confusion_matrix[real_charge, pred_charge]+1    

    #one batch is done 
avg_loss_val=avg_loss_val/number_of_batch_val    
for i in range (0, num_class):
    print("avg accuracy for z=%d is %g, amount %d"%(i, confusion_matrix[i, i]/real_class[i], real_class[i]))
    accuracy_measure[0, i]=confusion_matrix[i, i]/real_class[i]

accuracy_measure[0, num_class]=avg_loss_val
avg_sensitivity=sum(accuracy_measure[0, 0:6])/num_class

print('avg loss %g, avg sensitivity %g'%(avg_loss_val, avg_sensitivity) )    
max_sensitivity=avg_sensitivity
min_loss=avg_loss_val


################ training ############################################################
saver.restore(sess, modelpath+log_no+'_epoch.ckpt')
val_start=95
with sess.as_default():    
    for epoch_idx in range(79,100):
        # go to each feature
        start_time=time()
        print("epoch", epoch_idx)
        count_batch=0
        avg_loss=0                
        total_feature=len(feature_set)
        number_of_batch=total_feature//batch_size
        count_batch=count_batch+number_of_batch
        random_pick=np.random.permutation(total_feature)
        r=-1
        _current_state = np.zeros((batch_size, state_size))               
        for batch_idx in range (0,  number_of_batch):
            batch_ms1=np.zeros((batch_size,total_frames_var, RT_window,mz_window))
            batch_label=np.zeros((batch_size, total_frames_var))            
            sequence_length_mask=np.zeros((batch_size, total_frames_var))  
            count=0
            while count!=batch_size:
                r=r+1
                ftr=random_pick[r]
                for i in range (0, sequence_length[ftr]):
                    batch_ms1[count, i, :, :]=np.copy(feature_set[ftr][i:i+RT_window, :])
                
                batch_label[count, :]=np.copy(label_set[ftr])
                sequence_length_mask[count, 0:sequence_length[ftr]]=1 #sequence_length[ftr]=[1-total_frames_var]
                count=count+1
            # one batch is formed
#                print('batch %d is formed'%batch_idx)

#                accuracy_measure=np.zeros((1, num_class+2))
#                confusion_matrix=np.zeros((num_class, num_class))
#                real_class=np.zeros((num_class))
#                class_loss=np.zeros((1, num_class))
#            _current_state = np.zeros((batch_size, state_size))    
            batch_loss=0
            for row_idx in range(0, total_frames_var): # total_hops_horizontal=87 in each hop, 6 windows are considered as truncated backprop length is 6
                batchX = batch_ms1[:,row_idx,:,:]                    
                batchY = batch_label[:,row_idx]                    
                batch_weight=sequence_length_mask[:, row_idx]
                _total_loss=0
#                print('batch_weight %g'%np.sum(batch_weight))
                if np.sum(batch_weight)==0:
                    break
                
                _total_loss, _train_step, _current_state = sess.run(
                    [total_loss, train_step, next_state],
                    feed_dict={
                        batchX_placeholder:batchX,
                        batchY_placeholder:batchY ,
                        init_state:_current_state, 
                        sample_weight:batch_weight, 
                        keep_prob:0.5
                    })                                        
                
                batch_loss=batch_loss+_total_loss
                
            avg_loss=avg_loss+batch_loss/total_frames_var
            #one batch is done 
################################################################################                
            if (epoch_idx>=val_start and batch_idx%10==0) or (batch_idx==number_of_batch-1 and epoch_idx%5==0):
                print('starting validation')
                accuracy_measure=np.zeros((1, num_class+1))
                confusion_matrix=np.zeros((num_class, num_class))
                real_class=np.zeros((num_class))                    
                batch_size_val=10000 #len(feature_set_val) #
                count_batch_val=0
                avg_loss_val=0
                
                total_feature_val=len(feature_set_val)
                number_of_batch_val=total_feature_val//batch_size_val
                count_batch_val=count_batch_val+number_of_batch_val
                _current_state_val = np.zeros((batch_size_val, state_size))
                ftr=0
                for batch_idx_val in range (0,  number_of_batch_val):
                    start_ftr=ftr
                    batch_ms1_val=np.zeros((batch_size_val,total_frames_var, RT_window,mz_window))
                    batch_label_val=np.zeros((batch_size_val, total_frames_var)) 
                    batch_prediction_val=np.zeros((batch_size_val, total_frames_var)) 
                    sequence_length_mask_val=np.zeros((batch_size_val, total_frames_var))  
                    count_val=0
                    while count_val!=batch_size_val:
                        for i in range (0, sequence_length_val[ftr]):
                            batch_ms1_val[count_val, i, :, :]=np.copy(feature_set_val[ftr][i:i+RT_window, :])
                        
                        batch_label_val[count_val, :]=np.copy(label_set_val[ftr])
                        sequence_length_mask_val[count_val, 0:sequence_length_val[ftr]]=1 #sequence_length[ftr]=[1-total_frames_var]
                        count_val=count_val+1
                        ftr=ftr+1
                        
                    # one batch is formed
#                    _current_state_val = np.zeros((batch_size_val, state_size))               
                    for row_idx in range(0, total_frames_var): # total_hops_horizontal=87 in each hop, 6 windows are considered as truncated backprop length is 6
                        batchX = batch_ms1_val[:,row_idx,:,:]                    
                        batchY = batch_label_val[:,row_idx]                    
                        batch_weight=sequence_length_mask_val[:, row_idx]
                        _total_loss=0
                        if np.sum(batch_weight)==0:
                            break
                        
                        _total_loss, _current_state_val, _predictions_series = sess.run(
                            [total_loss, next_state, predictions_series],
                            feed_dict={
                                batchX_placeholder:batchX,
                                batchY_placeholder:batchY ,
                                init_state:_current_state_val, 
                                sample_weight:batch_weight, 
                                keep_prob:1.0
                            })                                        
                        
                        avg_loss_val=avg_loss_val+_total_loss
                        batch_prediction_val[:, row_idx]=_predictions_series[:]
                    avg_loss_val=avg_loss_val/total_frames_var
                    for b in range (0, batch_size_val):
                        for row_idx in range (0, sequence_length_val[start_ftr+b]):
                            real_charge=int(batch_label_val[b, row_idx])
                            pred_charge=int(batch_prediction_val[b, row_idx])
                            real_class[real_charge]=real_class[real_charge]+1
                            confusion_matrix[real_charge, pred_charge]=confusion_matrix[real_charge, pred_charge]+1    

                    #one batch is done 
                avg_loss_val=avg_loss_val/number_of_batch_val    
                for i in range (0, num_class):
#                    print("avg accuracy for z=%d is %g, amount %d"%(i, confusion_matrix[i, i]/real_class[i], real_class[i]))
                    accuracy_measure[0, i]=confusion_matrix[i, i]/real_class[i]

                
                accuracy_measure[0, num_class]=avg_loss_val
                avg_sensitivity=sum(accuracy_measure[0, 0:6])/num_class
                print('for epoch %d, batch %d, avg loss %g, avg sensitivity %g'%(epoch_idx,batch_idx, avg_loss_val, avg_sensitivity) )    
                logfile=open(modelpath+log_no+'.csv', 'ab')
                np.savetxt(logfile,accuracy_measure, delimiter=',')
                logfile.close() 
                for i in range (0, num_class):
                    print("avg accuracy for z=%d is %g, amount %d"%(i, confusion_matrix[i, i]/real_class[i], real_class[i]))


                if avg_loss_val<=min_loss:
                    min_loss=avg_loss_val
                    print('best_loss found')
                    saver.save(sess, modelpath+log_no+'_best_loss_model.ckpt')

                if avg_sensitivity>=max_sensitivity:
                    max_sensitivity=avg_sensitivity
                    #save the model
                    saver.save(sess, modelpath+log_no+'_best_sen_model.ckpt')
                    print('best_sen found')
#                    for i in range (0, num_class):
#                        print("avg accuracy for z=%d is %g, amount %d"%(i, confusion_matrix[i, i]/real_class[i], real_class[i]))
                    
        elapsed_time=time()-start_time
        print('elapsed time:%g, total_batch %d, avg_loss for training %g'%(elapsed_time, count_batch, avg_loss/count_batch))
        saver.save(sess, modelpath+log_no+'_epoch.ckpt')



