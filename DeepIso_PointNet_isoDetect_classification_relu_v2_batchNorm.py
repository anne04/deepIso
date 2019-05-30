# consecutive scan along RT axis
from __future__ import print_function, division
import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt
import pickle
#import math
from time import time
import sys
import copy


log_no='deepIso_pointNet_isoDetect_c_relu_v2_batchNorm' 
#model_load=
activation_func=2
set_lr_rate=0.05
reg_weight=0.001
total_frames_var=20
RT_window=15
mz_window=211
num_class=10
############## load data #################
modelpath='/data/fzohora/dilution_series_syn_pep/model/deepIso_PointNet/'
#datapath='/data/fzohora/dilution_series_syn_pep/deepIso_pointNet/'      #'/data/fzohora/water/' #'/media/anne/Study/study/PhD/bsi/update/data/water/'  #
path='/data/fzohora/dilution_series_syn_pep/'  #'/media/anne/Study/bsi/dilution_series_syn_peptide/feature_list/' #'/data/fzohora/water_raw_ms1/'
dataname=['130124_dilA_1_01','130124_dilA_1_02','130124_dilA_1_03','130124_dilA_1_04', 
'130124_dilA_2_01','130124_dilA_2_02','130124_dilA_2_03','130124_dilA_2_04','130124_dilA_2_05','130124_dilA_2_06','130124_dilA_2_07',
'130124_dilA_3_01','130124_dilA_3_02','130124_dilA_3_03','130124_dilA_3_04','130124_dilA_3_05','130124_dilA_3_06','130124_dilA_3_07',
'130124_dilA_4_01','130124_dilA_4_02','130124_dilA_4_03','130124_dilA_4_04','130124_dilA_4_05','130124_dilA_4_06','130124_dilA_4_07',
'130124_dilA_5_01','130124_dilA_5_02','130124_dilA_5_03','130124_dilA_5_04',
'130124_dilA_6_01','130124_dilA_6_02','130124_dilA_6_03','130124_dilA_6_04',
'130124_dilA_7_01','130124_dilA_7_02','130124_dilA_7_03','130124_dilA_7_04',
'130124_dilA_8_01','130124_dilA_8_02','130124_dilA_8_03','130124_dilA_8_04',
'130124_dilA_9_01','130124_dilA_9_02','130124_dilA_9_03','130124_dilA_9_04',
'130124_dilA_10_01','130124_dilA_10_02', '130124_dilA_10_03', '130124_dilA_10_04', 
'130124_dilA_11_01', '130124_dilA_11_02', '130124_dilA_11_03', '130124_dilA_11_04', 
'130124_dilA_12_01', '130124_dilA_12_02', '130124_dilA_12_03', '130124_dilA_12_04'] 

#########Create Log##############################################################
logfile=open(modelpath+log_no+'.csv', 'wb')
logfile.close()
#######################################################################
#fc_size=4
num_class=10
#state_size = fc_size
#num_neurons= num_class #mz_window*RT_window

def weight_variable(shape, variable_name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=variable_name)

def bias_variable(shape, variable_name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=variable_name)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def bias_variable_Tnet(shape, variable_name):
    initial = tf.constant(np.eye(shape).flatten(), dtype=tf.float32)
    return tf.Variable(initial, name=variable_name)


#def max_pool_2x2(x):
#    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

datapoints=3000

with tf.device('/gpu:'+'0'):
    #15 x 211
    #datapoints=tf.placeholder(tf.int32, [None])
    is_train = tf.placeholder(tf.bool, name="is_train")
    batchX_placeholder = tf.placeholder(tf.float32, [None, datapoints, 3]) #image block to consider for one run of training by back propagation
    #sample_weight = tf.placeholder(tf.float32, [None]) 
    batchY_placeholder = tf.placeholder(tf.int32, [None])
    keep_probability = tf.placeholder(tf.float32)
    learn_rate=tf.placeholder(tf.float32)

    # T-Net
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------
    T_net_W_conv0 = weight_variable([1, 3 , 1, 16], 'W_conv0')# 64 kernels each having [1,3] sized filter.
    T_net_b_conv0 = bias_variable([16], 'b_conv0') #for each of feature maps

    T_net_W_conv1 = weight_variable([1, 1 , 16, 32], 'W_conv1')# #20x193
    T_net_b_conv1 = bias_variable([32], 'b_conv1') #for each of feature maps

    #T_net_W_conv2 = weight_variable([1, 1, 128, 256],  'W_conv2')  #18x186
    #T_net_b_conv2 = bias_variable([256], 'b_conv2') 

    T_net_W_fc0 = weight_variable([32, 64], 'W_fc1')
    T_net_b_fc0 = bias_variable([64], 'b_fc1')

    T_net_W_fc1 = weight_variable([64, 32], 'W_fc1')
    T_net_b_fc1 = bias_variable([32], 'b_fc1')
    #
    T_net_W_out = weight_variable([32, 3*3], 'W_out')
    T_net_b_out = bias_variable_Tnet(3, 'b_out')

    # -------------------

    nw_W_conv0 = weight_variable([1, 3 , 1, 16], 'nw_W_conv0')# 64 kernels each having [1,3] sized filter.
    nw_b_conv0 = bias_variable([16], 'nw_b_conv0') #for each of feature maps

    #--------------------

    F_net_W_conv0 = weight_variable([1, 16 , 1, 16], 'F_W_conv0')# 64 kernels each having [1,3] sized filter.
    F_net_b_conv0 = bias_variable([16], 'F_b_conv0') #for each of feature maps

    F_net_W_conv1 = weight_variable([1, 1 , 16, 32], 'F_W_conv1')# #20x193
    F_net_b_conv1 = bias_variable([32], 'F_b_conv1') #for each of feature maps

    #F_net_W_conv2 = weight_variable([1, 1, 128, 256],  'F_W_conv2')  #18x186
    #F_net_b_conv2 = bias_variable([256], 'F_b_conv2') 

    F_net_W_fc0 = weight_variable([32, 64], 'F_W_fc0')
    F_net_b_fc0 = bias_variable([64], 'F_b_fc0')

    F_net_W_fc1 = weight_variable([64, 32], 'F_W_fc1')
    F_net_b_fc1 = bias_variable([32], 'F_b_fc1')

    F_net_W_out = weight_variable([32, 16*16], 'F_W_out')
    F_net_b_out = bias_variable_Tnet(16, 'F_b_out')

    #--------------------------
    nw_W_conv1 = weight_variable([1, 16 , 1, 16], 'nw_W_conv1')# 64 kernels each having [1,3] sized filter.
    nw_b_conv1 = bias_variable([16], 'nw_b_conv1') #for each of feature maps

    nw_W_conv2 = weight_variable([1, 1 , 16, 32], 'nw_W_conv2')# #20x193
    nw_b_conv2 = bias_variable([32], 'nw_b_conv2') #for each of feature maps

    #nw_W_conv3 = weight_variable([1, 1, 128, 256],  'nw_W_conv3')  #18x186
    #nw_b_conv3 = bias_variable([256], 'nw_b_conv3')  # global feature

    #---------------------------

    nw_W_fc0 = weight_variable([32, 64], 'nw_W_fc0')
    nw_b_fc0 = bias_variable([64], 'nw_b_fc0')

    nw_W_fc1= weight_variable([64, 32], 'nw_W_fc1')
    nw_b_fc1= bias_variable([32], 'nw_b_fc1')

    nw_W_out = weight_variable([32, num_class], 'nw_W_out')
    nw_b_out = bias_variable([num_class], 'nw_b_out')



    if (activation_func==1):       
        print('tor matha')
    else:   
        T_net_mlp_0 = tf.nn.relu(conv2d(tf.layers.batch_normalization(tf.reshape(batchX_placeholder[:, :, :], [-1, datapoints, 3, 1]),training=is_train) , T_net_W_conv0) + T_net_b_conv0) # now the layer is : b x n x 64  
        T_net_mlp_1 = tf.nn.relu(conv2d(tf.layers.batch_normalization(T_net_mlp_0,training=is_train), T_net_W_conv1) + T_net_b_conv1) # now layer is : b x n x 128
    #    T_net_mlp_2 = tf.tanh(conv2d(T_net_mlp_1, T_net_W_conv2) + T_net_b_conv2) # now the layer is : b x n x 1024
        T_net_maxpool_points=tf.nn.max_pool(tf.layers.batch_normalization(T_net_mlp_1,training=is_train), ksize=[1, datapoints, 1, 1], strides=[1, datapoints, 1, 1], padding='SAME')
        T_net_maxpool_points = tf.reshape(T_net_maxpool_points, [-1, 32])
        T_net_fc0 = tf.nn.relu(tf.matmul(T_net_maxpool_points, T_net_W_fc0) + T_net_b_fc0) # finally giving the output
        T_net_fc1 = tf.nn.relu(tf.matmul(tf.layers.batch_normalization(T_net_fc0,training=is_train), T_net_W_fc1) + T_net_b_fc1) # finally giving the output
        T_net_point_transformation_matrix = tf.nn.relu(tf.matmul(tf.layers.batch_normalization(T_net_fc1,training=is_train), T_net_W_out) + T_net_b_out) # finally giving the output [b x 3 x 3]
        ##############################
        
        T_net_point_transformation_matrix = tf.reshape(T_net_point_transformation_matrix, [-1, 3, 3])
        T_net_point_transformation = tf.matmul(tf.layers.batch_normalization(batchX_placeholder,training=is_train), T_net_point_transformation_matrix) # dimension [n x 3]
        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        nw_mlp_0 = tf.nn.relu(conv2d(tf.reshape(T_net_point_transformation[:, :, :], [-1, datapoints, 3, 1]), nw_W_conv0) + nw_b_conv0) # now the layer is : b x n x 64  

        nw_mlp_0 = tf.reshape(tf.layers.batch_normalization(nw_mlp_0,training=is_train),[-1,datapoints,16])
        
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        F_net_mlp_0 = tf.nn.relu(conv2d(tf.reshape(nw_mlp_0[:, :, :], [-1, datapoints, 16, 1]), F_net_W_conv0) + F_net_b_conv0) # now the layer is : b x n x 64  
        F_net_mlp_1 = tf.nn.relu(conv2d(tf.layers.batch_normalization(F_net_mlp_0,training=is_train), F_net_W_conv1) + F_net_b_conv1) # now layer is : b x n x 128
    #    F_net_mlp_2 = tf.tanh(conv2d(F_net_mlp_1, F_net_W_conv2) + F_net_b_conv2) # now the layer is : b x n x 1024
        F_net_maxpool_points=tf.nn.max_pool(tf.layers.batch_normalization(F_net_mlp_1,training=is_train), ksize=[1, datapoints, 1, 1], strides=[1, datapoints, 1, 1], padding='SAME')
        F_net_maxpool_points = tf.reshape(F_net_maxpool_points, [-1, 32])
        F_net_fc0 = tf.nn.relu(tf.matmul(F_net_maxpool_points, F_net_W_fc0) + F_net_b_fc0) # finally giving the output
        F_net_fc1 = tf.nn.relu(tf.matmul(tf.layers.batch_normalization(F_net_fc0,training=is_train), F_net_W_fc1) + F_net_b_fc1) # finally giving the output
        F_net_point_transformation_matrix = tf.nn.relu(tf.matmul(tf.layers.batch_normalization(F_net_fc1,training=is_train), F_net_W_out) + F_net_b_out) # finally giving the output [b x 3 x 3]
        ##############################
        
        F_net_point_transformation_matrix = tf.reshape(F_net_point_transformation_matrix, [-1, 16, 16])
        F_net_point_transformation = tf.matmul(nw_mlp_0 , F_net_point_transformation_matrix) # dimension [n x 64]
        
        # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
        nw_mlp_1 = tf.nn.relu(conv2d(tf.reshape(F_net_point_transformation, [-1, datapoints, 16, 1]), nw_W_conv1) + nw_b_conv1) # now the layer is : b x n x 64  
        nw_mlp_2 = tf.nn.relu(conv2d(tf.layers.batch_normalization(nw_mlp_1,training=is_train), nw_W_conv2) + nw_b_conv2) # now layer is : b x n x 128
    #    nw_mlp_3 = tf.tanh(conv2d(nw_mlp_2, nw_W_conv3) + nw_b_conv3) # now the layer is : b x n x 1024
        nw_maxpool_points=tf.nn.max_pool(tf.layers.batch_normalization(nw_mlp_2,training=is_train), ksize=[1, datapoints, 1, 1], strides=[1, datapoints, 1, 1], padding='SAME')
        nw_maxpool_points = tf.reshape(nw_maxpool_points, [-1, 32])
        
        #---------------------------------------------------------------------------------------------------------
        nw_fc0 = tf.nn.relu(tf.matmul(nw_maxpool_points, nw_W_fc0) + nw_b_fc0) # finally giving the output
        nw_fc0_dropout=tf.nn.dropout(tf.layers.batch_normalization(nw_fc0,training=is_train) , keep_prob=keep_probability)
        nw_fc1 = tf.nn.relu(tf.matmul(nw_fc0_dropout, nw_W_fc1) + nw_b_fc1) # finally giving the output
        nw_fc1_dropout=tf.nn.dropout(tf.layers.batch_normalization(nw_fc1,training=is_train), keep_prob=keep_probability)

    nw_out = tf.matmul(nw_fc1_dropout, nw_W_out) + nw_b_out # finally giving the output [b x num_class]
    prediction = tf.argmax(tf.nn.softmax(nw_out), 1)
    loss=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=nw_out, labels=batchY_placeholder)

    mat_diff=tf.constant(np.eye(F_net_point_transformation_matrix.get_shape()[1].value), dtype=tf.float32) - tf.matmul(F_net_point_transformation_matrix, tf.transpose(F_net_point_transformation_matrix, perm=[0,2,1]))
    L_reg=tf.nn.l2_loss(mat_diff)* reg_weight
    total_loss = tf.reduce_mean(loss)+L_reg
    #with tf.device('/gpu:'+'0'):
    #    train_step = tf.train.AdamOptimizer(learn_rate).minimize(total_loss)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops): #https://towardsdatascience.com/batch-normalization-theory-and-how-to-use-it-with-tensorflow-1892ca0173ad
        train_step = tf.train.AdagradOptimizer(learn_rate).minimize(total_loss)


config=tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.save(sess, modelpath+log_no+'_init.ckpt')
#saver.restore(sess, modelpath+log_no+'_init.ckpt')


#################dataset##################################################################
f=open(path+'feature_list/'+'pointCloud_training_set_small', 'rb')
train_data, train_label=pickle.load(f)
f.close()
#


#--------------------------------------------------------------------
f=open(path+'feature_list/pointCloud_'+dataname[0]+'_pozData', 'rb')
val_data, val_label=pickle.load(f)
f.close()

#---------------------
RT_mz_I_dict=0
train_data_next=0
train_label_next=0
val_data_next=0
val_label_next=0
################################
class_distribution=np.zeros((1, num_class))
for i in range (0,  len(train_label)):
    class_distribution[0, train_label[i]]=class_distribution[0, train_label[i]]+1

for class_index in range (0,  num_class):
    print('class %d is %d'%(class_index, class_distribution[0, class_index]))
####start training##########################################################
batch_size=32
min_loss=1000
#saver.restore(sess, modelpath+log_no+'_epoch.ckpt')
with sess.as_default():    
    for epoch_idx in range(0,100):
        confusion_matrix_train=np.zeros((num_class, num_class))
        real_class_train=np.zeros((num_class))        
        # go to each feature
        start_time=time()
        print("epoch", epoch_idx)
        count_batch=0
        avg_loss=0                
        total_feature=len(train_data)
        number_of_batch=total_feature//batch_size
        random_pick=np.random.permutation(total_feature)
        r=-1
#        _current_state = np.zeros((batch_size, state_size))               
        for batch_idx in range (0,  number_of_batch):
            batch_ms1=np.zeros((batch_size,datapoints, 3))
            batch_label=np.zeros((batch_size))            
            count=0
            while count!=batch_size:
                r=r+1
                ftr=random_pick[r]
                for i in range (0, len(train_data[ftr])):
                    batch_ms1[count, i, :]=train_data[ftr][i]#np.copy(train_data[ftr][i])
                    
                # remaining datapoints are zero
                
                batch_label[count]=train_label[ftr]
                count=count+1

            # one batch is formed
#                print('batch %d is formed'%batch_idx)

                #accuracy_measure_train=np.zeros((1, num_class+2))
#                class_loss=np.zeros((1, num_class))
#            _current_state = np.zeros((batch_size, state_size))    
            for row_idx in range(0, 1): # total_hops_horizontal=87 in each hop, 6 windows are considered as truncated backprop length is 6
                batchX = batch_ms1 #[:,row_idx,:,:]                    
                batchY = batch_label #[:,row_idx]                    
                _total_loss=0                
                _total_loss, _train_step, _prediction = sess.run(
                    [total_loss, train_step,  prediction],
                    feed_dict={
                        batchX_placeholder:batchX,
                        batchY_placeholder:batchY ,
                        keep_probability:0.5, 
                        learn_rate:set_lr_rate,
                        is_train:True
                    })                                        
#                print('batch loss %g'%_total_loss)
#                print("hello")
#                print(sess.run(W))
                
                avg_loss=avg_loss+_total_loss
                batch_prediction=_prediction

            #one batch is done 
            for b in range (0, batch_size):
                real_charge=int(batch_label[b])
                pred_charge=int(batch_prediction[b])
                real_class_train[real_charge]=real_class_train[real_charge]+1
                confusion_matrix_train[real_charge, pred_charge]=confusion_matrix_train[real_charge, pred_charge]+1    
#            print('batch %d done'%batch_idx)
################################################################################                  
            if (batch_idx==number_of_batch-1): #(epoch_idx>=val_start and batch_idx%10==0) or :
                print('starting validation')
                accuracy_measure=np.zeros((1, num_class+2))
                confusion_matrix=np.zeros((num_class, num_class))
                real_class=np.zeros((num_class))                    
                batch_size_val=32 #len(feature_set_val) #
                count_batch_val=0
                avg_loss_val=0
                
                total_feature_val=len(val_label)
                number_of_batch_val=total_feature_val//batch_size_val
                count_batch_val=count_batch_val+number_of_batch_val
                ftr=0
                for batch_idx_val in range (0,  number_of_batch_val):
                    start_ftr=ftr
                    batch_ms1_val=np.zeros((batch_size_val,datapoints, 3))
                    batch_label_val=np.zeros((batch_size_val)) 
                    batch_prediction_val=np.zeros((batch_size_val)) 
                    count_val=0
                    
                    while count_val!=batch_size_val:
                        for i in range (0, len(val_data[ftr])):
                            batch_ms1_val[count_val, i, :]=val_data[ftr][i] #np.copy(val_data[ftr][i])
                            
                        # remaining datapoints are zero
                        
                        batch_label_val[count_val]=val_label[ftr]
                        count_val=count_val+1
                        ftr=ftr+1
                       
                        
                    # one batch is formed
#                    _current_state_val = np.zeros((batch_size_val, state_size))               
                    for row_idx in range(0, 1): # total_hops_horizontal=87 in each hop, 6 windows are considered as truncated backprop length is 6
                        batchX = batch_ms1_val #[:,row_idx,:,:]                    
                        batchY = batch_label_val #[:,row_idx]                    
                        _total_loss=0                        
                        _total_loss, _prediction = sess.run(
                            [total_loss, prediction],
                            feed_dict={
                                batchX_placeholder:batchX,
                                batchY_placeholder:batchY ,
                                keep_probability:1.0, 
                                is_train:False
                            })                                        
                        
                        avg_loss_val=avg_loss_val+_total_loss
                        batch_prediction_val[:]=_prediction[:]
                        
                    for b in range (0, batch_size_val):
                        real_charge=int(batch_label_val[b])
                        pred_charge=int(batch_prediction_val[b])
                        real_class[real_charge]=real_class[real_charge]+1
                        confusion_matrix[real_charge, pred_charge]=confusion_matrix[real_charge, pred_charge]+1    

                    #one batch is done 
                avg_loss_val=avg_loss_val/number_of_batch_val    
                for i in range (0, num_class):
                    print("avg accuracy for z=%d is val: %g, train: %g, amount %d"%(i, confusion_matrix[i, i]/real_class[i],  confusion_matrix_train[i, i]/real_class_train[i], real_class[i]))
                    accuracy_measure[0, i]=confusion_matrix[i, i]/real_class[i]
                    

                
                accuracy_measure[0, num_class]=avg_loss_val # validation loss
                accuracy_measure[0, num_class+1]=avg_loss/number_of_batch # training loss
                print('for epoch %d, batch %d, avg val loss %g'%(epoch_idx,batch_idx, avg_loss_val) )    
                logfile=open(modelpath+log_no+'.csv', 'ab')
                np.savetxt(logfile,accuracy_measure, delimiter=',')
                logfile.close() 

                if avg_loss_val<=min_loss:
                    min_loss=avg_loss_val
#                if avg_sensitivity>=max_sensitivity:
#                    max_sensitivity=avg_sensitivity
                    #save the model
                    saver.save(sess, modelpath+log_no+'_best_model.ckpt')
                    print('best found')
#                    for i in range (0, num_class):
#                        print("avg accuracy for z=%d is %g, amount %d"%(i, confusion_matrix[i, i]/real_class[i], real_class[i]))
                    
        elapsed_time=time()-start_time
        print('elapsed time:%g, total_batch %d, avg training loss %g'%(elapsed_time, count_batch, avg_loss/number_of_batch))
        saver.save(sess, modelpath+log_no+'_epoch.ckpt')


#small
### 25, 26--> all
#f=open(path+'feature_list/pointCloud_'+dataname[25]+'_pozData', 'rb')
#train_data, train_label=pickle.load(f)
#f.close()
#
#for data_index in range (26, 27):
#    f=open(path+'feature_list/pointCloud_'+dataname[data_index]+'_pozData', 'rb')
#    train_data_next, train_label_next=pickle.load(f)
#    f.close()
#    train_data.extend(train_data_next)
#    train_label.extend(train_label_next)
#
## 27, 28, 29--> 1, 4, 5, 6, .., 9
#for data_index in range (27, 30):
#    f=open(path+'feature_list/pointCloud_'+dataname[data_index]+'_pozData', 'rb')
#    train_data_next, train_label_next=pickle.load(f)
#    f.close()
#    for sample_index in range (0, len(train_label_next)): 
#        if train_label_next[sample_index]==2 or train_label_next[sample_index]==3:
#            continue
#        train_data.append(train_data_next[sample_index])
#        train_label.append(train_label_next[sample_index])
#
## 30, 33, 34, 35, 36 --> 4, 5, 6, 7, ..,9
#for data_index in range (30, 37):
#    f=open(path+'feature_list/pointCloud_'+dataname[data_index]+'_pozData', 'rb')
#    train_data_next, train_label_next=pickle.load(f)
#    f.close()
#    for sample_index in range (0, len(train_label_next)): 
#        if train_label_next[sample_index]>=1 and train_label_next[sample_index]<=3:
#            continue
#            
#        if  train_label_next[sample_index]==4:
#            max_repeat=2
#        elif train_label_next[sample_index]==5:
#            max_repeat=12
#        else:
#            max_repeat=25
#        for repeat_time in range (0, max_repeat):
#            train_data.append(train_data_next[sample_index])
#            train_label.append(train_label_next[sample_index])

#f=open(path+'feature_list/pointCloud_'+dataname[0]+'_training_set_small', 'wb')
#pickle.dump([train_data, train_label],f,protocol=2.0)
#f.close()

#big
#f=open(path+'feature_list/pointCloud_'+dataname[25]+'_pozData', 'rb')
#train_data, train_label=pickle.load(f)
#f.close()
#
#for data_index in  range (26, 30):
#    f=open(path+'feature_list/pointCloud_'+dataname[data_index]+'_pozData', 'rb')
#    train_data_next, train_label_next=pickle.load(f)
#    f.close()
#    train_data.extend(train_data_next)
#    train_label.extend(train_label_next)
#
## 27, 28, 29--> 1, 4, 5, 6, .., 9
#for data_index in range (30, 33):
#    f=open(path+'feature_list/pointCloud_'+dataname[data_index]+'_pozData', 'rb')
#    train_data_next, train_label_next=pickle.load(f)
#    f.close()
#    for sample_index in range (0, len(train_label_next)): 
#        if train_label_next[sample_index]==2: # or train_label_next[sample_index]==3:
#            continue
#        train_data.append(train_data_next[sample_index])
#        train_label.append(train_label_next[sample_index])
#
#
#for data_index in range (33, 37):
#    f=open(path+'feature_list/pointCloud_'+dataname[data_index]+'_pozData', 'rb')
#    train_data_next, train_label_next=pickle.load(f)
#    f.close()
#    for sample_index in range (0, len(train_label_next)): 
#        if train_label_next[sample_index]==2 or train_label_next[sample_index]==3:
#            continue
#        train_data.append(train_data_next[sample_index])
#        train_label.append(train_label_next[sample_index])
#
#
## 30, 33, 34, 35, 36 --> 4, 5, 6, 7, ..,9
#for data_index in range (25, 37):
#    f=open(path+'feature_list/pointCloud_'+dataname[data_index]+'_pozData', 'rb')
#    train_data_next, train_label_next=pickle.load(f)
#    f.close()
#    for sample_index in range (0, len(train_label_next)): 
#        if train_label_next[sample_index]>=1 and train_label_next[sample_index]<=3:
#            continue
#            
#        if  train_label_next[sample_index]==4:
#            max_repeat=5
#        if  train_label_next[sample_index]==5: # or train_label_next[sample_index]==6:
#            max_repeat=25
#        else:
#            max_repeat= 25
#        for repeat_time in range (0, max_repeat):
#            train_data.append(train_data_next[sample_index])
#            train_label.append(train_label_next[sample_index])


