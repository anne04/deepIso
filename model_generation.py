# The original script was developed for Tensorflow Version 1
# To make the script compatible with newer versions, we have commented out line 7, and use line 8 and line 9
# Also training step line 153 is commented out to avoid version problems. 
# So if you need to retrain this model, then you may need to change the script according to the newer versions
from __future__ import division
from __future__ import print_function
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import os
########## file run parameters #################################
current_path=os.system("pwd")
modelpath=current_path+'/DeepIsoV1/model/'
################## deep learning ##############################################
truncated_backprop_length = 5
total_frames_hor=truncated_backprop_length
total_hops_horizontal= total_frames_hor//truncated_backprop_length 
num_class=total_frames_hor # number of isotopes to report
drop_out_k=0.5
RT_window=15
mz_window=11
frame_width=11
mz_unit=0.01
RT_unit=0.01

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


####################################################################
class isoGrouping_model:
    
    def __init__(self, state_size, fc_size,  learn_rate, model_name):
        self.my_graph= tf.Graph()
        with self.my_graph.as_default():        
            self.state_size=state_size
            self.learn_rate=learn_rate
            self.fc_size=fc_size

            self.batchX_placeholder = tf.placeholder(tf.float32, [None, RT_window, mz_window*truncated_backprop_length]) #image block to consider for one run of training by back propagation
            self.keep_prob = tf.placeholder(tf.float32)

            # each image is 15 x 11
            self.W_conv0 = weight_variable([2, 2 , 1, 8], 'W_conv0')#v10: 
            self.b_conv0 = bias_variable([8], 'b_conv0') #15-2+1=14,11-2+1=10
            # pool - 7, 5

            self.W_conv1 = weight_variable([2, 2 , 8, 16], 'W_conv1')#v10: 12-4+1=9, 7-4+1=4 # 6,4
            self.b_conv1 = bias_variable([16], 'b_conv1') #for each of feature maps
            # pool - 3, 2

            self.W_conv2 = weight_variable([2, 2 , 16, 32], 'W_conv2')#v10: 12-4+1=9, 7-4+1=4 # 2, 1
            self.b_conv2 = bias_variable([32], 'b_conv2') #for each of feature maps


            self.W_conv3 = weight_variable([2, 1, 32, 64], 'W_conv3')  # 1, 1
            self.b_conv3 = bias_variable([64], 'b_conv3') 


            #2 x 1
            self.W_fc1 = weight_variable([1 * 1 * 64, 64], 'W_fc1') #
            self.b_fc1 = bias_variable([64], 'b_fc1')

            #
            #W_fc2 = weight_variable([128, 256], 'W_fc2') #
            #b_fc2 = bias_variable([256], 'b_fc2')

            self.W_out = weight_variable([64+1, fc_size], 'W_out')
            self.b_out = bias_variable([fc_size], 'b_out')

            #param_loader = tf.train.Saver({'W_conv0': W_conv0, 'W_conv1': W_conv1, 'W_conv2': W_conv2, 'W_conv3': W_conv3, 'W_fc1':W_fc1, 'W_out':W_out, 'b_conv0':b_conv0, 'b_conv1':b_conv1, 'b_conv2':b_conv2, 'b_conv3':b_conv3, 'b_fc1':b_fc1, 'b_out':b_out})

            self.batchY_placeholder = tf.placeholder(tf.float32, [None, num_class])
            self.batchZ_placeholder = tf.placeholder(tf.float32, [None, 1])
            self.batchAUC_placeholder = tf.placeholder(tf.float32, [None, num_class, 1])

            self.init_state = tf.placeholder(tf.float32, [None, self.state_size])

            self.W = tf.Variable(np.random.rand(fc_size+self.state_size, self.state_size), dtype=tf.float32) 
            self.b = tf.Variable(np.zeros((1,self.state_size)), dtype=tf.float32) # 2D RNN

            self.W_attention = tf.Variable(np.random.rand(self.state_size, self.state_size), dtype=tf.float32) 
            self.b_attention = tf.Variable(np.zeros((1,self.state_size)), dtype=tf.float32) #final output


            self.W2 = tf.Variable(np.random.rand(self.state_size, num_class),dtype=tf.float32) #final output
            self.b2 = tf.Variable(np.zeros((1,num_class)), dtype=tf.float32) #final output

            # Forward pass
            self.current_state = self.init_state
            self.states_series = []
            for j in range (0, truncated_backprop_length):
                ##############################
                self.x_image = tf.reshape(self.batchX_placeholder[:, : , mz_window*j : mz_window* (j+1)], [-1, RT_window, mz_window, 1]) #flatten to 2d: row: RT, column: mz

                self.h_conv0 = tf.tanh(conv2d(self.x_image, self.W_conv0) + self.b_conv0) # now the input is: (15-8+1) x (211-22+1) x 16 = 8 x 190 x 16           
                self.h_pool0 = max_pool_2x2(self.h_conv0)    

                self.h_conv1 = tf.tanh(conv2d(self.h_pool0, self.W_conv1) + self.b_conv1) # now the input is: (8-4+1) x (190-6+1) x 16 = 5 x 185 x 16
                self.h_pool1 = max_pool_2x2(self.h_conv1)

                self.h_conv2 = tf.tanh(conv2d(self.h_pool1, self.W_conv2) + self.b_conv2) # now the input is: (8-4+1) x (190-6+1) x 16 = 5 x 185 x 16
            #    h_pool2 = max_pool_2x2(h_conv2)


                self.h_conv3 = tf.tanh(conv2d(self.h_conv2, self.W_conv3) + self.b_conv3) # now the input is: (5-3+1) x (185-4+1) x 8 = 3  x 182  x 8
                self.h_conv3_flat = tf.reshape(self.h_conv3, [-1, 1 * 1  * 64])

            #    h_conv3_flat_drop = tf.nn.dropout(h_conv3_flat, keep_prob)

                self.h_fc1 = tf.tanh(tf.matmul(self.h_conv3_flat, self.W_fc1) + self.b_fc1)
                self.h_fc1_dropout=tf.nn.dropout(self.h_fc1, self.keep_prob)
            #
            #    h_fc2 = tf.tanh(tf.matmul(h_fc1_dropout, W_fc2) + b_fc2)
            #    h_fc2_dropout=tf.nn.dropout(h_fc2, keep_prob)
            # 
                self.h_fc1_dropout_z = tf.concat([self.h_fc1_dropout, self.batchZ_placeholder], 1)
            #        h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)
                self.h_fc2= tf.tanh(tf.matmul(self.h_fc1_dropout_z, self.W_out) + self.b_out) # finally this will connect with RNN
                ##############################
                self.current_FC  = tf.nn.dropout(self.h_fc2, self.keep_prob) # [batch_size, fc_size])
                
                self.FC_and_state_concatenated = tf.concat([self.current_FC, self.current_state], 1) # row --> batch
                self.weighted_FC_state = tf.matmul(self.FC_and_state_concatenated , self.W) + self.b # 
                
                self.cand_next_state = tf.tanh(self.weighted_FC_state)  # ht
                self.at= tf.sigmoid(tf.matmul(self.cand_next_state, self.W_attention) + self.b_attention) # attention of ht 
                self.next_state=  tf.multiply(self.current_state, (1-self.at))+ tf.multiply(self.cand_next_state, self.at) 
                    
                self.states_series.append(self.next_state)
                self.current_state = self.next_state

            self.logit = tf.matmul(self.current_state, self.W2) + self.b2 
            self.prediction = tf.argmax(tf.nn.softmax(self.logit), 1)
            self.decision_array=tf.nn.softmax(self.logit)

            self.loss=tf.nn.softmax_cross_entropy_with_logits(logits=self.logit, labels=self.batchY_placeholder)

                
            self.total_loss = tf.reduce_mean(self.loss)
            #self.train_step = tf.train.AdagradOptimizer(self.learn_rate).minimize(self.total_loss)

        config=tf.ConfigProto(log_device_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config, graph=self.my_graph)

        with self.sess.as_default():
            with self.my_graph.as_default():
                saver = tf.train.Saver()    
                saver.restore(self.sess, modelpath+'trained-model_'+model_name+'_best.ckpt')
      

