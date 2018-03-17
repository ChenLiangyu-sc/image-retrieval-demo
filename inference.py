from datetime import datetime
import math
import time
import tensorflow as tf
import numpy as np
from tensorflow.python.training import moving_averages

class siamese:

    # Create model
    def __init__(self, vggs_npy_path='./vggs1.npy'):

        self.data_dict = np.load(vggs_npy_path, encoding='latin1').item()  
        print("npy file loaded")  
  
        self.mode = 'train'
        self._extra_train_ops = []

        self.x1 = tf.placeholder(tf.float32, [None, 227, 227, 3])
        self.x2 = tf.placeholder(tf.float32, [None, 227, 227, 3])
        self.labels1 = tf.placeholder(tf.int32)
        self.labels2 = tf.placeholder(tf.int32)
        self.keep_prob = tf.constant(1,tf.float32)
        self.y_ = tf.placeholder(tf.float32, [None])

        with tf.variable_scope("siamese") as scope:
            self.o1, self.p1, self.q1, self.k1, self.pool51, self.pool21, self.loss4, self.conv21, self.norm11, self.conv11, self.conv21, self.conv51, self.conv31, self.conv41, self.retrieved_img_feat1 = self.network(self.x1)
            scope.reuse_variables()
            self.o2, self.p2, self.q2, self.k2, self.pool52, self.pool22, self.loss5, self.conv22, self.norm12, self.conv12, self.conv22, self.conv52, self.conv32, self.conv42, self.retrieved_img_feat2 = self.network(self.x2)

        self.predictions = self.loss_With_Softmax1(self.p1)
        
    def conv_op(self, input_op, name, kh, kw, n_out, dh, dw, is_training):
        n_in = input_op.get_shape()[-1].value

        kernel = self.get_conv_filter(name, kh, kw, n_in, n_out)

        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding='SAME')
        bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        biases = self.get_bias(name, n_out)  
        z = tf.nn.bias_add(conv, biases)

        activation = tf.nn.relu(z, name)
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(0.0005)(kernel))
        return activation   

    def conv_op1(self, input_op, name, kh, kw, n_out, dh, dw, is_training):
        n_in = input_op.get_shape()[-1].value

        kernel = self.get_conv_filter(name, kh, kw, n_in, n_out)

        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding='VALID')
        bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        biases = self.get_bias(name, n_out)  
        z = tf.nn.bias_add(conv, biases)

        activation = tf.nn.relu(z, name)
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(0.0005)(kernel))
        return activation 

    def conv_op2(self, input_op, name, kh, kw, n_out, dh, dw, is_training):
        n_in = input_op.get_shape()[-1].value

        kernel = self.get_conv_filter(name)

        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding='VALID')
        bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        activation = tf.nn.relu(z, name)
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(0.0005)(kernel))
        return activation 


    
    def fc_op(self, input_op, name, n_out, is_training):
        assert len(input_op.get_shape()) == 2
        n_prev_weight = input_op.get_shape()[1]
        w = self.get_fc_weight(name, n_prev_weight, n_out)
        b = self.get_bias(name, n_out) 
        fc = tf.add(tf.matmul(input_op, w), b)
        
        fc1 = tf.nn.relu(fc, name)
        
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(0.0005)(w))
        return fc1
   

    
    
    def mpool_op(self, input_op, name, kh, kw, dh, dw):
        shp = input_op.get_shape()
        if (shp[1]-kh)%dh != 0:
            input_op = tf.pad(input_op, [[0,0],[0,1],[0, 1],[0,0]])
            shp = input_op.get_shape()
            if (shp[1]-kh)%dh != 0:
                input_op = tf.pad(input_op, [[0,0],[0,1],[0, 1],[0,0]])
                shp = input_op.get_shape()
        return tf.nn.max_pool(input_op, 
                             ksize=[1, kh, kw, 1],
                             strides=[1, dh, dw, 1],
                             padding='VALID',
                             name=name)
    def mpool_op1(self, input_op, name, kh, kw, dh, dw):

        return tf.nn.max_pool(input_op, 
                             ksize=[1, kh, kw, 1],
                             strides=[1, dh, dw, 1],
                             padding='SAME',
                             name=name)


    def network(self, x):
        conv1 = self.conv_op1(x, name="conv1", kh=7, kw=7, n_out=96, dh=2, dw=2,is_training=True)
        norm1 = tf.nn.lrn(conv1, 2, bias=2, alpha=0.0001, beta=0.75)
        pool1 = self.mpool_op(norm1, name="pool1", kh=3, kw=3, dw=3, dh=3)

        conv2 = self.conv_op1(pool1, name="conv2", kh=5, kw=5, n_out=256, dh=1, dw=1,is_training=True)
        pool2 = self.mpool_op(conv2, name="pool2", kh=2, kw=2, dh=2, dw=2)

        conv3 = self.conv_op(pool2, name="conv3", kh=3, kw=3, n_out=512, dh=1, dw=1,is_training=True)
        conv4 = self.conv_op(conv3, name="conv4", kh=3, kw=3, n_out=512, dh=1, dw=1,is_training=True)
        conv5 = self.conv_op(conv4, name="conv5", kh=3, kw=3, n_out=512, dh=1, dw=1,is_training=True)
        pool5 = self.mpool_op(conv5, name="pool5", kh=3, kw=3, dh=3, dw=3)
        pool6 = self.mpool_op(pool5, name="pool6", kh=3, kw=3, dh=1, dw=1)

        shp = pool5.get_shape()
        flattened_shape = shp[1].value * shp[2].value * shp[3].value
        flattened_shape1 = pool6.get_shape()[1].value * pool6.get_shape()[2].value * pool6.get_shape()[3].value
        resh1 = tf.reshape(pool5, [-1, flattened_shape], name="resh1")

        retrieved_img_feat = tf.reshape(pool6, [-1, flattened_shape1], name="resh2")

        fc6 = self.fc_op(resh1, name="fc6", n_out=4096,is_training=True)  #4096
        fc6_drop = tf.nn.dropout(fc6, self.keep_prob, name = "fc6_drop")
        fc7 = self.fc_op(fc6_drop, name="fc7", n_out=4096,is_training=True)
        fc7_drop = tf.nn.dropout(fc7, self.keep_prob, name="fc7_drop") #4096

        fc8 = self.fc_op(fc7_drop, name="fc8", n_out=1960,is_training=True)
        loss = tf.add_n(tf.get_collection('losses'))
        input_dim = len(fc7_drop.get_shape()) - 1
        fc7_L2Norm = tf.nn.l2_normalize(fc7_drop, input_dim, epsilon=1e-12, name="fc7_L2Norm")
        return fc7_L2Norm ,fc8, resh1, pool6, pool5, pool2, loss, conv2, norm1, conv1, conv2, conv5, conv3, conv4, retrieved_img_feat


    def contrastive_loss(self):
        self.margin = 2.24
        labels_t = self.y_
        labels_f = tf.subtract(1.0, self.y_, name="1-yi")          # labels_ = !labels;
        eucd2 = tf.pow(tf.subtract(self.o1, self.o2), 2)
        eucd2 = tf.reduce_sum(eucd2, 1)
        eucd = tf.sqrt(eucd2+1e-6, name="eucd")
        C = tf.constant(self.margin, name="C")
        pos = tf.multiply(labels_t, eucd2, name="yi_x_eucd2")

        neg = tf.multiply(labels_f, tf.pow(tf.maximum(tf.subtract(C, eucd), 0), 2), name="Nyi_x_C-eucd_xx_2")
        self.posloss = tf.reduce_mean(pos)
        self.negloss = tf.reduce_mean(neg)
        losses = tf.add(pos, neg, name="losses")
        self.loss6 = tf.reduce_mean(losses, name="loss1")
        

    
    
    def loss_With_Softmax(self, input_data, labels, name):
 
        with tf.name_scope(name) as scope:
            labels1 = tf.one_hot(labels, 1960)
            truth = tf.argmax(labels1, axis=1)
            y=tf.nn.softmax(input_data)
            predictions = tf.argmax(y, axis=1)
            loss = tf.reduce_mean(-tf.reduce_sum(labels1*tf.log(y), reduction_indices = [1])) 
            precision = tf.reduce_mean(tf.to_float(tf.equal(predictions, truth))) 

            return loss, precision
    

    def loss_With_Softmax1(self, input_data):

        y=tf.nn.softmax(input_data)
        predictions = tf.argmax(y, axis=1)

        return predictions

    
    def get_conv_filter(self, name, kh, kw, n_in, n_out):  

        if self.data_dict is not None and name in self.data_dict:
            kernel = tf.Variable(self.data_dict[name]['weights'],trainable=True, name = name+"filter")
        else:
            kernel = tf.get_variable(name = name+"filter", shape=[kh, kw, n_in, n_out], dtype=tf.float32,
                                         initializer=tf.contrib.layers.xavier_initializer_conv2d())            
        return kernel  
  
    def get_bias(self, name, n_out):

        if self.data_dict is not None and name in self.data_dict:
            biases = tf.Variable(self.data_dict[name]['biases'], trainable=True, name=name+"biases")
        else:
            bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
            biases = tf.Variable(bias_init_val, trainable=True, name=name+"biases")  

        return biases  
  
    def get_fc_weight(self, name, n_prev_weight, n_out): 

        if self.data_dict is not None and name in self.data_dict:
            weights = tf.Variable(self.data_dict[name]['weights'],trainable=True, name=name+"weights")  
        else:
            initer = tf.truncated_normal_initializer(stddev=0.01)
            weights = tf.get_variable(name+'weights', dtype=tf.float32, shape=[n_prev_weight, n_out], initializer=initer)
            
        return weights
