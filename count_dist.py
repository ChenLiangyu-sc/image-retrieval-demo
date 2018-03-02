from datetime import datetime
import math
import time
import tensorflow as tf
import numpy as np
from tensorflow.python.training import moving_averages
import os
class count:
    
    def __init__(self):
        self.g1 = tf.Graph()
        self.g2 = tf.Graph()
        self.g3 = tf.Graph()
        self.g4 = tf.Graph()

        with self.g1.as_default():
            self.eval_feats1 = tf.constant(np.load('./eval_feats1.npy'), dtype = tf.float32)
            self.retrieved_img_feat1 = tf.placeholder(tf.float32,[None,8192])
            self.dist1()
        with self.g2.as_default():
            self.eval_feats2 = tf.constant(np.load('./eval_feats2.npy'), dtype = tf.float32)
            self.retrieved_img_feat2 = tf.placeholder(tf.float32,[None,8192])
            self.dist2()
        with self.g3.as_default():
            self.eval_feats3 = tf.constant(np.load('./eval_feats3.npy'), dtype = tf.float32)
            self.retrieved_img_feat3 = tf.placeholder(tf.float32,[None,8192])
            self.dist3()
        with self.g4.as_default():
            self.eval_feats4 = tf.constant(np.load('./eval_feats4.npy'), dtype = tf.float32)
            self.retrieved_img_feat4 = tf.placeholder(tf.float32,[None,8192])
            self.dist4()

        

    def dist1(self):
        with self.g1.as_default():
            self.dist1 = tf.subtract(1., tf.divide(tf.matmul(self.retrieved_img_feat1, self.eval_feats1), (tf.reshape(tf.sqrt(tf.reduce_sum(tf.pow(tf.transpose(self.eval_feats1,[1,0]),2),1)),[1,-1]))))
    def dist2(self):
        with self.g2.as_default():
            self.dist2 = tf.subtract(1., tf.divide(tf.matmul(self.retrieved_img_feat2, self.eval_feats2), (tf.reshape(tf.sqrt(tf.reduce_sum(tf.pow(tf.transpose(self.eval_feats2,[1,0]),2),1)),[1,-1]))))
    def dist3(self):
        with self.g3.as_default():
            self.dist3 = tf.subtract(1., tf.divide(tf.matmul(self.retrieved_img_feat3, self.eval_feats3), (tf.reshape(tf.sqrt(tf.reduce_sum(tf.pow(tf.transpose(self.eval_feats3,[1,0]),2),1)),[1,-1]))))
    def dist4(self):
        with self.g4.as_default():
            self.dist4 = tf.subtract(1., tf.divide(tf.matmul(self.retrieved_img_feat4, self.eval_feats4), (tf.reshape(tf.sqrt(tf.reduce_sum(tf.pow(tf.transpose(self.eval_feats4,[1,0]),2),1)),[1,-1]))))



