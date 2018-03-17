from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import tensorflow as tf
import numpy as np
import os

import time

import cv2

import inference_youpiao as inference2
import count_dist as count



class image_retrieval:

    def __init__(self, args):
        self.args = args
        config = tf.ConfigProto()
        os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu_id #use GPU with ID=0
        config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=config)
        self.siamese = inference2.siamese()
        config.gpu_options.allow_growth = True

        self.count = count.count()

        with self.count.g1.as_default():
            self.sess1 = tf.Session(graph=self.count.g1)
            self.sess1.run(tf.global_variables_initializer())
        with self.count.g2.as_default():
            self.sess2 = tf.Session(graph=self.count.g2)
            self.sess2.run(tf.global_variables_initializer())
        with self.count.g3.as_default():
            self.sess3 = tf.Session(graph=self.count.g3)
            self.sess3.run(tf.global_variables_initializer())
        with self.count.g4.as_default():  
            self.sess4 = tf.Session(graph=self.count.g4)
            self.sess4.run(tf.global_variables_initializer())              
        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(self.sess, self.args.pretrained_model)


        image_mean = np.load(args.image_mean_dir)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=self.sess)
        self.image_mean = np.transpose(image_mean, (0, 2, 3, 1))

        self.img_id = np.load(self.args.img_id)
        dict = np.load(self.args.dict_idna)
        self.dict = dict.tolist()
        path, _ = self.retrieval(path = self.args.initialize_image_dir)


    def retrieval(self,path):
        image=cv2.imread(path)
        
        image=cv2.resize(image, (227,227))
        img = image[np.newaxis]
        img = img - self.image_mean

        start_time = time.time()

        self.retrieved_img_feat, self.predictions = self.sess.run([self.siamese.retrieved_img_feat1, self.siamese.predictions], feed_dict={
            self.siamese.x1: img
            })
        print(self.predictions.shape)
        print(self.predictions)
        name = []
        if self.predictions[0] in self.dict:            
            name.append(self.dict[self.predictions[0]])
        else:
            name.append('')

        with self.count.g1.as_default():
            print(self.retrieved_img_feat.shape)
            dist1 = self.sess1.run(self.count.dist1, feed_dict={
                    self.count.retrieved_img_feat1: self.retrieved_img_feat,
                    })
        with self.count.g2.as_default():
            dist2 = self.sess2.run(self.count.dist2, feed_dict={
                    self.count.retrieved_img_feat2: self.retrieved_img_feat,
                    })
        with self.count.g3.as_default():
            dist3 = self.sess3.run(self.count.dist3, feed_dict={
                    self.count.retrieved_img_feat3: self.retrieved_img_feat,
                    })
        with self.count.g4.as_default():
            dist4 = self.sess4.run(self.count.dist4, feed_dict={
                    self.count.retrieved_img_feat4: self.retrieved_img_feat,
                    })

        dist12 = np.concatenate((dist1[0],dist2[0]),axis = 0)
        dist123 = np.concatenate((dist12,dist3[0]),axis = 0)
        dist = np.concatenate((dist123,dist4[0]),axis = 0)

        sortedIndex_mat=np.argsort(dist)

        img_id2 = [str(img_id1) for img_id1 in self.img_id]
        img_len = len(img_id2)
        img_id2 = [img_id2[sortedIndex_mat[i]] for i in range(img_len)]
        img_id20 = []
        for mm in img_id2:#返回根目录和图片名字
            mm1 = mm.split('/')

            mm2 = mm1[6:9]

            mm3 = '/'.join(mm2)

            img_id20.append(mm3)


        return img_id20, name

