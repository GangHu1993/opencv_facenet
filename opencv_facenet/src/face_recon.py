import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import os
from os.path import join as pjoin
import sys
import copy
import detect_face
import nn4 as network
import random
import datetime

import sklearn

from sklearn.externals import joblib

#face detection parameters
minsize = 20 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor
frame_interval = 1

#facenet embedding parameters
detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
model_dir='../model_check_point/model.ckpt-500000'#"Directory containing the graph definition and checkpoint files.")
model_def= 'models.nn4'  # "Points to a module containing the definition of the inference graph.")
image_size=96 #"Image size (height, width) in pixels."
pool_type='MAX' #"The type of pooling to use for some of the inception layers {'MAX', 'L2'}.
use_lrn=False #"Enables Local Response Normalization after the first layers of the inception network."
seed=42,# "Random seed."
batch_size= None # "Number of images to process in a batch."
data_dir='../train_dir/'#your own train folder


print('build facenet embedding')
with tf.Graph().as_default():
	gpu_options = tf.GPUOptions()
	gpu_options.allow_growth=True
	sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
	with sess.as_default():
		pnet, rnet, onet = detect_face.create_mtcnn(sess, '../model_check_point/')
tf.Graph().as_default()
sess = tf.Session()
images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, 3), name='input')
phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
embeddings = network.inference(images_placeholder, pool_type, use_lrn, 1.0, phase_train=phase_train_placeholder)
ema = tf.train.ExponentialMovingAverage(1.0)
saver = tf.train.Saver(ema.variables_to_restore())
model_checkpoint_path='../model_check_point/model-20160506.ckpt-500000'
#saver.restore(sess, ckpt.model_checkpoint_path)
saver.restore(sess, model_checkpoint_path)
#restore pre-trained knn classifier
model = joblib.load('../model_check_point/knn_classifier.model')
#face_dict = load_face_dict()

class Face_recon:
	def __init__(self):
		print ('init_____________________')
		self.frame_counter = 0

	def det_face_reco(self,frame):
		print ('begin')
		if (self.frame_counter%frame_interval ==0):
			find_results = []
			predict_results = []
			print ('dect face...')
			#bounding_boxes= detect_faces(frame)
			bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
			print (bounding_boxes)
			print ('face recongni....')
			for face_position in bounding_boxes:
				face_position=face_position.astype(int)
				#cv2.rectangle(frame, (face_position[0], face_position[1]), (face_position[2], face_position[3]), (0, 255, 0), 2)
				crop=frame[face_position[1]:face_position[3],face_position[0]:face_position[2],]
				if crop.shape[0]<=0 or crop.shape[1]<=0:
					break
				crop = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_CUBIC )
				data=crop.reshape(-1,96,96,3)
				emb_data = sess.run([embeddings], feed_dict={images_placeholder: np.array(data), phase_train_placeholder: False })[0]
				predict = model.predict(emb_data)
				print ('predict:')
				print (predict)
				predict_results.append(predict[0])
			#predict_label = predict_face(predict_results, face_dict)
		return predict

