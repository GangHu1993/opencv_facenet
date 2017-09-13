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
import sys

import sklearn

from sklearn.externals import joblib

#face detection parameters
minsize = 20 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor

#facenet embedding parameters

model_dir='../model_check_point/model.ckpt-500000'#"Directory containing the graph definition and checkpoint files.")
model_def= 'models.nn4'  # "Points to a module containing the definition of the inference graph.")
image_size=96 #"Image size (height, width) in pixels."
pool_type='MAX' #"The type of pooling to use for some of the inception layers {'MAX', 'L2'}.
use_lrn=False #"Enables Local Response Normalization after the first layers of the inception network."
seed=42,# "Random seed."
batch_size= None # "Number of images to process in a batch."

input_img_name = '../images/input/3115.JPG'
output_img_name = '../images/out/700.jpg'
write_file_name = '../images/fecture/face_fecture.fec'
frame_interval=1 # frame intervals  

def write_fecture(file_path, fecture_content):
	f=open(write_file_name,'w')
	f.write(fecture_content)
	f.close()

def to_rgb(img):
	 w, h = img.shape
	 ret = np.empty((w, h, 3), dtype=np.uint8)
	 ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
	 return ret

#def build_mtcnn():
def main():
	print('begin build mtcnn')
	gpu_memory_fraction=0.5
	with tf.Graph().as_default():
	    	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
    		sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    		with sess.as_default():
        		pnet, rnet, onet = detect_face.create_mtcnn(sess, '../model_check_point/')
	print('finish build mtcnn')

	print('begin build facenet embedding')
	tf.Graph().as_default()
	sess = tf.Session()
	images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, 3), name='input')

	phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
	
	embeddings = network.inference(images_placeholder, pool_type, use_lrn, 1.0, phase_train=phase_train_placeholder)

	ema = tf.train.ExponentialMovingAverage(1.0)
	saver = tf.train.Saver(ema.variables_to_restore())
	#ckpt = tf.train.get_checkpoint_state(os.path.expanduser(model_dir))
	#saver.restore(sess, ckpt.model_checkpoint_path)

	model_checkpoint_path='../model_check_point/model-20160506.ckpt-500000'
	#ckpt = tf.train.get_checkpoint_state(os.path.expanduser(model_dir))
	#model_checkpoint_path='model-20160506.ckpt-500000'

	#saver.restore(sess, ckpt.model_checkpoint_path)
	saver.restore(sess, model_checkpoint_path)
	print('finish build facenet embedding')

	#restore pre-trained knn classifier
	model = joblib.load('../model_check_point/knn_classifier.model')

	#obtaining frames from camera--->converting to gray--->converting to rgb
	#--->detecting faces---->croping faces--->embedding--->classifying--->print
#	tf.Graph().as_default()
#	sess = tf.Session()
	'''
	print('reading camera')
	video_capture = cv2.VideoCapture(0)'''
	print('reading image')
	im = mpimg.imread(input_img_name)
	gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

	start_time = datetime.datetime.now()
	print('begin time:'+str(start_time))
	
	img = to_rgb(gray)
	
	find_results = []

	print ('dect face...')
	bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
	print ('after dect...')
	nrof_faces = bounding_boxes.shape[0]#number of faces
	print ('face recongni....')
	face_counter = 1


	for face_position in bounding_boxes:
		face_position=face_position.astype(int)
		cv2.rectangle(im, (face_position[0], face_position[1]), (face_position[2], face_position[3]), (0, 255, 0), 2)
		crop=img[face_position[1]:face_position[3],face_position[0]:face_position[2],]
		if crop.shape[0]<=0 or crop.shape[1]<=0:
			break
		crop = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_CUBIC )
		data=crop.reshape(-1,96,96,3)
		emb_data = sess.run([embeddings], feed_dict={images_placeholder: np.array(data), phase_train_placeholder: False })[0]
		predict = model.predict(emb_data)
		print ('predict:')
		print (predict)
		if predict == 1:
			find_results.append('ZhangYanMing')
		elif predict ==2:
			find_results.append('WangBo')
		elif predict ==3:
			find_results.append('Caiwu')
		elif predict ==4:
			find_results.append('SongZuoHua')
		else :
			find_results.append('Other')
	print ('put txt...')
	cv2.putText(im,'detected:{}'.format(find_results), (250,250), cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (255, 0 ,0), thickness = 2, lineType = 2)
	out_img = Image.fromarray(im,'RGB')
	out_img.save(output_img_name)

	end_time = datetime.datetime.now()
	print('end time:'+str(end_time))
	print('run time:'+str((end_time - start_time).seconds))

if __name__ == '__main__':
	main()
