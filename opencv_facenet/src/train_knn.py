import tensorflow as tf
import numpy as np
import cv2

import os
from os.path import join as pjoin
import sys
import copy
import detect_face
import nn4 as network
import matplotlib.pyplot as plt


import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics  
from sklearn.externals import joblib

#pylab inline

#face detection parameters
minsize = 20 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor

#facenet embedding parameters
data_dir='../train_dir/'#your own train folder
model_dir='../model_check_point/model.ckpt-500000'#"Directory containing the graph definition and checkpoint files.")
model_def= 'models.nn4'  # "Points to a module containing the definition of the inference graph.")
image_size=96 #"Image size (height, width) in pixels."
pool_type='MAX' #"The type of pooling to use for some of the inception layers {'MAX', 'L2'}.
use_lrn=False #"Enables Local Response Normalization after the first layers of the inception network."
seed=42,# "Random seed."
batch_size= None # "Number of images to process in a batch."


#convert image to RGB
def to_rgb(img):
	w, h = img.shape
	ret = np.empty((w, h, 3), dtype=np.uint8)
	ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
	return ret

#read image
def read_img(person_dir,f):
	img=cv2.imread(pjoin(person_dir, f))
	#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	#if gray.ndim == 2:
        #	img = to_rgb(gray)
	return img

#load data
def load_data(data_dir):
	data = {}
	pics_ctr = 0
	for guy in os.listdir(data_dir):
		person_dir = pjoin(data_dir, guy)
	        curr_pics = [read_img(person_dir, f) for f in os.listdir(person_dir)]
		data[guy] = curr_pics
	return data

def svm_classifier(train_x, train_y):    
	from sklearn.svm import SVC    
	model = SVC(kernel='rbf', probability=True)    
	model.fit(train_x, train_y)    
	return model 

# KNN Classifier  
def knn_classifier(train_x, train_y):  
	from sklearn.neighbors import KNeighborsClassifier  
	model = KNeighborsClassifier()
	print (model) 
	model.fit(train_x, train_y)  
	return model  

#building the facenet model and loading arguments
def train():
	print('Creating networks and loading parameters')
	gpu_memory_fraction=1.0
	with tf.Graph().as_default():
	    	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
    		sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
	    	with sess.as_default():
        		pnet, rnet, onet = detect_face.create_mtcnn(sess, '../model_check_point/')

	#building facenet embedding model
	print('Building facenet embedding model')
	tf.Graph().as_default()
	sess = tf.Session()
	images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, 3), name='input')
	phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
	embeddings = network.inference(images_placeholder, pool_type, use_lrn, 1.0, phase_train=phase_train_placeholder)
	ema = tf.train.ExponentialMovingAverage(1.0)
	saver = tf.train.Saver(ema.variables_to_restore())
	model_checkpoint_path='../model_check_point/model-20160506.ckpt-500000'
	saver.restore(sess, model_checkpoint_path)
	print('Finished facenet embedding building')

	#loading data and print
	print('Loading data')
	data=load_data(data_dir)
	keys=data.keys()
	print(keys)
	'''
	keys=[]
	for key in data.iterkeys():
	 	keys.append(key)
    		print('foler:{},image numbers:{}'.format(key,len(data[key])))
	print('Finish load data')
	'''

	train_x=[]
	train_y=[]

	counter = 0
	for k, imgs in data.items():
		print(k)
		print (counter)
		for y in imgs:
			bounding_boxes, _ = detect_face.detect_face(y, minsize, pnet, rnet, onet, threshold, factor)
			nrof_faces = bounding_boxes.shape[0]#number of faces
			for face_position in bounding_boxes:
				print (counter)
				face_position=face_position.astype(int)
				cv2.rectangle(y, (face_position[0], face_position[1]), (face_position[2], face_position[3]), (0, 255, 0), 2)
				crop=y[face_position[1]:face_position[3],face_position[0]:face_position[2],]
				if crop.shape[0]<=0 or crop.shape[1]<=0:
					break
				crop = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_CUBIC )
				data=crop.reshape(-1,96,96,3)
				emb_data = sess.run([embeddings], feed_dict={images_placeholder: np.array(data), phase_train_placeholder: False })[0]
				train_x.append(emb_data)
				train_y.append([counter])
		counter = counter + 1

	print('Number of sample:{}'.format(len(train_x)))

	#train/test split
	train_x=np.array(train_x)
	train_x=train_x.reshape(-1,128)
	train_y=np.array(train_y)
	print(train_x.shape)
	print(train_y.shape)


	X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=.33, random_state=42)
	print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)

	# train knn classifier
	classifiers = knn_classifier

	model = classifiers(X_train,y_train)  
	predict = model.predict(X_test)  

	accuracy = metrics.accuracy_score(y_test, predict)  
	print ('accuracy: %.2f%%' % (100 * accuracy)  ) 
  
	#save model
	joblib.dump(model, '../model_check_point/knn_classifier.model')
	#model = joblib.load('_2017_1_24_knn.model')

	model = joblib.load('../model_check_point/knn_classifier.model')
	predict = model.predict(X_test) 
	accuracy = metrics.accuracy_score(y_test, predict)  
	print ('accuracy: %.2f%%' % (100 * accuracy)  ) 


def main():
	train()


if __name__ == '__main__':
	main()
