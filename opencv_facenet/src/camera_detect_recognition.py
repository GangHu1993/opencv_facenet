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

frame_interval=1 # frame intervals  


#network_emotion = EmotionRecognition()
#network_emotion.build_network()

#feelings_faces = []
#for index, emotion in enumerate(EMOTIONS):
#    feelings_faces.append(cv2.imread('./emojis/' + emotion + '.png', -1))


def to_rgb(img):
	 w, h = img.shape
	 ret = np.empty((w, h, 3), dtype=np.uint8)
	 ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
	 return ret

def detect_faces(image_name):
    img = image_name
    face_cascade = cv2.CascadeClassifier(detection_model_path)
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img 
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    result = []
    for (x,y,width,height) in faces:
        result.append((x,y,x+width,y+height))
    return result

def read_img(person_dir, f):
	img = cv2.imread(pjoin(person_dir, f))
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	if gray.ndim == 2:
		img = to_rgb(gray)
	return img


def load_data(data_dir):
	data = {}
	pic_ctr = 0
	for guy in os.listdir(data_dir):
		person_dir = pjoin(data_dir,guy)
		curr_pics = [read_img(person_dir, f) for f in os.listdir(person_dir)]
		data[guy] = curr_pics
	return data

def load_face_dict():
	face_dict = []
	for guy in os.listdir(data_dir):
		face_dict.append(guy)
	#face_dict.reverse()
	print (face_dict)
	return face_dict

def predict_face(predict, face_dict):
	face_label = []
	for pre in predict:
		face_label.append(face_dict[pre])
		#print (face_dict[pre])
	return face_label


#def build_mtcnn():
def main():
	print('build facenet embedding')
	with tf.Graph().as_default():
		gpu_options = tf.GPUOptions()
		gpu_options.allow_growth=True
		sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
		with sess.as_default():
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
	face_dict = load_face_dict()
	#data = load_data(data_dir)
	#keys = data.keys()
	#face_dict = keys
	print (face_dict)

	print('reading camera')
	video_capture = cv2.VideoCapture(0)
	frame_counter = 0
	start_time = datetime.datetime.now()
	print('begin time:'+str(start_time))
	while (1):
		ret, frame = video_capture.read()
		if not ret:
			print ('Frame is None. Please check the cammera!')
			break
		#timeF = frame_interval
		if (frame_counter%frame_interval ==0):
			find_results = []
			predict_results = []
			#print ('dect face...')
			bounding_boxes= detect_faces(frame)
			#print (bounding_boxes)
			#print ('face recongni....')
			for face_position in bounding_boxes:
				print ('face recon...')
				#face_position=face_position.astype(int)
				cv2.rectangle(frame, (face_position[0], face_position[1]), (face_position[2], face_position[3]), (0, 255, 0), 2)
				crop=frame[face_position[1]:face_position[3],face_position[0]:face_position[2],]
				if crop.shape[0]<=0 or crop.shape[1]<=0:
					break
				crop = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_CUBIC )
				data=crop.reshape(-1,96,96,3)
				emb_data = sess.run([embeddings], feed_dict={images_placeholder: np.array(data), phase_train_placeholder: False })[0]
				predict = model.predict(emb_data)
				print (predict)
				#print ('predict:')
				#print (predict)
				predict_results.append(predict[0])
			predict_label = predict_face(predict_results, face_dict)
			#print ('put txt...')
			cv2.putText(frame,'detected:{}'.format(predict_label), (50,100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0 ,0), thickness = 2, lineType = 2)
		frame_counter +=1
		cv2.imshow('Video', frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	video_capture.release()
	cv2.destroyAllWindows()

	end_time = datetime.datetime.now()
	print('end time:'+str(end_time))
	print('frame number'+str(frame_counter))
	print('run time:'+str((end_time - start_time).seconds))

if __name__ == '__main__':
	main()
