import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import os
from os.path import join as pjoin
import sys
import copy
import random
import datetime

frame_interval=5 # frame intervals  


def main():
	print('reading camera')
	video_capture = cv2.VideoCapture(0)
	frame_counter = 0
	while (1):
		ret, frame = video_capture.read()
		if not ret:
			print ('Frame is None. Please check the camera!')
			break
		print (frame_counter)
		if (frame_counter%frame_interval ==0):
			#out_img = Image.fromarray(frame,'RGB')
			#out_img.save('../train_dir/test/'+str(frame_counter)+'.jpg')
			cv2.imwrite('../train_dir/test/'+str(frame_counter)+'.jpg',frame)
		frame_counter  = frame_counter +1
		if (frame_counter == 1000):
			return 0
		cv2.imshow('Video', frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	video_capture.release()
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()
