# USAGE
# python detect.py --images images

# import the necessary packages
from __future__ import print_function
import numpy as np
import pandas as pd
pd.set_option('display.width', 1000)
import argparse
#import imutils
import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
filelogger = logging.getLogger('test_result')
filelogger.setLevel(logging.DEBUG)
fh = logging.FileHandler('test_result.log')
fh.setLevel(logging.DEBUG)
filelogger.addHandler(fh) 
import errno    


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def detect_pedestrian(input_video_path, overlap_thresh, snapshot_rate, resize_min, output_info_path, detection_box_dir, frame_debug_dir, 
						winStride, padding, scale, hitThreshold, finalThreshold, useMeanshiftGrouping):
	
	# initialize the HOG descriptor/person detector
	#http://www.pyimagesearch.com/2015/11/16/hog-detectmultiscale-parameters-explained/
	hog = cv2.HOGDescriptor()
	hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
	
	# loop over the video
	cap = cv2.VideoCapture(input_video_path)
	tkline_size = 2
	

	detect_info_mx = []
	

	ii=0
	while cap.isOpened():
		ret,frame = cap.read()
		if(ret == 0):
			break
			
		ii = ii+1
		if(ii%snapshot_rate != 0):
			continue
		
		frame_gray = frame.copy()
		#frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		frame_copy = frame.copy()

		"""
		if resize_min > 0 :
			frame_gray = imutils.resize(frame_gray, width=min(resize_min, frame_gray.shape[1]))
			frame_copy = imutils.resize(frame_copy, width=min(resize_min, frame_copy.shape[1]))
			frame = imutils.resize(frame, width=min(resize_min, frame.shape[1]))
		"""
		
		
		# detect people in the image 
		#(rects, weights) = hog.detectMultiScale(frame_gray, winStride=(2, 2), padding=(2, 2), scale=1.0)
		(rects, weights) = hog.detectMultiScale(frame_gray, winStride=winStride, padding=padding, 
							scale=scale, hitThreshold=hitThreshold, finalThreshold=finalThreshold, useMeanshiftGrouping=useMeanshiftGrouping)
	
		# apply non-maxima suppression to the bounding boxes using a
		# fairly large overlap threshold to try to maintain overlapping
		# boxes that are still people
		rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
		#pick = non_max_suppression(rects, probs=None, overlapThresh=overlap_thresh)
		pick = rects
	
		frame_file_path = output_info_path + '/frame_debug/frame_'+str(ii)+'.jpg'
		# draw the final bounding boxes
		jj=0
		for (xA, yA, xB, yB) in pick:
			
			frame_num = ii
			crop_file_path = output_info_path+'/detection_box/frame_'+str(ii)+'_crop_'+str(jj)+'.jpg'
			crop_num = jj
			id_tag = ''
			
			crop=frame[(yA+tkline_size):(yB-tkline_size), (xA+tkline_size):(xB-tkline_size)]
			cv2.imwrite(crop_file_path, crop)
			cv2.rectangle(frame_copy, (xA, yA), (xB, yB), (0, 255, 0), tkline_size)
			
			detect_row = [frame_num, frame_file_path, crop_file_path, crop_num, xA, yA, xB, yB, id_tag]
			detect_info_mx.append(detect_row)
			
			jj = jj+1
			
			
		# show some information on the number of bounding boxes
		logging.debug("[INFO] frame {}: {} original boxes, {} after suppression".format(str(ii), len(rects), len(pick)))
			
		cv2.imwrite(frame_file_path, frame_copy)
		
	detect_info_df=pd.DataFrame(detect_info_mx, columns=['FRAME', 'FRAME_FILE_PATH', 'CROP_FILE_PATH', 'CROP_NUM', 'xA', 'yA', 'xB', 'yB', 'ID'])
	detect_info_df.to_csv(output_info_path + '/detect_info.csv')
	print (detect_info_df.head())
	quit()

if(__name__ == '__main__'):

	output_info_path = 'video_result'
	detection_box_dir = output_info_path+'/detection_box'
	mkdir_p(detection_box_dir)
	frame_debug_dir = output_info_path + '/frame_debug'
	mkdir_p(frame_debug_dir)

	#params
	overlap_thresh=0.55
	snapshot_rate = 1#24
	resize_min = -1#500
	
	#input_video_path = 'video_data/WalkByShop1cor.mpg' 
	input_video_path = 'video_data/1.mp4' 
	#input_video_path = 'video_data/2.mp4'
	
	
	winStride=(4, 4)
	padding=(2, 2)
	scale=1.05
	hitThreshold=1
	finalThreshold=2.0
	useMeanshiftGrouping=False
	
	detect_pedestrian(input_video_path, overlap_thresh, snapshot_rate, resize_min, output_info_path, detection_box_dir, frame_debug_dir, winStride, padding, scale, hitThreshold, finalThreshold, useMeanshiftGrouping)
	
	
	
	



