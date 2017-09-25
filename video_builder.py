# USAGE
# python detect.py --images images

# import the necessary packages
from __future__ import print_function
import numpy as np
import pandas as pd
pd.set_option('display.width', 1000)
import argparse
import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import errno
import sys

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
filelogger = logging.getLogger('test_result')
filelogger.setLevel(logging.DEBUG)
fh = logging.FileHandler('test_result.log')
fh.setLevel(logging.DEBUG)
filelogger.addHandler(fh) 


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
            
def detect_pedestrian(input_video_path, output_info_path, out_video_path, frame_video_debug_path, detect_info_df):
    

    detect_info_nparray = detect_info_df.values
 
	
	# loop over the video
    cap = cv2.VideoCapture(input_video_path)
    tkline_size = 2
    #fourcc = cv2.VideoWriter_fourcc('MJPG')
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fourcc = cv2.VideoWriter_fourcc(*'FMP4')
    #out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
    #out = cv2.VideoWriter(output_info_path +'/out_video/output.mp4', fourcc, 23.0, (1280,720), False)
   
    
    out = cv2.VideoWriter(out_video_path + '/output.mp4', fourcc, 23.0, (1280,720), True)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
	
    ii=0
    while cap.isOpened():
    	ret,frame = cap.read()
    	if(ret == 0):
    		break
    	ii = ii+1
    	detect_info_nparray_frame = detect_info_nparray [(ii==detect_info_nparray[:,1])]
    	for f in detect_info_nparray_frame:
    		xA = f[5]
    		yA = f[6]
    		xB = f[7]
    		yB = f[8]
    		cv2.rectangle(frame,(xA-1,yA-20),(xB+1,yA),(0,0,0),-1)
    		cv2.rectangle(frame,(xA,yA),(xB,yB),(0,0,0),2)
    		cv2.putText(frame,'Mattia',(xA+3,yA-6), font, 0.5, (0,255,255),2,cv2.LINE_AA)
    	cv2.imwrite(frame_video_debug_path + '/' + str(ii)+'.jpg', frame)
    	out.write(frame)
    
    cap.release()
    out.release()


if(__name__ == '__main__'):


	#input_video_path = 'video_data/WalkByShop1cor.mpg' 
	input_video_path = 'video_data/1.mp4' 
	#input_video_path = 'video_data/2.mp4'
	
	output_info_path = 'video_result'
	out_video_path = output_info_path +'/out_video'
	frame_video_debug_path = output_info_path+'/frame_video_debug'
	mkdir_p(out_video_path)
	mkdir_p(frame_video_debug_path)
	
	detect_info_df = pd.read_csv(output_info_path + '/detect_info.csv')
	detect_pedestrian(input_video_path, output_info_path, out_video_path, frame_video_debug_path, detect_info_df)