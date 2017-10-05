import pandas as pd
import os
import numpy as np

pd.set_option('display.width', 1000)

import datetime
import logging
#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
acc_filelogger = logging.getLogger('accuracy')
acc_filelogger.setLevel(logging.DEBUG)
acc_fh = logging.FileHandler('log/accuracy.log')
acc_fh.setLevel(logging.DEBUG)
acc_filelogger.addHandler(acc_fh) 



if(__name__ == '__main__'):
    video_name = 'cvpr10_tud_stadtmitte'
    detect_info_df = pd.read_csv('video_data/'+video_name+'_gt.csv')
    #print (detect_info_df.head())
    detect_info_nparray = detect_info_df.values
    true_match=0
    nones = 0
    nones_similarity = []
    false_match=0
    acc_filelogger.debug("******* TEST ******")
    for row in detect_info_nparray:
        gt = row[10]
        r = str(row[11])
        rname = r.split(' ')[0]
        rsim = float(r.split(' ')[1])
        if gt == rname:
            true_match=true_match+1
        elif gt == 'none' :
            nones = nones+1
            nones_similarity.append(rsim)
        else:
           false_match=false_match+1
           
        #acc_filelogger.debug('row: %s, gr: %s <-> %s ', row[0], gt, rname)
    
    tot = len(detect_info_nparray)
    nones_similarity = np.array(nones_similarity)
    mean_nones_sim = nones_similarity.mean()
    mean_nones_sim_max = np.amax(nones_similarity)
    mean_nones_sim_min = np.amin(nones_similarity)
    acc_filelogger.debug('tot: %i, true_match: %i, nones: %i, false_match: %i', tot, true_match, nones, false_match)
    acc_filelogger.debug('true_match: %f, nones: %f, false_match: %f', true_match/tot, nones/tot, false_match/tot)
    acc_filelogger.debug('mean_nones_sim: %f, mean_nones_sim_max: %f, mean_nones_sim_min: %f',  mean_nones_sim, mean_nones_sim_max, mean_nones_sim_min)

    

    