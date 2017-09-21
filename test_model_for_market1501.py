# -*- coding: utf-8 -*-
import numpy as np
np.random.seed(1217)
import h5py
import tensorflow as tf
#tf.python.control_flow_ops = tf
from PIL import Image
from keras import backend as K
from keras.models import Model
from keras.layers import Input,Dense,Convolution2D,Activation,MaxPooling2D,Flatten,merge
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.preprocessing import image as pre_image
from make_hdf5_for_market1501 import get_image_path_list, get_list_positive_index_market1501, get_list_negative_index_market1501
from model_for_market1501 import model_def, compiler_def

def random_select_pos(f, data_dir, num):
    
    indexs = list(np.random.choice(range(f.shape[0]),num))
    A = []
    B = []
    for index in indexs:
        path1 = f[index,0]
        path2 = f[index,1]
        #print (str(path1[0:7]) + "(" + path1 + ") - " + str(path2[0:7] + "(" + path2 + ")"))
        A.append(np.array(Image.open(data_dir + '/bounding_box_test/' + path1)))
        B.append(np.array(Image.open(data_dir + '/bounding_box_test/' + path2)))
        
    return np.array(A)/255.,np.array(B)/255.
    
def single_test(model, path1, path2):
    
    A = []
    B = []
    #A.append(np.array(Image.open(path1).convert('RGB')))
    #B.append(np.array(Image.open(path2).convert('RGB')))
    A.append(np.array(Image.open(path1))[:, :, :3]) #remove alpha
    B.append(np.array(Image.open(path2))[:, :, :3]) #remove alpha
    #print (A[0].shape)
    #print (B[0].shape)
    A = np.array(A)/255.
    B = np.array(B)/255.
    
    return model.predict([A,B],batch_size = 100)[:,1]

if __name__ == '__main__':
    print ('default dim order is:' + K.image_dim_ordering())

    model = model_def()
    print ('model definition done.')
    model = compiler_def(model)
    print ('model compile done.')
    
    model.load_weights('weights.1/weights_on_market1501_0_0_2.h5')
    #print (model.to_json())
    
    data_dir = 'dataset/market-1501'
    training_set_positive_index_market1501 = get_list_positive_index_market1501('train', data_dir)
    test_set_positive_index_market1501 = get_list_positive_index_market1501('test', data_dir)
    test_set_negative_index_market1501 = get_list_negative_index_market1501('test', data_dir)
    num = 2000
    A,B = random_select_pos(test_set_positive_index_market1501, data_dir, num)
    pred_pos =  model.predict([A,B],batch_size = 100)[:,1]
    print("pred_pos: " + str(np.mean(pred_pos)))
    
    A,B = random_select_pos(test_set_negative_index_market1501, data_dir, num)
    pred_neg =  model.predict([A,B],batch_size = 100)[:,1]
    print("pred_neg: " + str(np.mean(pred_neg)))
    
    #print(pred)
    #print(np.mean(pred))
    
    path1 = 'dataset/market-1501/bounding_box_test/0194_c6s1_059901_04.jpg'
    path2 = 'dataset/market-1501/bounding_box_test/1148_c4s5_025904_02.jpg'
    pred = single_test(model, path1, path2)
    print("pedestrian diversi " +  str(pred))
    
    dictionary_test = ['01.jpg', '02.jpg'] 
    path_test = 'dataset/pedestrian-reidentification-model-trainer/test/01_002.jpg'
    path_dict = 'dataset/pedestrian-reidentification-model-trainer/dictionary/'
    test_set = ['01_001.jpg' , '01_002.jpg',  '01_003.jpg',  '02_001.jpg',  '02_002.jpg',  '02_003.jpg',  '99_001.jpg',  '99_002.jpg',  '99_003.jpg']
    
    for t in test_set:
        path_img_test = 'dataset/pedestrian-reidentification-model-trainer/test/'+ t
        for d in dictionary_test:
            path_img_dict = 'dataset/pedestrian-reidentification-model-trainer/dictionary/'+d
            pred = single_test(model, path_img_test, path_img_dict)
            print(t,  d,  str(pred))
        print ("--------------")
    
        
    """
    pred = single_test(model, path1, path2)
    print("pedestrian 02 -> 01_002 : " +  str(pred))
    
    path1 = 'dataset/pedestrian-reidentification-model-trainer/dictionary/01.jpg'
    path2 = 'dataset/pedestrian-reidentification-model-trainer/test/99_001.jpg'
    pred = single_test(model, path1, path2)
    print("pedestrian 01 -> 99_001 : " +  str(pred))
    
    path1 = 'dataset/pedestrian-reidentification-model-trainer/dictionary/02.jpg'
    path2 = 'dataset/pedestrian-reidentification-model-trainer/test/99_001.jpg'
    pred = single_test(model, path1, path2)
    print("pedestrian 02 -> 99_001 : " +  str(pred))

    a= np.array(Image.open(path1))[:, :, :3]
    img = Image.fromarray(a, 'RGB')
    img.save('dataset_a.jpg')
    print(a)
    b= np.array(Image.open(path2))[:, :, :3]
    img = Image.fromarray(b, 'RGB')
    img.save('dataset_n.jpg')
    print(b)
    """
    
    
    
    
    
