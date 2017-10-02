#PROJECT STRUCTURE
whe have to develop 4 modules


1)  video-filtering: try generate a new video in b/w and backoground suppession to make easy people detection
    TO SEE
    https://github.com/blacksteed232/PedestrianCounter/blob/master/traffic.py
    

2)  pedestrian-detection-video: this program try to detect people and  for every frame generate a frame with bounded box and a set of 
    images "detected pedestrian" with only pedestrian, this process is useful to speed up detection cause the images "detected pedestrian" 
    are smaller than the entire frame.
    [ALREADY DEVELOPED BY MATTIA]
    
    TO SEE
    http://www.pyimagesearch.com/2015/11/09/pedestrian-detection-opencv/
    http://www.pyimagesearch.com/2015/11/16/hog-detectmultiscale-parameters-explained/
    http://funvision.blogspot.it/2016/03/opencv-31-people-detection-at-13-fps-by.html
    https://github.com/afazel/CS231A_Project
    https://github.com/Luigolas/PyReID
    
    
3)  pedestrian-reidentification-model-trainer: this software train a SVM model aim to get in input hog feature and to predict witch is the 
    pedestrian in a set of pedestrian.
    To train this model we have set of annotated image for every pedestrian
    
    We use a siamese 
    https://github.com/torrvision/siamfc-tf


4) pedestrian-reidentification-model-prediction: this software load a pretrained SVM model and take in input a image "detected pedestrian" and 
    return the prediction telling us which pedestrian is in a set of given people
    
    



#DATASET 
http://robustsystems.coe.neu.edu/sites/robustsystems.coe.neu.edu/files/systems/projectpages/reiddataset.html
https://data.vision.ee.ethz.ch/cvl/aess/dataset/
http://riemenschneider.hayko.at/vision/dataset/
http://www.lorisbazzani.info/caviar4reid.html
http://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/
http://homepages.inf.ed.ac.uk/rbf/CAVIARDATA2/
http://www.liangzheng.com.cn/Datasets.html
http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/
http://vislab.isr.ist.utl.pt/hda-dataset/ [NON RISESCO A SCARICARLO]
https://www.d2.mpi-inf.mpg.de/node/428/



#BIBLIO
http://cvgl.stanford.edu/teaching/cs231a_winter1415/prev/projects/pedestrian.pdf
https://github.com/afazel/CS231A_Project
https://web.stanford.edu/class/cs231a/prev_projects_2016/pedestrian-detection-tracking.pdf



#PLAYGROUND/PRESENTATION SITE:
study that: https://pages.github.com/