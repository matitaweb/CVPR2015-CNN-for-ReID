#! /bin/bash

# Creating environment (sandbox instance called py3 [choose the name you want])
conda create -n py3 python=3 ipython
 
# Activating created environment
source activate py3

# Install package manager pip
conda install pip
# The installation installs the packages
 #pip install numpy
 #pip install pandas
 #pip install matplotlib
# which ipython is to be used in the environment? pip freeze shows it
pip freeze
# Installing ipython notebook
# conda install ipython-notebook

# Installing the packages
sudo conda install -y -c conda-forge numpy 

sudo conda install -y -c conda-forge matplotlib
sudo conda install -y -c anaconda scikit-learn 
 
sudo conda install -y -c conda-forge scipy

sudo conda install -y -c conda-forge opencv

sudo conda install -y -c conda-forge h5py

sudo conda install -y -c conda-forge tensorflow

sudo conda install -y -c conda-forge keras

sudo conda install -y -c conda-forge pillow

#sudo pip install -r https://raw.githubusercontent.com/Lasagne/Lasagne/master/requirements.txt
#sudo pip install https://github.com/Lasagne/Lasagne/archive/master.zip

