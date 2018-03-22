
# coding: utf-8

# In[7]:


import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pylab
from scipy.io import wavfile
from scipy.fftpack import fft
from scipy.signal import stft
from scipy import signal
import pickle
import re
import datetime


# In[2]:


source_dir='/home/ec2-user/juee/datasets'
target_dir_stft_pickles='/home/ec2-user/juee/stft'


# In[3]:


def get_stft(fileName,targetFileName=None):
    if targetFileName is None:
        targetFileName=re.sub('.wav','.pkl',re.sub('datasets','stft',fileName))
    if not os.path.isfile(targetFileName):
        myAudio = fileName
        samplingFreq, mySound = wavfile.read(myAudio)
        mySound = mySound / (2.**15)
        signal_stft=signal.stft(mySound,samplingFreq,'hann')
        with open(targetFileName,'wb') as f:
            pickle.dump(signal_stft,f)
        return signal_stft
    else:
        return 0


# In[4]:


list_of_files=[]
for root, subdirs, files in os.walk(source_dir):
    for each_file in files:
        if re.search('wav',each_file):
            file_name=root+'/'+each_file
            if os.path.isfile(file_name):
                list_of_files.append(file_name)


# In[5]:


len(list_of_files)


# In[8]:


counter=0
for each_file in list_of_files:
    if (counter%250==0):
        print ('file '+ str(counter)+' : '+each_file)
        print (datetime.datetime.now())
    try:
        _=get_stft(each_file)
    except Exception as e:
        print (str(e))
    counter=counter+1

