#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Wei-Cheng (Winston) Lin
"""
import os
import librosa
import numpy as np
from scipy.io import savemat


"""
Input Audio Format: sampling rate=16000, mono channel
Mel-Spec-Feature Configs: 512 window size, 256 step size [32ms with 16ms (50%) overlap], 128-mels
"""

def Mel_Spec(fpath):
    signal, rate  = librosa.load(fpath, sr=16000)
    signal = signal/np.max(abs(signal)) # Restrict value between [-1,1]
    mel_spec = librosa.feature.melspectrogram(signal, sr=16000, n_fft=512, hop_length=256, n_mels=128)
    mel_spec = mel_spec.T
    return mel_spec

def Extract_AcousticFeat(input_path, output_path):
    """
    Extract Mel-Spec Acoustic Features
    Args:
        input_path$  (str): input directory to the audio (*.wav) folder
        output_path$ (str): target directory to the acoustic feature (*.mat) folder
    """
    ERROR_record = ''
    # Walk the tree.
    for root, directories, files in os.walk(input_path):
        files = sorted(files)
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            if '.wav' in filepath:
                try:
                    features = Mel_Spec(filepath)
                    filename = filename.replace('wav','mat')
                    savemat(os.path.join(output_path, filename), {'Audio_data':features})
                except:
                    ERROR_record += 'Error: '+filename+'\n'
            else:
                ERROR_record += 'Unsupported input file: '+filename+'\n'
    record_file = open("ErrorRecord.txt","w")
    record_file.write(ERROR_record)
    record_file.close()
###############################################################################


if __name__=='__main__':
    
    # Setting I/O Paths
    input_path = '/YOUR/ROOT/PATH/TO/MSP-PODCAST-Publish-1.8/Audio/'
    output_path = '/YOUR/ROOT/PATH/TO/MSP-PODCAST-Publish-1.8/Features/Mel_Spec128/feat_mat/'

    # Creating output folder
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
   
    # Extract features
    Extract_AcousticFeat(input_path, output_path)
