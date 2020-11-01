import h5py
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np


f = h5py.File("../data/raw/X_test.h5/X_test.h5","r")

y_train = pd.read_csv('../data/raw/y_train.csv')
sample_sub = pd.read_csv('../data/raw/sample_submission.csv')


def windows_split(X_data):
    """split the data in windows of length n for manipulation such as MMD (ref entropy.pdf figure 7"""
    n = len(X_data)
    return np.array([X_data[i*100:100*(i+1)] for i in range(n//100)])

def MMD(data_array):
    """return MMD of a signal"""
    n_pts = len(data_array)
    min_val = data_array[0]
    max_val = data_array[0]
    min_index = 0
    max_index = 0
    for i in range(n_pts):
        if data_array[i]<min_val:
            min_val = data_array[min_index]
            min_index = i
        if data_array[i]>max_val:
            max_val = data_array[max_index]
            min_index = i
        
    d = np.sqrt((max_index-min_index)**2 + (max_val-min_val)**2)
    return d


def MMD_epoch(data_epoch):
    """return sum of the MMD of each window of an epoch"""
    return sum([MMD(split) for split in windows_split(data_epoch)])

def get_mid_freq(freq_seq):
    """get the frequence where the amplitude of the integral at left and right are equal"""
    n_pts = len(freq_seq)
    integral = np.sum(np.abs(freq_seq[:n_pts//2]))
    cum_sum = 0 
    for i in range(len(freq_seq)):
        cum_sum = cum_sum + np.abs(freq_seq[i])
        if cum_sum>=integral/2:
            print(cum_sum,integral)
            return i

def get_amplitude_range_freq(freq_seq,fmin,fmax):
    """get the sum of amplitude of frequencies in a given range"""
    return np.sum(np.abs(freq_seq[fmin:fmax]))
    



def esis(data_array):
    """retur sum of amplitude of the signal"""
    return np.sum(np.abs(data_array))

def esis_epoch(data_epoch):
    """return sum of amplitude of the signal for an epoch"""
    return sum([esis(split) for split in windows_split(data_epoch)])


# delta = get_amplitude_range_freq(fft,0,4)
# theta = get_amplitude_range_freq(fft,4,8)
# alpha = get_amplitude_range_freq(fft,8,13)
# beta = get_amplitude_range_freq(fft,13,22)
# gamma = get_amplitude_range_freq(fft,30,len(fft)//2)


# features_array=[]
# for i in range(1,8) : features_array.append(MMD_epoch(eeg_epoch) for eeg_epoch in f['eeg_' +str(i)][:n])
# features_array.append([MMD_epoch(pos_epoch) for pos_epoch in f['x'][:n]])
# features_array.append([MMD_epoch(pos_epoch) for pos_epoch in f['y'][:n]])
# features_array.append([MMD_epoch(pos_epoch) for pos_epoch in f['z'][:n]])
# features_array.append([MMD_epoch(pos_epoch) for pos_epoch in f['pulse'][:n]])

# for i in range(1,8) : features_array.append([esis_epoch(eeg_epoch) for eeg_epoch in f['eeg_'+str(i)][:n]]) 
# features_array.append([esis_epoch(pos_epoch) for pos_epoch in f['x'][:n]])
# features_array.append([esis_epoch(pos_epoch) for pos_epoch in f['y'][:n]])
# features_array.append([esis_epoch(pos_epoch) for pos_epoch in f['z'][:n]])
# features_array.append([esis_epoch(pos_epoch) for pos_epoch in f['pulse'][:n]])

# for i in range(1,8): features_array.append([get_mid_freq(eeg_epoch) for eeg_epoch in f['eeg_'+str(i)][:n]]) 
