import numpy as np
import matplotlib.pyplot as plt
import pylab
from scipy.io import wavfile
from scipy.fftpack import fft
from scipy.signal import stft,istft
from scipy import signal
import IPython
import pickle
import datetime
from math import ceil
import configparser as cp
import random
import glob
import os

from sklearn.decomposition import DictionaryLearning, SparseCoder,dict_learning,dict_learning_online,sparse_encode,TruncatedSVD
from sklearn.linear_model import OrthogonalMatchingPursuitCV,OrthogonalMatchingPursuit,orthogonal_mp
from sklearn.preprocessing import normalize,StandardScaler
from scipy.optimize import nnls

np.set_printoptions(precision=4,suppress=True)

samplingFreq=25000

stft_directory_clean='C:\\SujoyRc\\Personal\\JUEE\\First_work\\datasets\\stft\\speech_in_stationary_noise\\clean'
#stft_directory='C:\\SujoyRc\\Personal\\JUEE\\First_work\\backup_20180113\\juee.tar\\juee\\datasets\\ssn\\m6dB'
#os.chdir(stft_directory)
list_of_stft_files=glob.glob(stft_directory_clean+'/*')

signal_stft_list=[]
for each_file in list_of_stft_files:
    with open(each_file,'rb') as f:
        signal_stft_info=pickle.load(f)
        print (signal_stft_info[0].shape,signal_stft_info[1].shape)
    signal_stft_list.append(signal_stft_info[2])

signal_stft_list_mag=[np.real(x) for x in signal_stft_list]  ### This is actually the real part only. Need to fix in code later
signal_stft_im=[np.imag(x) for x in signal_stft_list]

max_time=max([x.shape[1] for x in signal_stft_list])
freq_bins=max([x.shape[0] for x in signal_stft_list])

padded_mag=[np.pad(signal_stft_list_mag[i]\
                      ,((0,0),(0,max_time-signal_stft_list_mag[i].shape[1])),mode='constant',constant_values=0)\
               for i in range(len(signal_stft_list_mag))]

padded_im=[np.pad(signal_stft_im[i]\
                      ,((0,0),(0,max_time-signal_stft_im[i].shape[1])),mode='constant',constant_values=0)\
               for i in range(len(signal_stft_im))]

def make_long(x):
    ret_array=np.concatenate([x[i,:] for i in range(x.shape[0])])
    return ret_array

input_matrix=np.array([make_long(padded_mag[i]) for i in range(len(padded_mag))]).T

class SpeechSparseCoding:
    '''
    This model is for custom built dictionary learning
    '''
    def __init__(self,ratios=[1.25,1.5,1.75,2],num_iter=20,omp_tol=1e-3,max_tol=1e-3,ncoef=None,maxit=20,tol=1e-3,ztol=1e-3,nonneg=True,train_ratio=0.7):
        self.num_iter=num_iter
        self.omp_tol=omp_tol
        self.tol=omp_tol
        self.max_tol=max_tol
        self.ncoef=ncoef
        self.maxit=maxit
        self.ztol=ztol
        self.nonneg=nonneg
        self.sparse_code_results={}
        self.min_set={}
        self.train_ratio=train_ratio
        self.sparse_code_test_results={}
        self.min_set_test={}
        self.ratios=ratios
        self.train_indices=[]
        self.test_indicies=[]
        
    def omp(self,x,D,alpha):
        '''
        x=D*alpha
        '''
        def norm2(x):
            return np.linalg.norm(x) / np.sqrt(len(x))
        if self.ncoef is None:
            self.ncoef = int(D.shape[1]/2)
        D_T=D.T
        active=[]
        coef = np.zeros(D.shape[1], dtype=float) # solution vector
        residual = x                             # residual vector
        xpred = np.zeros(x.shape, dtype=float)
        xnorm = norm2(x)                         # store for computing relative err
        err = np.zeros(self.maxit, dtype=float)       # relative err vector
        tol=self.tol*xnorm
        ztol=self.ztol*xnorm
        for each_iter in range(self.maxit):
            rcov=np.dot(D_T,residual)
            if self.nonneg:
                i = np.argmax(rcov)
                rc = rcov[i]
            else:
                i = np.argmax(np.abs(rcov))
                rc = np.abs(rcov[i])
            if rc<ztol:
                print('All residual covariances are below threshold.')
                break
            if i not in active:
                active.append(i)
            if self.nonneg:
                coefi, _ = nnls(D[:, active], x)
            else:
                coefi, _, _, _ = np.linalg.lstsq(D[:, active], x)
            coef[active] = coefi   # update solution
            # update residual vector and error
            residual = x - np.dot(D[:,active], coefi)
            xpred = x - residual
            err[each_iter] = norm2(residual) / xnorm
            #       
            # check stopping criteria
            if err[each_iter] < self.tol:  # converged
                print('\nConverged.')
                break
            if len(active) >= self.ncoef:   # hit max coefficients
                print('\nFound solution with max number of coefficients.')
                break
            if each_iter == self.maxit-1:  # max iterations
                print('\nHit max iterations.')
                break
        return coef
    
    def create_sparse_coding(self,mag_spectrum):
        X=mag_spectrum
        N=X.shape[1]
        D=X.shape[0]
        ratios=self.ratios#np.arange(1.1,2.1,0.2)
        sparse_code_results={}
        for each_ratio in ratios:
            L=ceil(N*each_ratio)
            print (datetime.datetime.now())
            print ("Running for L="+str(L))
            print ("#############")
            Dict_notNorm=np.random.random((D,L))
            standardScaler_dict=StandardScaler()
            Dict=standardScaler_dict.fit_transform(Dict_notNorm)
            # Coding Matrix Update Process
            C=np.random.random((L,N))
            final_error=0
            sparse_code_results[L]=[]
            break_flag=0
            for each_iter in range(self.num_iter):
                temp_results={}
                Dict_last=Dict
                C_last=C
                last_err=final_error
                print ("Running for iter="+str(each_iter))
                print (datetime.datetime.now())
                for i in range(N):
                    #c_n=orthogonal_mp(Dict,X[:,i])
                    c_n=self.omp(X[:,i],Dict,C[:,i])
                    C[:,i]=c_n
                for l in range(L):
                    other_col_sum=0
                    for j in range(L):
                        if j==l:
                            pass
                        else:
                            p=Dict[:,j].reshape((D,1))
                            q=C[j,:].reshape(1,N)
                            other_col_sum=other_col_sum+p.dot(q)
                    R_l=X-other_col_sum
                    c_l=C[l,:].reshape(1,N)
                    non_zero_columns=np.sum(c_l!=0)
                    if non_zero_columns>1:
                        #print (each_iter,l)
                        indices=np.nonzero(c_l)[1]
                        svd_model=TruncatedSVD(n_components=1)
                        U=svd_model.fit_transform(R_l[:,indices])
                        Sigma=svd_model.explained_variance_ratio_
                        VT=svd_model.components_
                        Dict[:,l]=U.reshape(Dict[:,l].shape)
                        c_l_new=Sigma.dot(VT)
                        k=0
                        for each_index in indices:
                            try:
                                C[l,each_index]=c_l_new[k]
                                k=k+1
                            except:
                                break
                final_error=np.linalg.norm((X-Dict.dot(C)),ord="fro")
                print (np.sum(C!=0,axis=0))
                print ("After "+str(each_iter+1)+" iterations error is "+str(final_error))
                if (each_iter==self.num_iter-1):
                    print (final_error,last_err)
                    Dict=Dict_last
                    C=C_last
                    final_error=last_err
                    break_flag=1
                delta=abs(last_err-final_error)
                temp_results['C']=C
                temp_results['Dict']=Dict
                temp_results['final_error']=final_error
                temp_results['C']=C
                sparse_code_results[L].append(temp_results)
                if break_flag==1:
                    break
                else:
                    if delta < self.max_tol:
                        break
            print ("Done for L= "+str(L))
            #self.sparse_code_results=sparse_code_results
        return sparse_code_results
    
    def get_best_results(self,results):
        min_set={}
        min_value=1000000
        for each_key in results.keys():
            for i in range(len(results[each_key])):
                if results[each_key][i]['final_error']<min_value:
                    min_set=results[each_key][i]
        return min_set
    
    def get_best_test_results(self,results):
        min_set={}
        min_value=1000000
        for each_key in results.keys():
            if results[each_key]['final_test_error']<min_value:
                min_set=results[each_key]
        return min_set
                    
    def train_test_model(self,input_matrix):
        number_of_signals=input_matrix.shape[1]
        train_signal_count=ceil(number_of_signals*self.train_ratio)
        self.train_indices=random.sample(range(number_of_signals),train_signal_count)
        self.test_indices=[x for x in list(range(number_of_signals)) if x not in self.train_indices]
        train_set=input_matrix[:,self.train_indices]
        test_set=input_matrix[:,self.test_indices]        
        self.sparse_code_results=self.create_sparse_coding(train_set)
        self.min_set=self.get_best_results(self.sparse_code_results)
        self.sparse_code_test_results=self.get_test_errors(test_set)
        self.min_set_test=self.get_best_test_results(self.sparse_code_test_results)
        return self.min_set,self.min_set_test
        
        
    def get_test_errors(self,test_matrix):
        print ("Running on test data")
        num_test_signals=test_matrix.shape[1]
        sparse_code_test_results={}
        for each_key in self.sparse_code_results.keys():
            sparse_code_test_results[each_key]={}
            print ("running test for "+str(each_key))
            self.sparse_code_test_results[each_key]={}
            to_process={}
            min_value=1000000
            for i in range(len(self.sparse_code_results[each_key])):
                if self.sparse_code_results[each_key][i]['final_error']<min_value:
                    to_process=self.sparse_code_results[each_key][i]
            Dict=to_process['Dict']
            C=np.zeros((Dict.shape[1],num_test_signals))
            for i in range(num_test_signals):
                c_n=self.omp(test_matrix[:,i],Dict,C[:,i])
                C[:,i]=c_n
            recovered_signal=Dict.dot(C)
            final_test_error=np.linalg.norm((test_matrix-Dict.dot(C)),ord="fro")
            sparse_code_test_results[each_key]['Dict']=Dict
            sparse_code_test_results[each_key]['C']=C
            sparse_code_test_results[each_key]['final_test_error']=final_test_error
        return sparse_code_test_results
    

speechSparseCoding=SpeechSparseCoding()
min_set,min_set_test=speechSparseCoding.train_test_model(input_matrix)

######################

trained_matrix=min_set['Dict'].dot(min_set['C'])

trained_signals_list=[]
for k in range(len(speechSparseCoding.train_indices)):
    train_index=speechSparseCoding.train_indices[k]
    each_signal_long=trained_matrix[:,k]
    each_signal_mag=each_signal_long.reshape((freq_bins,max_time))        
    #each_signal_real=np.nan_to_num(np.sqrt(each_signal_mag**2-padded_im[train_index]**2))
    each_signal_real=each_signal_mag
    each_signal_stft_padded=each_signal_real+1j*padded_im[train_index]
    actual_length=signal_stft_list_mag[train_index].shape[1]
    #print (actual_length)
    each_signal_stft=each_signal_stft_padded[:,0:actual_length]
    trained_signals_list.append(each_signal_stft)
    
trained_time_signal_list=[]
for each_signal in trained_signals_list:
    time_signal=istft(each_signal,samplingFreq,'hann')
    trained_time_signal_list.append(time_signal)

os.chdir("C:\\SujoyRc\\Personal\\JUEE\\First_work\\code")   
for l in range(len(trained_signals_list)):
    train_index=speechSparseCoding.train_indices[l]
    file_base_name=list_of_stft_files[train_index].split('\\')[-1].split('.')[0]
    file_name="../temp/train_"+file_base_name+'.wav'
    wavfile.write(file_name,samplingFreq,np.asarray(trained_time_signal_list[l][1]*(2.**15),dtype=np.int16))    
    #file_name_orig='C://SujoyRc//Personal//JUEE//First_work//datasets//speech_in_stationary_noise//m6dB//'+file_base_name+'.wav'

    
########################   

recovered_matrix=min_set_test['Dict'].dot(min_set_test['C']) 

recovered_signals_list=[]
for k in range(len(speechSparseCoding.test_indices)):
    test_index=speechSparseCoding.test_indices[k]
    each_signal_long=recovered_matrix[:,k]
    each_signal_mag=each_signal_long.reshape((freq_bins,max_time))        
    #each_signal_real=np.nan_to_num(np.sqrt(each_signal_mag-padded_im[test_index]**2))
    each_signal_real=each_signal_mag
    each_signal_stft_padded=each_signal_real+1j*padded_im[test_index]
    actual_length=signal_stft_list_mag[test_index].shape[1]
    #print (actual_length)
    each_signal_stft=each_signal_stft_padded[:,0:actual_length]
    recovered_signals_list.append(each_signal_stft)
    
recovered_time_signal_list=[]
for each_signal in recovered_signals_list:
    time_signal=istft(each_signal,samplingFreq,'hann')
    recovered_time_signal_list.append(time_signal)
   
os.chdir("C:\\SujoyRc\\Personal\\JUEE\\First_work\\code")    
for l in range(len(recovered_time_signal_list)):
    test_index=speechSparseCoding.test_indices[l]
    file_base_name=list_of_stft_files[test_index].split('\\')[-1].split('.')[0]
    file_name="../temp/test_"+file_base_name+'.wav'
    wavfile.write(file_name,samplingFreq,np.asarray(recovered_time_signal_list[l][1]*(2.**15),dtype=np.int16))    
    
    
#############################

stft_directory_noisy='C:\\SujoyRc\\Personal\\JUEE\\First_work\\datasets\\stft\\speech_in_stationary_noise\\m6dB'
#stft_directory='C:\\SujoyRc\\Personal\\JUEE\\First_work\\backup_20180113\\juee.tar\\juee\\datasets\\ssn\\m6dB'
#os.chdir(stft_directory)
list_of_stft_files_noisy=glob.glob(stft_directory_noisy+'/*')

noisy_signal_stft_list=[]
for each_file in list_of_stft_files_noisy:
    with open(each_file,'rb') as f:
        noisy_signal_stft_info=pickle.load(f)
        print (noisy_signal_stft_info[0].shape,noisy_signal_stft_info[1].shape)
    noisy_signal_stft_list.append(noisy_signal_stft_info[2])

noisy_signal_stft_list_mag=[np.real(x) for x in noisy_signal_stft_list]  ### This is actually the real part only. Need to fix in code later
noisy_signal_stft_im=[np.imag(x) for x in noisy_signal_stft_list]

#max_time=max([x.shape[1] for x in noisy_signal_stft_list])
#freq_bins=max([x.shape[0] for x in noisy_signal_stft_list])

noisy_padded_mag=[np.pad(noisy_signal_stft_list_mag[i]\
                      ,((0,0),(0,max_time-noisy_signal_stft_list_mag[i].shape[1])),mode='constant',constant_values=0)\
               for i in range(len(noisy_signal_stft_list_mag))]

noisy_padded_im=[np.pad(noisy_signal_stft_im[i]\
                      ,((0,0),(0,max_time-noisy_signal_stft_im[i].shape[1])),mode='constant',constant_values=0)\
               for i in range(len(noisy_signal_stft_im))]

noisy_input_matrix=np.array([make_long(noisy_padded_mag[i]) for i in range(len(noisy_padded_mag))]).T

#################################    
    
    
def orth_match_pursuit(x,D,alpha,ratios=[1.25,1.5,1.75,2],num_iter=20,omp_tol=1e-5,max_tol=1e-5,ncoef=None,maxit=20,tol=1e-5,ztol=1e-5,nonneg=True,train_ratio=0.7):
        '''
        x=D*alpha
        '''
        def norm2(x):
            return np.linalg.norm(x) / np.sqrt(len(x))
        if ncoef is None:
            ncoef = int(D.shape[1]/2)
        D_T=D.T
        active=[]
        coef = np.zeros(D.shape[1], dtype=float) # solution vector
        residual = x                             # residual vector
        xpred = np.zeros(x.shape, dtype=float)
        xnorm = norm2(x)                         # store for computing relative err
        err = np.zeros(maxit, dtype=float)       # relative err vector
        tol=tol*xnorm
        ztol=ztol*xnorm
        for each_iter in range(maxit):
            rcov=np.dot(D_T,residual)
            if nonneg:
                i = np.argmax(rcov)
                rc = rcov[i]
            else:
                i = np.argmax(np.abs(rcov))
                rc = np.abs(rcov[i])
            if rc<ztol:
                print('All residual covariances are below threshold.')
                break
            if i not in active:
                active.append(i)
            if nonneg:
                coefi, _ = nnls(D[:, active], x)
            else:
                coefi, _, _, _ = np.linalg.lstsq(D[:, active], x)
            coef[active] = coefi   # update solution
            # update residual vector and error
            residual = x - np.dot(D[:,active], coefi)
            xpred = x - residual
            err[each_iter] = norm2(residual) / xnorm
            #       
            # check stopping criteria
            if err[each_iter] < tol:  # converged
                print('\nConverged.')
                break
            if len(active) >= ncoef:   # hit max coefficients
                print('\nFound solution with max number of coefficients.')
                break
            if each_iter == maxit-1:  # max iterations
                print('\nHit max iterations.')
                break
        return coef

Dict=min_set['Dict']

num_noisy_signals=noisy_input_matrix.shape[1]
C=np.zeros((Dict.shape[1],num_noisy_signals))
for i in range(num_noisy_signals):
    c_n=orth_match_pursuit(noisy_input_matrix[:,i],Dict,C[:,i])
    C[:,i]=c_n
denoised_matrix=Dict.dot(C)

denoised_signals_list=[]
for k in range(num_noisy_signals):
    #test_index=speechSparseCoding.test_indices[k]
    each_signal_long=denoised_matrix[:,k]
    each_signal_mag=each_signal_long.reshape((freq_bins,max_time))        
    #each_signal_real=np.nan_to_num(np.sqrt(each_signal_mag-padded_im[test_index]**2))
    each_signal_real=each_signal_mag
    each_signal_stft_padded=each_signal_real+1j*noisy_padded_im[k]
    actual_length=noisy_signal_stft_list_mag[k].shape[1]
    #print (actual_length)
    each_signal_stft=each_signal_stft_padded[:,0:actual_length]
    denoised_signals_list.append(each_signal_stft)
    
    
denoised_time_signal_list=[]
for each_signal in denoised_signals_list:
    time_signal=istft(each_signal,samplingFreq,'hann')
    denoised_time_signal_list.append(time_signal)
    
os.chdir("C:\\SujoyRc\\Personal\\JUEE\\First_work\\code")    
for l in range(len(denoised_time_signal_list)):
    file_base_name=list_of_stft_files_noisy[l].split('\\')[-1].split('.')[0]
    file_name="../temp/denoised_"+file_base_name+'.wav'
    wavfile.write(file_name,samplingFreq,np.asarray(denoised_time_signal_list[l][1]*(2.**15),dtype=np.int16)) 

#############

directory_wav_clean_files='C:/SujoyRc/Personal/JUEE/First_work/datasets/speech_in_stationary_noise/clean'
list_of_clean_wav_files=glob.glob(directory_wav_clean_files+'/*')
clean_audio_list=[]
for i in range(len(list_of_clean_wav_files)):
    fileName=list_of_clean_wav_files[i]
    myAudio = fileName
    samplingFreq, mySound = wavfile.read(myAudio)
    mySound = mySound / (2.**15)
    clean_audio_list.append(mySound)
    
directory_wav_noisy_files='C:/SujoyRc/Personal/JUEE/First_work/datasets/speech_in_stationary_noise/m6dB'
list_of_noisy_wav_files=glob.glob(directory_wav_noisy_files+'/*')
noisy_audio_list=[]
for i in range(len(list_of_noisy_wav_files)):
    fileName=list_of_noisy_wav_files[i]
    myAudio = fileName
    samplingFreq, mySound = wavfile.read(myAudio)
    mySound = mySound / (2.**15)
    noisy_audio_list.append(mySound)
    
def get_mse(clean_audio_list,estimated_audio_list,list_of_clean_wav_files=list_of_clean_wav_files,list_of_stft_files=list_of_stft_files):
    error=0
    count=0
    for k in range(len(list_of_clean_wav_files)):
        file_name=list_of_clean_wav_files[k]
        file_base_name=file_name.split('\\')[-1].split('.')[0]
        stft_files_bases=[x.split('\\')[-1].split('.')[0] for x in list_of_stft_files]
        stft_file_index=[p[0] for p in enumerate(stft_files_bases) if p[1]==file_base_name][0]
        clean_file=clean_audio_list[k]
        denoised_file=estimated_audio_list[stft_file_index]
        if clean_file.shape[0]>denoised_file.shape[0]:
            clean_file=clean_file[0:denoised_file.shape[0]]
        else:
            denoised_file=denoised_file[0:clean_file.shape[0]]
        this_err=np.sum((clean_file-denoised_file)**2)
        error=error+this_err
        count=count+1
    mse=error/count
    return mse
### Error in original noisy signal
get_mse(clean_audio_list,noisy_audio_list)
# 455.43
### Error in denoised signal
get_mse(clean_audio_list,[p[1] for p in denoised_time_signal_list])
#  165.67
def get_snr(clean_audio_list,estimated_audio_list,list_of_clean_wav_files=list_of_clean_wav_files,list_of_stft_files=list_of_stft_files):
    sum_db=0
    count=0
    for k in range(len(list_of_clean_wav_files)):
        file_name=list_of_clean_wav_files[k]
        file_base_name=file_name.split('\\')[-1].split('.')[0]
        stft_files_bases=[x.split('\\')[-1].split('.')[0] for x in list_of_stft_files]
        stft_file_index=[p[0] for p in enumerate(stft_files_bases) if p[1]==file_base_name][0]
        clean_file=clean_audio_list[k]
        denoised_file=estimated_audio_list[stft_file_index]
        if clean_file.shape[0]>denoised_file.shape[0]:
            clean_file=clean_file[0:denoised_file.shape[0]]
        else:
            denoised_file=denoised_file[0:clean_file.shape[0]]
        this_err=np.sum((clean_file-denoised_file)**2)
        num_samples=clean_file.shape[0]
        power_signal=sum([p**2 for p in clean_file])/num_samples
        power_noise=sum([p**2 for p in denoised_file])/num_samples
        db=10*np.log10(power_signal/power_noise)
        sum_db=sum_db+db
        count=count+1
    snr_avg=sum_db/count
    return snr_avg    
#
get_snr(clean_audio_list,noisy_audio_list)
# -6.98 dB
get_snr(clean_audio_list,[p[1] for p in denoised_time_signal_list])    
# -3.2 dB        
        
        
    
