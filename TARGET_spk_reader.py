import numpy as np
import matplotlib.pyplot as plt
import os, sys
import librosa
import sounddevice as sd
import pickle
import hashlib

import h5py
import math

from audio_lib import calc_MFCC_input


class TARGET_spk:

    def __init__(self, cfg_d, verbose=True):
        self.cfg_d = cfg_d
        self.verbose = verbose
        

    def read_mp3(self):
        self.ds = {'wav': [],
                   'name': []}
        
        for file_name in sorted(os.listdir(self.cfg_d['ds_path'])):
            if len(file_name) > 4 and file_name[-4:] == '.mp3':
                exclude = False
                for excl in self.cfg_d['exclude_files_with']:
                    if excl in file_name:
                        exclude = True
                        break
                    
                if exclude:
                    print(' Excluded:', file_name)
                    continue

                    
                file_path = os.path.join(self.cfg_d['ds_path'], file_name)

                if self.verbose:
                    print(' Reading:', file_name)
                    
                y, sr = librosa.load(file_path, self.cfg_d['sample_rate'])
                self.ds['wav'].append(y)
                self.ds['name'].append(file_name)


        self.ds['wav']  = np.array( self.ds['wav'] )
        self.ds['name'] = np.array( self.ds['name'] )
        return None


    def save_dataset_cache(self):
        if self.verbose:
            print(' - save_dataset_cache: Salvando Archivo de cache: "{}"'.format(self.ds_cache_name))

        with open(os.path.join(self.ds_path, self.ds_cache_name), 'wb') as f:
            pickle.dump(self.ds, f)

        if self.verbose:
            print(' - save_dataset_cache: OK !')
            
        return None
            

    def load_dataset_cache(self):
        if self.verbose:
            print(' - load_dataset_cache: Leyendo Archivo de cache: "{}"'.format(self.ds_cache_name))

        with open(os.path.join(self.ds_path, self.ds_cache_name), 'rb') as f:
            self.ds = pickle.load(f)


        if self.verbose:
            print(' - load_dataset_cache: OK !')

        return None

        
    def stop(self):
        sd.stop()
        return None

    
    def play(self, wave, blocking=False):
        sd.play(np.concatenate([np.zeros(1000),wave]), self.sample_rate, blocking=blocking,loop=False)
        return None



if __name__ == '__main__':
    if os.name == 'nt':
        ds_path = r'G:\Downloads\timit'
    else:
        ds_path = '/media/sergio/EVO970/UNIR/TFM/code/data_sets/TRG/L. Frank Baum/The Wonderful Wizard of Oz'

        
    ds_cfg_d = {'ds_path':ds_path,
                'sample_rate':16000,  #Frecuencia de muestreo los archivos de audio Hz
                'exclude_files_with':['Oz-01', 'Oz-25'],
                'ds_cache_name':'AH_target_cache.pickle',


                'ds_norm':(0.0, 10.0),
                'remake_samples_cache':False,
                'random_seed':None,
                'ds_cache_name':'timit_cache.pickle',
                'phn_mfcc_cache_name':'phn_mfcc_cache.h5py',
                'verbose':True,

                

                'pre_emphasis': 0.97,
                
                'hop_length_ms':   5.0, # 2.5ms = 40c | 5.0ms = 80c (@ 16kHz)
                'win_length_ms':  25.0, # 25.0ms = 400c (@ 16kHz)
                'n_timesteps':   400, # 800ts*(win_length_ms=2.5ms)= 2000ms  Cantidad de hop_length_ms en una ventana de prediccion.
                
                'n_mels':80,
                'n_mfcc':40,
                'window':'hann',
                'mfcc_normaleze_first_mfcc':True,
                'mfcc_norm_factor': 0.01,
                'calc_mfcc_derivate':False,
                'M_dB_norm_factor':0.01,
                'P_dB_norm_factor':0.01,
                
                'mean_abs_amp_norm':0.003,
                'clip_output':True}


    target_spk = TARGET_spk(ds_cfg_d)

    target_spk.read_mp3()
    
