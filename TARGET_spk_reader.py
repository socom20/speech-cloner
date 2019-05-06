import numpy as np
import matplotlib.pyplot as plt
import os, sys
import librosa
import pickle
import hashlib

import h5py
import math

from audio_lib import calc_MFCC_input
from sound_ds import Sound_DS

class TARGET_spk(Sound_DS):

    def __init__(self, cfg_d):
        self.cfg_d       = cfg_d

        if 'hop_length' not in self.cfg_d.keys():
            self.cfg_d['hop_length'] = int(self.cfg_d['hop_length_ms'] * self.cfg_d['sample_rate'] / 1000.0)
            print(" - cfg_d['hop_length'] = {:d}".format(self.cfg_d['hop_length']))

        if 'win_length' not in self.cfg_d.keys():
            self.cfg_d['win_length'] = int(self.cfg_d['win_length_ms'] * self.cfg_d['sample_rate'] / 1000.0)
            print(" - cfg_d['win_length'] = {:d}".format(self.cfg_d['win_length']))

            
        self.sample_rate = cfg_d['sample_rate']
        self.verbose     = cfg_d['verbose']
    
        self.ds_path          = cfg_d['ds_path']
        self.ds_norm          = cfg_d['ds_norm']
        self.n_mfcc           = cfg_d['n_mfcc']         # Cantidad de mfcc en la salida 
        self.n_timesteps      = cfg_d['n_timesteps']    # Cantidad de de pasos temporales para el muestreo  window_sampler

        
        if cfg_d['random_seed'] is not None:
            np.random.seed(self.random_seed)
            

        if not os.path.exists(os.path.join(self.cfg_d['ds_path'], self.cfg_d['ds_cache_name'])) or cfg_d['remake_samples_cache']:
            self._read_mp3()
            self.save_dataset_cache()
        else:
            self.load_dataset_cache()

        if self.ds_norm != (0, 1):
            self._normalize_ds()

        spec_name_id = hashlib.md5('_'.join([str(cfg_d[k]) for k in ('sample_rate',
                                                                     'pre_emphasis',
                                                                     'hop_length',
                                                                     'win_length',
                                                                     'n_mels',
                                                                     'n_mfcc',
                                                                     'n_fft',
                                                                     'window',
                                                                     'mfcc_normaleze_first_mfcc',
                                                                     'mfcc_norm_factor',
                                                                     'calc_mfcc_derivate',
                                                                     'M_dB_norm_factor',
                                                                     'P_dB_norm_factor',
                                                                     'mean_abs_amp_norm',
                                                                     'clip_output')]).encode()).hexdigest()

        self.spec_cache_name = '.'.join(cfg_d['spec_cache_name'].split('.')[:-1]) + '_' + spec_name_id + '.' + cfg_d['spec_cache_name'].split('.')[-1]

        if not os.path.exists(os.path.join(self.ds_path, self.spec_cache_name)):
            r = ''
            while not r in ['y', 'n']:
                print(' - TIMIT, no se encontr´o el archivo de cache "{}", desea construirlo (y/n):'.format(self.spec_cache_name), end='')
                r = input()
            if r == 'y':
                self.create_spec_cache()
            else:
                print(' - TIMIT, no se puede continuar sin generar el archivo de cache.', file=sys.stderr)
                return None

        return None


    

    
    def _read_mp3(self):
        self.ds = {'wav':  [],
                   'name': [],
                   'len':  []}
        
        for file_name in sorted(os.listdir(self.cfg_d['ds_path'])):
            if len(file_name) > 4 and file_name[-4:] == '.mp3':
                exclude = False
                for excl in self.cfg_d['exclude_files_with']:
                    if excl in file_name:
                        exclude = True
                        break
                    
                if exclude:
                    print(' Excluded: "{}"'.format(file_name))
                    continue

                    
                file_path = os.path.join(self.cfg_d['ds_path'], file_name)

                if self.verbose:
                    print(' Reading: "{}" ... '.format(file_name), end='')
                    
                y, sr = librosa.load(file_path, self.cfg_d['sample_rate'])
                lenght = y.shape[0] / sr
                
                self.ds['wav'].append(y)
                self.ds['name'].append(file_name)
                self.ds['len'].append(y.shape[0] / sr)
                
                if self.verbose:
                    print(' Ok!!! length = {:0.02f} s'.format( self.ds['len'][-1] ) )


        self.ds['wav']  = np.array( self.ds['wav'] )
        self.ds['name'] = np.array( self.ds['name'] )
        self.ds['len']  = np.array( self.ds['len'] )

        if self.verbose:
            m, s = divmod(int(self.ds['len'].sum()), 60)
            h, m = divmod(m, 60)
            print(' Total wavs length = {:02d}:{:02d}:{:02d} s'.format(h, m, s))

        
        return None


    def create_spec_cache(self, cfg_d=None):

        if cfg_d is None:
            cfg_d = self.cfg_d

        if os.path.exists(os.path.join(self.ds_path, self.spec_cache_name)):
            print(' WARNING, create_spec_cache: el archivo "{}" ya existe, para generarlo de nuevo primero se debe eliminar.', file=sys.stderr)
            return None
        
        n_samples  = len(self.ds['wav'])
        print(' - create_spec_cache, Salvando {} cache'.format(self.spec_cache_name))
        
        with h5py.File(os.path.join(self.ds_path, self.spec_cache_name), 'w') as ds_h5py:

            grp_mfcc     = ds_h5py.create_group("mfcc")
            grp_mel_dB   = ds_h5py.create_group("mel_dB")
            grp_power_dB = ds_h5py.create_group("power_dB")
            
            for i_sample in range(n_samples):
                if self.verbose:
                    print(' - Saved:{} of {} samples'.format(i_sample, n_samples))
                    
                y     = self.ds['wav'][i_sample]
                
                mfcc, mel_dB, power_dB = calc_MFCC_input(y,
                                                         sr=cfg_d['sample_rate'],
                                                         pre_emphasis=cfg_d['pre_emphasis'],
                                                         hop_length=cfg_d['hop_length'],
                                                         win_length=cfg_d['win_length'],
                                                         n_mels=cfg_d['n_mels'],
                                                         n_mfcc=cfg_d['n_mfcc'],
                                                         n_fft=cfg_d['n_fft'],
                                                         window=cfg_d['window'],
                                                         mfcc_normaleze_first_mfcc=cfg_d['mfcc_normaleze_first_mfcc'],
                                                         mfcc_norm_factor=cfg_d['mfcc_norm_factor'],
                                                         calc_mfcc_derivate=cfg_d['calc_mfcc_derivate'],
                                                         M_dB_norm_factor=cfg_d['M_dB_norm_factor'],
                                                         P_dB_norm_factor=cfg_d['P_dB_norm_factor'],
                                                         mean_abs_amp_norm=cfg_d['mean_abs_amp_norm'],
                                                         clip_output=cfg_d['clip_output'])


                 
                grp_mfcc.create_dataset(    str(i_sample), data=mfcc)
                grp_mel_dB.create_dataset(  str(i_sample), data=mel_dB)
                grp_power_dB.create_dataset(str(i_sample), data=power_dB)

        if self.verbose:
            print('Archivo "{}" escrito en disco.'.format(self.spec_cache_name))
            
        return None




    def spec_window_sampler(self, batch_size=32, n_epochs=1, randomize_samples=True, sample_trn=True, prop_val=0.3, yield_idxs=False):
        n_timesteps = self.n_timesteps

        n_samples = self.ds['wav'].shape[0]
        if sample_trn:
            samples_v = np.arange(0, int((1-prop_val)*n_samples))
        else:
            samples_v = np.arange(int((1-prop_val)*n_samples), n_samples)
            
        samples_v = [str(i) for i in samples_v]
        

        with h5py.File(os.path.join(self.ds_path, self.spec_cache_name),'r') as ds_h5py:
            mfcc_v     = []
            mel_dB_v   = []
            power_dB_v = []
            
            idxs_v = []
            for i_epoch in range(n_epochs):
                if randomize_samples:
                    np.random.shuffle(samples_v)
                
                for i_sample in samples_v:
                    spec_len = ds_h5py['mfcc'][i_sample].shape[0]
                    if spec_len <= n_timesteps:
                        print('WARNING: sample {} has spec_len <= n_timesteps'.format(i_sample))
                        continue

                    for i in range(batch_size):
                        # Solamente elegimos un frame por wav
                        i_s = np.random.randint(0, spec_len-n_timesteps)
                        i_e = i_s + n_timesteps
                        
                        mfcc     = ds_h5py["mfcc"][i_sample][i_s:i_e]
                        mel_dB   = ds_h5py["mel_dB"][i_sample][i_s:i_e]
                        power_dB = ds_h5py["power_dB"][i_sample][i_s:i_e]

                        mfcc_v.append( mfcc )
                        mel_dB_v.append( mel_dB )
                        power_dB_v.append( power_dB )
                        
                        idxs_v.append([i_s, i_e, int(i_sample)])

                    if len(mfcc_v) == batch_size:
                        mfcc_v     = np.array(mfcc_v)
                        mel_dB_v   = np.array(mel_dB_v)
                        power_dB_v = np.array(power_dB_v)
                        
                        assert mfcc_v.shape[1] == mel_dB_v.shape[1] == power_dB_v.shape[1] == n_timesteps

                        if yield_idxs:
                            idxs_v = np.array(idxs_v)
                            yield mfcc_v, mel_dB_v, power_dB_v, idxs_v
                        else:
                            yield mfcc_v, mel_dB_v, power_dB_v
                            
                        mfcc_v     = []
                        mel_dB_v   = []
                        power_dB_v = []
                        idxs_v     = []

                        
if __name__ == '__main__':
    if os.name == 'nt':
        ds_path = r'G:\Downloads\TRG\L. Frank Baum/The Wonderful Wizard of Oz'
    else:
        ds_path = '/media/sergio/EVO970/UNIR/TFM/code/data_sets/TRG/L. Frank Baum/The Wonderful Wizard of Oz'

        
    ds_cfg_d = {'ds_path':ds_path,
                'sample_rate':16000,  #Frecuencia de muestreo los archivos de audio Hz
                'exclude_files_with':['Oz-01', 'Oz-25'],
                'ds_cache_name':'AH_target_cache.pickle',
                'verbose':True,
                'spec_cache_name':'spec_cache.h5py',

                'ds_norm':(0.0, 1.0),
                'remake_samples_cache':False,
                'random_seed':None,
                
                'pre_emphasis': 0.97,
                
                'hop_length_ms':   5.0, # 2.5ms = 40c | 5.0ms = 80c (@ 16kHz)
                'win_length_ms':  25.0, # 25.0ms = 400c (@ 16kHz)
                'n_timesteps':   400, # 800ts*(hop_length_ms=2.5ms)= 2000ms  Cantidad de hop_length_ms en una ventana de prediccion.
                
                'n_mels':80,
                'n_mfcc':40,
                'n_fft':None, # None usa n_fft=win_length
                
                'window':'hann',
                'mfcc_normaleze_first_mfcc':True,
                'mfcc_norm_factor': 0.01,
                'calc_mfcc_derivate':False,
                'M_dB_norm_factor':0.01,
                'P_dB_norm_factor':0.01,
                
                'mean_abs_amp_norm':0.003,
                'clip_output':True}


    target_spk = TARGET_spk(ds_cfg_d)


##    MFCC, M, P = target_spk.get_spec(0)
##    for x in [MFCC, M, P]:
##        target_spk.spec_show(x[0:400], idxs_v=(0,400,0))


    for MFCC, M, P in target_spk.spec_window_sampler():
        for x in [MFCC, M, P]:
            target_spk.spec_show(x[0])
        
    



