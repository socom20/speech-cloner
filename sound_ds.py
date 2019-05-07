import numpy as np
import matplotlib.pyplot as plt
import os, sys
import librosa

import pickle
import h5py
from collections import namedtuple


class Sound_DS():
    def __init__():
        return None


    def save_dataset_cache(self):
        if self.verbose:
            print(' - save_dataset_cache: Salvando Archivo de cache: "{}"'.format(self.cfg_d['ds_cache_name']))

        with open(os.path.join(self.cfg_d['ds_path'], self.cfg_d['ds_cache_name']), 'wb') as f:
            pickle.dump(self.ds, f)

        if self.verbose:
            print(' - save_dataset_cache: OK !')
            
        return None
            

    def load_dataset_cache(self):
        if self.verbose:
            print(' - load_dataset_cache: Leyendo Archivo de cache: "{}"'.format(self.cfg_d['ds_cache_name']))

        with open(os.path.join(self.cfg_d['ds_path'], self.cfg_d['ds_cache_name']), 'rb') as f:
            self.ds = pickle.load(f)

        if self.verbose:
            print(' - load_dataset_cache: OK !')

        return None

    def stop(self):
        import sounddevice as sd
        sd.stop()
        return None

    
    def play(self, wave, blocking=False):
        import sounddevice as sd
        sd.play(np.concatenate([np.zeros(1000),wave]), self.sample_rate, blocking=blocking,loop=False)
        return None




    def _normalize_ds(self):
        if self.verbose:
            print(' - normalize_ds: Normalizando ondas con: add={:0.02f}  mult={:0.02f}'.format(*self.ds_norm))
            
        for i in range(len(self.ds['wav'])):
            self.ds['wav'][i] = self.ds_norm[1] * (self.ds['wav'][i] + self.ds_norm[0])

        return None



    def spec_show(self, spec, phn_v=None, idxs_v=None, aspect_ratio=3, cmap=None):
        m = spec

        n_repeat = m.shape[0] // m.shape[1] // int(aspect_ratio)
        if n_repeat > 1:
            m_repeat = np.repeat(m, n_repeat, axis=1).T
        else:
            m_repeat = m.T
        
        f, ax = plt.subplots(1,1, figsize=(aspect_ratio*5, 5))
        n = ax.imshow(m_repeat, cmap=cmap)
        cbar = f.colorbar(n)

        if phn_v is not None:
            last_i = 0
            print_up = True
            for i in range(phn_v.shape[0]-1):
                if (phn_v[i] != phn_v[i+1]).any() or i == phn_v.shape[0]-2:
                    if i != phn_v.shape[0]-2:
                        ax.plot([i+1, i+1], [0, m_repeat.shape[0]-1], 'y-')
                    
                    h = (0.85 if print_up else 0.95)*m_repeat.shape[0] 

                    ax.text(0.5*(i+last_i), h, self.idx2phn[np.argmax(phn_v[i])], horizontalalignment='center', color='r')
                    last_i = i
                    print_up = not print_up

        if idxs_v is not None:
            i_s, i_e, i_sample = idxs_v
            step   = self.cfg_d['hop_length']
            y_wave = self.ds['wav'][i_sample][step*i_s:step*i_e]
            x_wave = np.arange(-0.5, (i_e-i_s)-0.5, 1/step)

            if y_wave.shape[0] < x_wave.shape[0]:
                print(' WARNING, spec_show: padding y_wave !!')
                pad_len = x_wave.shape[0] - y_wave.shape[0]
                y_wave  = np.concatenate( [y_wave, np.zeros(pad_len, dtype=y_wave.dtype)] ) 
                
            
            h = m_repeat.shape[0]
            y_wave_morm = 0.5* h * ((y_wave-y_wave.min())/(y_wave.max()-y_wave.min()) - 0.5) + h/2
            plt.plot(x_wave, y_wave_morm,  'b', alpha=0.5)
        
        plt.show()

        return None



    def get_ds_filter(self, ds_filter_d={'spk_id':['bdl','rms','slt','clb']}):
        f = np.ones(self.ds['wav'].shape[0], dtype=np.bool)

        for c, v in ds_filter_d.items():
            if c not in self.ds.keys():
                raise Exception(' - ERROR, get_ds_fillter: campo "{}" no encontrado en el ds'.format(c))
            
            if v is None:
                continue

            v_v = v if type(v) in (list, tuple) else [v]                
            p_f = np.zeros_like(f) # partial filter
            for v in v_v:
                p_f = p_f + (self.ds[c] == v)  # or

            f = f * p_f  # and

        if f.sum() == 0:
            print('WARNING, no se selecciona ningun dato. Revisar campos de filtrado', file=sys.stderr)
        
        return f


    def get_n_windows(self, prop_val=0.3, ds_filter_d={}):
        f_s = self.get_ds_filter(ds_filter_d)
        
        n_windows     = sum([self.ds['wav'][i].shape[0] // (self.cfg_d['hop_length'] * self.cfg_d['n_timesteps']) for i in range(self.ds['wav'][f_s].shape[0])])
        n_windows_trn = int((1-prop_val)*n_windows)
        n_windows_val = n_windows - n_windows_trn
        
        return n_windows_trn, n_windows_val



    def get_spec(self, i_sample):
        ret_names_v = ["mfcc",
                       "mel_dB",
                       "power_dB",
                       "phn",
                       "input_mfcc",
                       "target_phn"]

        ret_val_v = []
        ret_str_v = []
        with h5py.File(os.path.join(self.ds_path, self.spec_cache_name), 'r') as ds_h5py:
            for name in ret_names_v:
                if name in ds_h5py.keys():
                    ret_val_v.append( ds_h5py[name][str(i_sample)][:] )
                    ret_str_v.append( name )
                    
##            mfcc     = ds_h5py["mfcc"][str(i_sample)][:]
##            mel_dB   = ds_h5py["mel_dB"][str(i_sample)][:]
##            power_dB = ds_h5py["power_dB"][str(i_sample)][:]
##            phn      = ds_h5py["phn"][str(i_sample)][:]

        ret_nt = namedtuple('ret', ' '.join(ret_str_v))
        return ret_nt(*ret_val_v)



    
