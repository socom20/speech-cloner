import numpy as np
import matplotlib.pyplot as plt
import os, sys
import librosa
import sounddevice as sd
import pickle
import hashlib

import h5py
import math

from audio_lib import calc_MFCC_input, calc_PHN_target

class TIMIT:
    def __init__(self, cfg_d={}):

        self.cfg_d = cfg_d

        if 'hop_length' not in self.cfg_d.keys():
            self.cfg_d['hop_length'] = int(self.cfg_d['hop_length_ms'] * self.cfg_d['sample_rate'] / 1000.0)
            print(" - cfg_d['hop_length'] = {:d}".format(self.cfg_d['hop_length']))

        if 'win_length' not in self.cfg_d.keys():
            self.cfg_d['win_length'] = int(self.cfg_d['win_length_ms'] * self.cfg_d['sample_rate'] / 1000.0)
            print(" - cfg_d['win_length'] = {:d}".format(self.cfg_d['win_length']))

        
        self.ds_path          = cfg_d['ds_path']
        self.use_all_phonemes = cfg_d['use_all_phonemes']

        self.random_seed      = cfg_d['random_seed']
        self.verbose          = cfg_d['verbose']

        self.ds_norm          = cfg_d['ds_norm']

        self.n_mfcc           = cfg_d['n_mfcc']         # Cantidad de mfcc en la salida 
        self.n_timesteps      = cfg_d['n_timesteps']    # Cantidad de de pasos temporales para el muestreo  window_sampler

        
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            
        
        self.sample_rate  = cfg_d['sample_rate']

        if self.sample_rate != 16000:
            raise Exception('- ERROR, TIMIT: para esta base de datos el sample_rate debe ser igual a 16000 Hz.')

        self.ds_type_v    = np.array(['TRAIN', 'TEST'])
        self.ds_dialect_v = np.array(['DR'+str(i) for i in range(1,9)])
        self.ds_gender_v  = np.array(['M', 'F'])
        
        self.ds_phoneme_61_v = np.array(['b', 'd', 'g', 'p', 't', 'k', 'dx', 'q',      # Stops
                                         'bcl','dcl','gcl','pcl','tcl','kcl',          # Silences
                                         'jh', 'ch',                                   # Affricates
                                         's', 'sh', 'z', 'zh', 'f', 'th', 'v', 'dh',   # Fricatives
                                         'm', 'n', 'ng', 'em', 'en', 'eng', 'nx',      # Nasals
                                         'l', 'r', 'w', 'y', 'hh', 'hv', 'el',         # Semivowels and Glides
                                         'iy', 'ih', 'eh', 'ey', 'ae', 'aa', 'aw', 'ay', 'ah', 'ao', 'oy', 'ow', 'uh', 'uw', 'ux', 'er', 'ax', 'ix', 'axr', 'ax-h', # Vowels
                                         'pau', 'epi', 'h#'])                          # Others ('1', '2' no aparece en el ds)
                                      

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # Conversion 61 to 39 phoneme simbols:
        # cite: Phoneme Recognition from the TIMIT database using Recurrent Error Propagation Networks
        # The TIMIT symbol set with the CMU/MIT reduction and IPA symbols
        self.phn_61to39_v = [['p',   'p'],[ 't',  't'],[ 'k',   'k'],['pcl','sil'],['tcl','sil'],['kcl','sil'],
                             ['dx', 'dx'],[ 'm',  'm'],[ 'n',   'n'],[ 'ng', 'ng'],[ 'nx',  'n'],[  's',  's'],
                             ['ch', 'ch'],['th', 'th'],[ 'f',   'f'],[  'l',  'l'],[  'r',  'r'],[  'y',  'y'],
                             ['hh', 'hh'],['eh', 'eh'],[ 'ao', 'aa'],[ 'aa', 'aa'],[ 'uw', 'uw'],[ 'er', 'er'],
                             ['ay', 'ay'],['ey', 'ey'],[ 'aw', 'aw'],[ 'ax', 'ah'],[ 'ix', 'ih'],[  'b',  'b'],
                             [ 'd',  'd'],[ 'g',  'g'],['bcl','sil'],['dcl','sil'],['gcl','sil'],[  'z',  'z'],
                             ['em',  'm'],['en',  'n'],['eng', 'ng'],[ 'sh', 'sh'],[ 'zh', 'sh'],[ 'jh', 'jh'],
                             ['dh', 'dh'],[ 'v',  'v'],[ 'el',  'l'],[  'w',  'w'],[ 'h#','sil'],['epi','sil'],
                             ['hv', 'hh'],['ih', 'ih'],[ 'ae', 'ae'],[ 'ah', 'ah'],[ 'uh', 'uh'],[ 'ux', 'uw'],
                             ['oy', 'oy'],['iy', 'iy'],[ 'ow', 'ow'],['axr', 'er'],['ax-h','ah'],['pau','sil'], ['q', '']] # fonema 'q' no tiene representaci´on

        self.phn_61to39_d = dict(self.phn_61to39_v)
        self.ds_phoneme_39_v = np.unique([p[1] for p in self.phn_61to39_v if p[1] != ''])

        self.phn_61to39_conv_matix = np.zeros( (61, 39), dtype=np.int32)
        for phn61, phn39 in self.phn_61to39_v:
            if phn39 != '':
                idx61 = np.argwhere(self.ds_phoneme_61_v==phn61)[0,0]
                idx39 = np.argwhere(self.ds_phoneme_39_v==phn39)[0,0]

                self.phn_61to39_conv_matix[idx61, idx39] = 1
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

        self.ds_cache_name  = cfg_d['ds_cache_name']
        phn_mfcc_name_id = hashlib.md5('_'.join([str(cfg_d[k]) for k in ('use_all_phonemes',
                                                                         'sample_rate',
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


        
        self.phn_mfcc_cache_name = '.'.join(cfg_d['phn_mfcc_cache_name'].split('.')[:-1]) + '_' + phn_mfcc_name_id + '.' + cfg_d['phn_mfcc_cache_name'].split('.')[-1]

        self.ds = None
        if not os.path.exists( os.path.join(self.ds_path, self.ds_cache_name) ) or cfg_d['remake_samples_cache']:
            self.read_dataset_from_disk(self.verbose)        
            self.save_dataset_cache()
        else:
            self.load_dataset_cache()


        self.normalize_ds()

        self.make_phoneme_convertion_dicts()


        if not os.path.exists(os.path.join(self.ds_path, self.phn_mfcc_cache_name)):
            r = ''
            while not r in ['y', 'n']:
                print(' - TIMIT, no se encontr´o el archivo de cache "{}", desea construirlo (y/n):'.format(self.phn_mfcc_cache_name), end='')
                r = input()
            if r == 'y':
                self.create_phn_mfcc_cache()
            else:
                print(' - TIMIT, no se puede continuar sin generar el archivo de cache.', file=sys.stderr)
                return None
                
        return None



    def normalize_ds(self):
        if self.verbose:
            print(' - TIMIT, normalize_ds: Normalizando ondas con: add={:0.02f}  mult={:0.02f}'.format(*self.ds_norm))
            
        for i in range(len(self.ds['wav'])):
            self.ds['wav'][i] = self.ds_norm[1] * (self.ds['wav'][i] + self.ds_norm[0])

        return None

    def create_phn_mfcc_cache(self, cfg_d=None):

        if cfg_d is None:
            cfg_d = self.cfg_d


        if os.path.exists(os.path.join(self.ds_path, self.phn_mfcc_cache_name)):
            print(' WARNING, create_phn_mfcc_cache: el archivo "{}" ya existe, para generarlo de nuevo primero se debe eliminar.', file=sys.stderr)
            return None
        
        phn_conv_d = self.phn2ohv
        n_samples  = len(self.ds['wav'])


        print(' - create_phn_mfcc_cache, Salvando {} cache'.format(self.phn_mfcc_cache_name))
        
        with h5py.File(os.path.join(self.ds_path, self.phn_mfcc_cache_name),'w') as ds_h5py:

            grp_input_mfcc = ds_h5py.create_group("input_mfcc")
            grp_target_phn = ds_h5py.create_group("target_phn")

            phn_conv_d = self.phn2ohv
            for i_sample in range(n_samples):
                if self.verbose and i_sample%100==0:
                    print(' - Saved: {} of {} samples'.format(i_sample, n_samples))
                y     = self.ds['wav'][i_sample]
                phn_v = self.ds['phn_v'][i_sample]
                
                mfcc, _, _ = calc_MFCC_input(y,
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

                phn  = calc_PHN_target(y, phn_v, phn_conv_d,
                                       hop_length=cfg_d['hop_length'],
                                       win_length=cfg_d['win_length'])

                assert mfcc.shape[0] == phn.shape[0], '- ERROR, create_phn_mfcc_cache: para la muestra {}, mfcc.shape[0] != phn.shape[0]'.format(i_sample)
                
                grp_input_mfcc.create_dataset(str(i_sample), data=mfcc)
                grp_target_phn.create_dataset(str(i_sample), data=phn)

##
        if self.verbose:
            print('Archivo "{}" escrito en disco.'.format(self.phn_mfcc_cache_name))
                
##                ds_h5py['input_mfcc'][i_sample] = mfcc
##                ds_h5py['target_phn'][i_sample] = phn
            
        return None
        
        
    def save_dataset_cache(self):
        if self.verbose:
            print(' - TIMIT, save_dataset_cache: Salvando Archivo de cache: "{}"'.format(self.ds_cache_name))

        with open(os.path.join(self.ds_path, self.ds_cache_name), 'wb') as f:
            pickle.dump(self.ds, f)

        if self.verbose:
            print(' - TIMIT, save_dataset_cache: OK !')
            
        return None
            

    def load_dataset_cache(self):
        if self.verbose:
            print(' - TIMIT, load_dataset_cache: Leyendo Archivo de cache: "{}"'.format(self.ds_cache_name))

        with open(os.path.join(self.ds_path, self.ds_cache_name), 'rb') as f:
            self.ds = pickle.load(f)


        if self.verbose:
            print(' - TIMIT, load_dataset_cache: OK !')

        return None

    def conv_61phn_to_39phn(self, phn61_v):
        ret = (phn61_v@self.phn_61to39_conv_matix)

        if ret.sum() != phn61_v.shape[0]:
            ret_sum = ret.sum(axis=1)
            i_q_v = np.argwhere(ret_sum==0).T[0]

            for i_q in i_q_v:
                for i_rep in range(i_q-1, -1, -1):
                    if ret_sum[i_rep] != 0:
                        ret[i_q] = ret[i_rep]
                        break
                else:
                    for i_rep in range(i_q, ret_sum.axis[0], 1):
                        if ret_sum[i_rep] != 0:
                            ret[i_q] = ret[i_rep]
                            break
                    else:
                        raise Exception(' ERROR: Replace for phoneme "p" not found!!')

        assert ret.sum() == phn61_v.shape[0], Exception(' ERROR: phoneme not indexed!!')
        return ret
                

            
    def read_dataset_from_disk(self, verbose=False):
        
        self.ds = {'wav':    [],  # Sound wave
                   'ds_type':[],  # DS type (trn/tst)
                   'spk_d':  [],  # Spreaker Dialect
                   'spk_g':  [],  # Spreaker Gender
                   'spk_id': [],  # Spreaker Id
                   'sts_id': [],  # Sentense Id
                   'phn_v':  [],  # Phoneme list
                   'txt_v':  [],  # Word sentense text
                   'wrd_v':  []}  # Word list

        if verbose:
            print(' - TIMIT, read_dataset_from_disk, leyendo TIMIT dataset desde:'.format(self.ds_path))

        n_samples = 0
        for ds_type in self.ds_type_v:
            for spk_d in self.ds_dialect_v:
                for spk in os.listdir( os.path.join(self.ds_path, '{}/{}'.format(ds_type, spk_d)) ):
                    spk_g  = spk[0]
                    spk_id = spk[1:]

                    all_file_v = os.listdir( os.path.join(self.ds_path, '{}/{}/{}'.format(ds_type, spk_d, spk)) )
                    file_base_name_v = []
                    for file_name in all_file_v:
                        file_base_name = file_name.split('.')[0]
                        if file_base_name not in file_base_name_v:
                            file_base_name_v.append( file_base_name )
                    
                    for sts_id in file_base_name_v:

                        file_sub_path = '{}/{}/{}/{}'.format(ds_type, spk_d, spk, sts_id)

                        if verbose:
                            print(' - Leyendo: "{}"'.format(file_sub_path))

                        wav   = self.read_wav(file_sub_path+'.WAV')
                        phn_v = self.read_txt(file_sub_path+'.PHN')
                        txt_v = self.read_txt(file_sub_path+'.TXT')[0]
                        wrd_v = self.read_txt(file_sub_path+'.WRD')

                        
                        self.ds['wav'].append(wav)

                        self.ds['ds_type'].append(ds_type)
                        self.ds['spk_d'].append(spk_d)
                        self.ds['spk_g'].append(spk_g)
                        self.ds['spk_id'].append(spk_id)
                        self.ds['sts_id'].append(sts_id)
                        
                        self.ds['phn_v'].append(phn_v)
                        self.ds['txt_v'].append(txt_v)
                        self.ds['wrd_v'].append(wrd_v)

                        n_samples += 1

        for k in self.ds.keys():
            self.ds[k] = np.array(self.ds[k])

        if verbose:
            print(' - TIMIT, read_dataset_from_disk, DateSet leido TIMIT, cantidad de archivos leidos: {}'.format(n_samples))

        return None         



    def read_wav(self, sub_path='./TEST/DR1/FAKS0/SA1.WAV'):
        y, sr = librosa.load(os.path.join( self.ds_path, sub_path), sr=self.sample_rate)
        return y

    def read_txt(self, sub_path='./TEST/DR1/FAKS0/SA1.PHN'):

        file_path = os.path.join(self.ds_path, sub_path)
        with open(file_path , 'r') as f:
            ml_v = f.readlines()

        txt_v = []
        for ml in ml_v:
            l_v = ml.split()
            txt_v.append( (int(l_v[0]), int(l_v[1]), ' '.join(l_v[2:])))
            
        return txt_v


    def stop(self):
        sd.stop()
        return None

    
    def play(self, wave, blocking=False):
        sd.play(np.concatenate([np.zeros(1000),wave]), self.sample_rate, blocking=blocking,loop=False)
        return None


    def get_ds_from_spk_id(self, skp_id='HIT0'):
        skp_id = skp_id.upper()
        ds_spk = {}
        
        f = (self.ds['spk_id'] == skp_id)
        for k in self.ds.keys():
            ds_spk[k] = self.ds[k][f]

        return ds_spk


    def get_ds_from_ds_type(self, ds_type='TRAIN'):
        if ds_type not in self.ds_type_v:
            raise Exception(' - ERROR TIMIT, phoneme_sampler, ds_type no reconocido')

        ds_ret = {}
        
        f = (self.ds['ds_type'] == ds_type)
        for k in self.ds.keys():
            ds_ret[k] = self.ds[k][f]

        return ds_ret


    def make_phoneme_convertion_dicts(self, use_all_phonemes=True):
        """ Arma los diccionarios de conversión de phonemes según la agrupación que se quiera usar"""
        
        self.phn2ohv = {} # Conversión de phonema_str a one_hot_vector
        self.phn2idx = {} # Conversión de phonema_str a index
        self.idx2phn = {} # Conversión de index a phonema_str

        if use_all_phonemes:
            for idx, phn in enumerate(self.ds_phoneme_61_v):
                ohv = np.zeros(len(self.ds_phoneme_61_v))
                ohv[idx] = 1.0

                self.phn2ohv[phn] = ohv
                self.phn2idx[phn] = idx
                self.idx2phn[idx] = phn

            self.n_phn = len(self.ds_phoneme_61_v)
        else:
            raise Exception('TODO: Hacer que se agrupen phonemas similares para disminuri la cantidad de salidas del clasificador.')
        
        return None



    def phoneme_sampler(self, ds_type='TRAIN', n_padd=3000, batch_size=32, n_epochs=1, one_phn_per_wav=True, randomize=True):
        if ds_type is not None:
            ds = self.get_ds_from_ds_type(ds_type)

        ds = self.ds

        if randomize:
            f = np.arange(ds['wav'].shape[0])
            np.random.shuffle(f)
            for k in ds.keys():
                ds[k] = ds[k][f]

        for i_epoch in range(n_epochs):
            if one_phn_per_wav:
                i_wav = 0
                x_v = []
                y_v = []
                while i_wav < ds['wav'].shape[0]:
                    i_phn = np.random.randint(0, len(ds['phn_v'][i_wav]))
                    a,b = ds['phn_v'][i_wav][i_phn][:2]
                    trg = ds['phn_v'][i_wav][i_phn][-1]
                    
                    phn = ds['wav'][i_wav][max(a,b-n_padd):b]
                    inp = np.concatenate( (np.zeros(n_padd-phn.shape[0]), phn) )
                    
                    x_v.append( inp )
                    y_v.append( trg )
                    i_wav += 1

                    if len(x_v) == batch_size:
                        x_v_ = np.array(x_v)
                        y_v_ = np.array(y_v)

                        x_v = []
                        y_v = []
                        yield x_v_, y_v_

    def get_ds_filter(self, ds_filter={'ds_type':'TRAIN'}):
        f = np.ones(self.ds['wav'].shape[0], dtype=np.bool)

        for c, v in ds_filter.items():
            if v is None:
                continue
            
            if c not in self.ds.keys():
                raise Exception(' - ERROR, get_ds_fillter: campo "{}" no encontrado en el ds'.format(c))
            
            f = (self.ds[c] == v) * f

        if f.sum() == 0:
                print('WARNING, no se selecciona ningun dato. Revisar campos de filtrado', file=sys.stderr)
        
        return f

    
    def frame_sampler(self, batch_size=32, n_epochs=1, randomize_samples=True, ds_filter_d={'ds_type':'TRAIN'}):
        
        f_s = self.get_ds_filter(ds_filter_d)
        samples_v = np.arange(f_s.shape[0])[f_s]
        samples_v = [str(i) for i in samples_v]
        

        with h5py.File(os.path.join(self.ds_path, self.phn_mfcc_cache_name),'r') as ds_h5py:
            for i_epoch in range(n_epochs):
                if randomize_samples:
                    np.random.shuffle(samples_v)
                    
                x_v = []
                y_v = []
                for i_s in samples_v:
##                    print('sample', i_s)
##                    print(input_mfcc.shape, target_phn.shape)
                    
                    input_mfcc = ds_h5py['input_mfcc'][i_s][:]
                    target_phn = ds_h5py['target_phn'][i_s][:]

                    for i_f in range(input_mfcc.shape[0]):                        
                        x_v.append( input_mfcc[i_f] )
                        y_v.append( target_phn[i_f] )

                        if len(x_v) == batch_size:
                            yield np.array(x_v), np.array(y_v)
                            x_v = []
                            y_v = []



    def window_sampler(self, batch_size=32, n_epochs=1, randomize_samples=True, ds_filter_d={'ds_type':'TRAIN'}, yield_idxs=False):
        n_timesteps=self.n_timesteps 
        f_s = self.get_ds_filter(ds_filter_d)
        samples_v = np.arange(f_s.shape[0])[f_s]
        samples_v = [str(i) for i in samples_v]
        

        with h5py.File(os.path.join(self.ds_path, self.phn_mfcc_cache_name),'r') as ds_h5py:
            x_v = []
            y_v = []

            idxs_v = []
            for i_epoch in range(n_epochs):
                if randomize_samples:
                    np.random.shuffle(samples_v)
                
                for i_sample in samples_v:
##                    print('sample', i_sample)
##                    print(input_mfcc.shape, target_phn.shape)

                    spec_len = ds_h5py['input_mfcc'][i_sample].shape[0]
                    if spec_len <= n_timesteps:
                        continue
                    
                    # Solamente elegimos un frame por wav
                    # TODO: llevar la cuenta de los frames elegidos como i_sample asi siempre elegimos uno distinto
                    i_s = np.random.randint(0, spec_len-n_timesteps)
                    i_e = i_s + n_timesteps
                    
                    input_mfcc = ds_h5py['input_mfcc'][i_sample][i_s:i_e]
                    target_phn = ds_h5py['target_phn'][i_sample][i_s:i_e]

                    x_v.append( input_mfcc )
                    y_v.append( target_phn )
                    idxs_v.append([i_s, i_e, int(i_sample)])

                    if len(x_v) == batch_size:
                        x_v = np.array(x_v)
                        y_v = np.array(y_v)
                        
                        assert x_v.shape[1] == y_v.shape[1] == n_timesteps

                        if yield_idxs:
                            idxs_v = np.array(idxs_v)
                            yield x_v, y_v, idxs_v
                        else:
                            yield x_v, y_v
                        x_v = []
                        y_v = []
                        idxs_v = []
                            

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
            y_wave = self.ds['wav'][i_sample][step*i_s:step*i_e:step]
            x_wave = np.arange(-0.5, (i_e-i_s)-0.5, 1/step)
            
            h = m_repeat.shape[0]
            y_wave_morm = 0.5* h * ((y_wave-y_wave.min())/(y_wave.max()-y_wave.min()) - 0.5) + h/2
            plt.plot(x_wave, y_wave_morm,  'b', alpha=0.5)
        
        plt.show()

        return None

    
    def calc_class_weights(self, clip=(0,10), ds_filter_d={'ds_type':'TRAIN'}):
        f_s = self.get_ds_filter(ds_filter_d)
        samples_v = np.arange(f_s.shape[0])[f_s]

        samples_v = [str(i) for i in samples_v]

        counter_v = None
        with h5py.File(os.path.join(self.ds_path, self.phn_mfcc_cache_name),'r') as ds_h5py:
            for i_s in samples_v:
                if counter_v is None:
                    counter_v = np.sum(ds_h5py['target_phn'][str(i_s)], axis=0)
                else:
                    counter_v += np.sum(ds_h5py['target_phn'][str(i_s)], axis=0)

        n_samples = int(np.sum(counter_v))
        
        majority = np.mean(counter_v)
        cw_d = {cls: float(majority/count) if count > 0 else 1.0 for cls, count in enumerate(counter_v)}

        if clip is not None:
            for k in cw_d.keys():
               cw_d[k] = np.clip(cw_d[k], clip[0], clip[1])
               
        return cw_d, n_samples


    
# 6 119 737

if __name__ == '__main__':
    import time
    
    if os.name == 'nt':
        ds_path = r'G:\Downloads\timit'
    else:
        ds_path = '/media/sergio/EVO970/UNIR/TFM/code/data_sets/TIMIT'


    ds_cfg_d = {'ds_path':ds_path,
                'use_all_phonemes':True,
                'ds_norm':(0.0, 10.0),
                'remake_samples_cache':False,
                'random_seed':None,
                'ds_cache_name':'timit_cache.pickle',
                'phn_mfcc_cache_name':'phn_mfcc_cache.h5py',
                'verbose':True,

                'sample_rate':16000,  #Frecuencia de muestreo los archivos de audio Hz

                'pre_emphasis': 0.97,
                
                'hop_length_ms':   5.0, # 2.5ms = 40c | 5.0ms = 80c (@ 16kHz)
                'win_length_ms':  25.0, # 25.0ms = 400c (@ 16kHz)
                'n_timesteps':   400, # 800ts*(win_length_ms=2.5ms)= 2000ms  Cantidad de hop_length_ms en una ventana de prediccion.
                
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



    timit = TIMIT(ds_cfg_d)

    
##    mfcc_batch, phn_v_batch, idxs_v_batch = next(iter(timit.window_sampler(50,1, yield_idxs=True)))
##    for mfcc, phn_v, idxs_v in zip(mfcc_batch, phn_v_batch, idxs_v_batch):
##        
##        timit.spec_show(mfcc, phn_v, idxs_v)

    

##    for i_sample in range(0, len(timit.ds['wav'])):
##        m, _, _ = calc_MFCC_input(timit.ds['wav'][i_sample])
##        p    = calc_PHN_target(timit.ds['wav'][i_sample], timit.ds['phn_v'][i_sample], timit.phn2ohv)
##        
##
##        for a, b, p_str in timit.ds['phn_v'][i_sample]:
##            print('{:5d} -> {:5d}   :  delta:{:5d} :  {}'.format(a//40,b//40, (b-a)//40, p_str))
##
##
##        timit.spec_show(m, p)
##
##        break

    
    t0 = time.time()
    n_batch=0
    for mfcc, phn in timit.window_sampler(batch_size=32, n_epochs=1, ds_filter_d={'ds_type':'TRAIN'}):
        n_batch += 1
##        print(mfcc.shape)
##        print(phn.shape)
    print(' Muestreo completo en {:0.02f} s, n_batches={}'.format(time.time() - t0, n_batch))

        
##    for x, y in timit.phoneme_sampler():
##        for i in range(len(x)):
##            if np.argmax(y[i]) == np.argmax(timit.phoneme_d['ae']):
##                timit.play(x[i])
##                input()

    
    
##    for a, b, p in phn_v:
##        y_aux = np.concatenate( (np.zeros(timit.sample_rate), y[a:b] ))
##        y_aux = np.concatenate([y_aux,y_aux,y_aux])
##        _=plt.plot(y_aux)
##        timit.play_sound(y_aux)
##        print(p)
##        plt.show()
