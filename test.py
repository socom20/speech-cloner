import numpy as np
import matplotlib.pyplot as plt
import os, sys
import math
from collections import namedtuple

import tensorflow as tf


from TIMIT_reader import TIMIT
from TARGET_spk_reader import TARGET_spk
from ARCTIC_reader import ARCTIC


from modules import prenet, CBHG

from encoder import encoder_spec_phn
from decoder import decoder_specs


from aux_func import *

from audio_lib import from_power_to_wav, calc_MFCC_input
import librosa
import sounddevice as sd


def show_spec_comp(mel_true, mel_pred, stft_true, stft_pred, vert=False):
    if vert:
        fig, axes = plt.subplots(2, 1)
    else:
        fig, axes = plt.subplots(1, 2)
    ax = axes[0]
    ax.imshow( np.repeat(np.concatenate([mel_pred.T, mel_true.T], axis=0), 2, axis=0) )
    ax.set_title('mel spectrogram')
    
    ax = axes[1]
    ax.imshow( np.concatenate([stft_pred.T, stft_true.T], axis=0) )
    ax.set_title('stft spectrogram')
    plt.tight_layout()
    plt.show()

    return None


def translate(decoder, mfcc, mel, stft, cfg_d, t_s=5, t_e=60, n_iter=200, output_path='./output', file_name='y_wav', realse=1.0, save_output=False, giffin_lim_input=True):

    hop     = cfg_d['hop_length']
    n_times = cfg_d['n_timesteps']

    if mfcc.shape[0] % n_times != 0:
        print('Padding!!')
        pad_len = (n_times) - (mfcc.shape[0] % n_times)
        
        pad_mfcc = np.zeros( (pad_len , mfcc.shape[1]) )
        mfcc = np.concatenate( [mfcc, pad_mfcc], axis=0 )

        pad_mel = np.zeros( (pad_len, mel.shape[1]) )
        mel = np.concatenate( [mel, pad_mel], axis=0 )
        
        pad_stft = np.zeros( (pad_len, stft.shape[1]) )
        stft = np.concatenate( [stft, pad_stft], axis=0 )
        
        print(mfcc.shape, mel.shape, stft.shape)

        
    n_hop_s = t_s*cfg_d['sample_rate']//hop
    n_hop_e = t_e*cfg_d['sample_rate']//hop

    n_hop_e = min(n_hop_e, mfcc.shape[0])
    
    n_delta = n_times*( (n_hop_e - n_hop_s) // n_times )
    n_s = n_hop_s
    n_e = n_hop_s + n_delta


    if n_e <= n_s:
        raise Exception(' - ERROR, translate: n_e <= n_s.')
    
    mfcc_input = mfcc[n_s:n_e].reshape( (-1, n_times, mfcc.shape[-1]) )
    stft_true  = stft[n_s:n_e].reshape( (-1, n_times, stft.shape[-1]) )

    y_pred = decoder.predict(mfcc_input)

    mel_true = mel[n_s:n_e]
    mel_pred = y_pred.y_mel.reshape( (-1, y_pred.y_mel.shape[-1]) )
    
    stft_true = stft[n_s:n_e]
    stft_pred = y_pred.y_stft.reshape( (-1, y_pred.y_stft.shape[-1]) )


    show_spec_comp(mel_true, mel_pred, stft_true, stft_pred, True)

    if giffin_lim_input:
        y_wav_true = from_power_to_wav(stft_true,
                                       P_dB_norm_factor=cfg_d['P_dB_norm_factor'],
                                       pre_emphasis=cfg_d['pre_emphasis'],
                                       hop_length=cfg_d['hop_length'],
                                       win_length=cfg_d['win_length'],
                                       mean_abs_amp_norm=15*cfg_d['mean_abs_amp_norm'],
                                       n_iter=n_iter,
                                       n_fft=cfg_d['n_fft'],
                                       realse=1.0)
    else:
        y_wav_true = None

    y_wav_pred = from_power_to_wav(stft_pred,
                                   P_dB_norm_factor=cfg_d['P_dB_norm_factor'],
                                   pre_emphasis=cfg_d['pre_emphasis'],
                                   hop_length=cfg_d['hop_length'],
                                   win_length=cfg_d['win_length'],
                                   mean_abs_amp_norm=15*cfg_d['mean_abs_amp_norm'],
                                   n_iter=n_iter,
                                   n_fft=cfg_d['n_fft'],
                                   realse=realse)


    if save_output:
        print('Salvando salida')
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        if giffin_lim_input:    
            librosa.output.write_wav(output_path+'/{}_true.wav'.format(file_name), y_wav_true, cfg_d['sample_rate'], norm=True)
        
        librosa.output.write_wav(output_path+'/{}_pred.wav'.format(file_name), y_wav_pred, cfg_d['sample_rate'], norm=True)

    if giffin_lim_input:
        input('ENTER for play y_wav_true: ')
        sd.play(y_wav_true, cfg_d['sample_rate'], blocking=True)
    
    input('ENTER for play y_wav_pred: ')
    sd.play(y_wav_pred, cfg_d['sample_rate'], blocking=True)
    
    return y_wav_true, y_wav_pred



    
if __name__ == '__main__':

    timit_ds_cfg_d  = load_cfg_d('./hp/ds_enc_cfg_d.json')
    target_ds_cfg_d = load_cfg_d('./hp/ds_dec_cfg_d.json')
    enc_cfg_d = load_cfg_d('./hp/encoder_cfg_d.json')
    dec_cfg_d = load_cfg_d('./hp/decoder_cfg_d.json')

    enc_cfg_d['is_training'] = False
    dec_cfg_d['is_training'] = False


##    timit   = TIMIT(timit_ds_cfg_d)
##    trg_spk = TARGET_spk(target_ds_cfg_d)
    trg_spk = ARCTIC(target_ds_cfg_d)

    encoder = encoder_spec_phn(cfg_d=enc_cfg_d, ds=None)
##    encoder.eval_acc(timit.window_sampler(ds_filter_d={'ds_type':'TEST'}  ) )


    print('Press ENTER to continue: ', end=''); input()
    
    dec_cfg_d['use_target_mel_step2'] = False
    decoder = decoder_specs(cfg_d=dec_cfg_d, ds=None, encoder=encoder)
##    encoder.restore()
    decoder.restore()


    if 0:
        print('TEST 1: trg_spk_mfcc to trg_stft')
        mfcc, mel, stft = next( iter( trg_spk.spec_window_sampler(sample_trn=True) ) )

        y_pred = decoder.predict(mfcc)
        
        for i in range(32):
            show_spec_comp(mel[i], y_pred.y_mel[i], stft[i], y_pred.y_stft[i])

            if True:
                y_wav_true = from_power_to_wav(stft[i],
                                               P_dB_norm_factor=target_ds_cfg_d['P_dB_norm_factor'],
                                               pre_emphasis=target_ds_cfg_d['pre_emphasis'],
                                               hop_length=target_ds_cfg_d['hop_length'],
                                               win_length=target_ds_cfg_d['win_length'],
                                               mean_abs_amp_norm=15*target_ds_cfg_d['mean_abs_amp_norm'],
                                               n_iter=200,
                                               n_fft=target_ds_cfg_d['n_fft'])

                y_wav_pred = from_power_to_wav(y_pred.y_stft[i],
                                               P_dB_norm_factor=target_ds_cfg_d['P_dB_norm_factor'],
                                               pre_emphasis=target_ds_cfg_d['pre_emphasis'],
                                               hop_length=target_ds_cfg_d['hop_length'],
                                               win_length=target_ds_cfg_d['win_length'],
                                               mean_abs_amp_norm=15*target_ds_cfg_d['mean_abs_amp_norm'],
                                               n_iter=200,
                                               n_fft=target_ds_cfg_d['n_fft'])

                print('Press ENTER to continue: y_wav_true: ', end=''); input()
                trg_spk.play(np.tile(y_wav_true,3), True)
                print('Press ENTER to continue: y_wav_pred: ', end=''); input()
                trg_spk.play(np.tile(y_wav_pred,3), True)


                print('Press ENTER to continue: ', end=''); input()
                

    if 0:
        print('TEST 2: trg_spk_mfcc to target_spk_wav')
        i = np.argwhere( trg_spk.get_ds_filter({'spk_id': 'bdl'}) ).T[0][10]
        mfcc, mel, stft = trg_spk.get_spec(i)[:3]
        y_wav_true, y_wav_pred = translate(decoder, mfcc, mel, stft, target_ds_cfg_d, t_s=0, t_e=120, output_path='./test_2', save_output=True)
    
    if 0:
        print('TEST 3: other_spk_mfcc to target_spk_wav')
        ds_arctic_cfg_d = { 'ds_path':'/media/sergio/EVO970/UNIR/TFM/code/data_sets/ARCTIC/cmu_arctic',
                            'ds_norm':(0.0, 1.0),
                            'remake_samples_cache':False,
                            'random_seed':None,
                            'ds_cache_name':'arctic_cache.pickle',
                            'spec_cache_name':'spec_cache.h5py',
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

        arctic = ARCTIC(ds_arctic_cfg_d)
        mfcc, mel, stft, phn = arctic.get_spec(np.argmax( arctic.get_ds_filter({'spk_id':'rms', 'sts_id':'a0407'}) ))
        y_wav_true, y_wav_pred = translate(decoder, mfcc, mel, stft, target_ds_cfg_d, t_s=0, t_e=5, output_path='./test_3', save_output=True)
    

    if 1:
        print('TEST 4: other_spk_mfcc to target_spk_wav')
##        wav_path = '/media/sergio/EVO970/UNIR/TFM/dataset/VCTK-Corpus/VCTK-Corpus/wav48/p299/p299_005.wav'
        wav_path = '/media/sergio/EVO970/UNIR/TFM/dataset/VCTK-Corpus/VCTK-Corpus/wav48/p374/p374_023.wav'
##        wav_path = '/media/sergio/EVO970/UNIR/TFM/dataset/VCTK-Corpus/VCTK-Corpus/wav48/p339/p339_008.wav'
 ##        wav_path = '/media/sergio/EVO970/UNIR/TFM/dataset/VCTK-Corpus/VCTK-Corpus/wav48/p267/p267_023.wav'

##        wav_path = '/media/sergio/EVO970/UNIR/TFM/dataset/VCTK-Corpus/VCTK-Corpus/wav48/p234/p234_020.wav'

##        wav_path = '/media/sergio/EVO970/UNIR/TFM/code/data_sets/ARCTIC/cmu_arctic/cmu_us_slt_arctic/wav/arctic_a0011.wav'
##        wav_path = '/media/sergio/EVO970/UNIR/TFM/code/data_sets/ARCTIC/cmu_arctic/cmu_us_rms_arctic/wav/arctic_a0011.wav'
##        wav_path = '/media/sergio/EVO970/UNIR/TFM/code/data_sets/ARCTIC/cmu_arctic/cmu_us_ksp_arctic/wav/arctic_a0011.wav'
##        wav_path = '/media/sergio/EVO970/UNIR/TFM/code/data_sets/ARCTIC/cmu_arctic/cmu_us_jmk_arctic/wav/arctic_a0011.wav'
##        wav_path = '/media/sergio/EVO970/UNIR/TFM/code/data_sets/ARCTIC/cmu_arctic/cmu_us_clb_arctic/wav/arctic_a0011.wav'
##        wav_path = '/media/sergio/EVO970/UNIR/TFM/code/data_sets/ARCTIC/cmu_arctic/cmu_us_bdl_arctic/wav/arctic_a0011.wav'  # Target

##        wav_path = '../sergio.ogg'


##        wav_path = '/media/sergio/EVO970/UNIR/TFM/code/data_sets/TRG/L. Frank Baum/The Wonderful Wizard of Oz/The Wonderful Wizard of Oz-04 Chapter 4.mp3'
        
        wav_cfg_d = {'wav_path':wav_path,
                     'wav_norm':(0.0, 1.0),
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

        wav_cfg_d['hop_length'] = int( wav_cfg_d['hop_length_ms'] * wav_cfg_d['sample_rate'] / 1000 )
        wav_cfg_d['win_length'] = int( wav_cfg_d['win_length_ms'] * wav_cfg_d['sample_rate'] / 1000 )
        
        y, _ = librosa.load(wav_cfg_d['wav_path'], wav_cfg_d['sample_rate'])

        mfcc, mel, stft = calc_MFCC_input( y,
                                           sr=wav_cfg_d['sample_rate'],
                                           pre_emphasis=wav_cfg_d['pre_emphasis'],
                                           hop_length=wav_cfg_d['hop_length'],
                                           win_length=wav_cfg_d['win_length'],
                                           n_mels=wav_cfg_d['n_mels'],
                                           n_mfcc=wav_cfg_d['n_mfcc'],
                                           n_fft=wav_cfg_d['n_fft'],
                                           window=wav_cfg_d['window'],
                                           mfcc_normaleze_first_mfcc=wav_cfg_d['mfcc_normaleze_first_mfcc'],
                                           mfcc_norm_factor=wav_cfg_d['mfcc_norm_factor'],
                                           calc_mfcc_derivate=wav_cfg_d['calc_mfcc_derivate'],
                                           M_dB_norm_factor=wav_cfg_d['M_dB_norm_factor'],
                                           P_dB_norm_factor=wav_cfg_d['P_dB_norm_factor'],
                                           mean_abs_amp_norm=wav_cfg_d['mean_abs_amp_norm'],
                                           clip_output=wav_cfg_d['clip_output'])
        
        y_wav_true, y_wav_pred = translate(decoder,
                                           mfcc,
                                           mel,
                                           stft,
                                           target_ds_cfg_d,
                                           t_s=0, t_e=52,
                                           output_path='./test_4', file_name='{}'.format(os.path.split(wav_path)[1].split('.')[0]),
                                           save_output=True,
                                           realse=1.4,
                                           giffin_lim_input=True)




