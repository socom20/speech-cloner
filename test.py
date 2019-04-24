import numpy as np
import matplotlib.pyplot as plt
import os, sys
import math
from collections import namedtuple

import tensorflow as tf


from TIMIT_reader import TIMIT
from TARGET_spk_reader import TARGET_spk

from modules import prenet, CBHG

from encoder import encoder_spec_phn
from decoder import decoder_specs


from aux import *


if __name__ == '__main__':

    timit_ds_cfg_d  = load_cfg_d('./hp/ds_enc_cfg_d.json')
    target_ds_cfg_d = load_cfg_d('./hp/ds_dec_cfg_d.json')
    enc_cfg_d = load_cfg_d('./hp/encoder_cfg_d.json')
    dec_cfg_d = load_cfg_d('./hp/decoder_cfg_d.json')

    enc_cfg_d['is_training'] = False
    dec_cfg_d['is_training'] = False


##    timit   = TIMIT(timit_ds_cfg_d)
    trg_spk = TARGET_spk(target_ds_cfg_d)

    encoder = encoder_spec_phn(cfg_d=enc_cfg_d, ds=None)
##    encoder.eval_acc(timit.window_sampler(ds_filter_d={'ds_type':'TEST'}) )


    input('Press ENTER: ')
    

    decoder = decoder_specs(cfg_d=dec_cfg_d, ds=None, encoder=encoder)
    encoder.restore()
    decoder.restore()


    mfcc, mel, stft = next( iter( trg_spk.spec_window_sampler(sample_trn=True) ) )
    y_pred = decoder.predict(mfcc)
    for i in range(32):
        
        fig, axes = plt.subplots(1, 2)
        ax = axes[0]
        ax.imshow( np.repeat(np.concatenate([y_pred.y_mel[i].T, mel[i].T], axis=0), 2, axis=0) )
        ax.set_title('mel spectrogram')
        
        ax = axes[1]
        ax.imshow( np.concatenate([y_pred.y_stft[i].T, stft[i].T], axis=0) )
        ax.set_title('stft spectrogram')
        plt.tight_layout()
        plt.show()

