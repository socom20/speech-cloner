import numpy as np
import matplotlib.pyplot as plt
import os, sys
import math

import keras
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


from TIMIT_reader import TIMIT




class phn_classificator_model:

    def __init__(self, ds):
        self.ds = ds

        return None

    def set_config_d(self, model_cfg_d):
        self.cfg_d = model_cfg_d


        self.model = self.create_model(input_shape=self.cfg_d['input_shape'],
                                       n_output=self.cfg_d['n_output'],
                                       n_hidden_v=self.cfg_d['n_hidden_v'],
                                       hidden_activation_type=self.cfg_d['hidden_activation_type'],
                                       output_activation=self.cfg_d['output_activation'],
                                       opt_hp_d=self.cfg_d['opt_hp_d'])


        if not os.path.exists(self.cfg_d['model_path']):
            print(' - Creando directorio: ""'.format(self.cfg_d['model_path']))
            os.mkdir(self.cfg_d['model_path'])

        return None
        
        
    def create_model(self, input_shape=(40,), n_output=61, n_hidden_v=[10,5,3], hidden_activation_type='relu', output_activation='softmax', opt_hp_d={'learning_rate':1e-3,'beta_1':0.9,'beta_2':0.999,'epsilon':1e-8,'decay':0.0}):
        model = keras.models.Sequential()

        if len(input_shape) == 1:
            model.add( keras.layers.InputLayer( input_shape=input_shape ) )
        else:
            model.add( keras.layers.Flatten( input_shape=input_shape ) )


        for n_dense in n_hidden_v:
            model.add( keras.layers.Dense(n_dense,
                                          activation=hidden_activation_type,
                                          kernel_initializer=keras.initializers.RandomNormal(stddev=0.1),
                                          bias_initializer=keras.initializers.RandomNormal(stddev=0.1) ) )

            model.add( keras.layers.Dropout(0.5) )

        model.add( keras.layers.Dense(n_output,
                                      activation=output_activation,
                                      kernel_initializer=keras.initializers.RandomNormal(stddev=0.1),
                                      bias_initializer=keras.initializers.RandomNormal(stddev=0.1) ) )
        
        opt = keras.optimizers.Adam(**opt_hp_d)
        
    ##    opt = keras.optimizers.rmsprop(learning_rate)

        if output_activation == 'softmax':
    ##        model.compile(loss='categorical_hinge', optimizer=opt, metrics=['categorical_accuracy'])
            model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_accuracy'])
        else:
            model.compile(loss='mse', optimizer=opt)

        return model


    def train(self):

        tbCallBack = keras.callbacks.TensorBoard(log_dir=self.cfg_d['log_dir'],
                                                 write_graph=True)

        svCallBack = keras.callbacks.ModelCheckpoint(filepath=self.cfg_d['model_path']+'/'+self.cfg_d['model_name'],
                                                     monitor='trn_loss',
                                                     verbose=0,
                                                     save_best_only=False,
                                                     save_weights_only=False,
                                                     mode='auto',
                                                     period=1)
        
        self.cfg_d['cw'], self.cfg_d['n_samples'] = self.ds.calc_class_weights()
        
        if not self.cfg_d['use_class_weight']:
            self.cfg_d['cw'] = None
            

        sampler_trn = timit.frame_sampler(batch_size=self.cfg_d['batch_size'],
                                          n_epochs=999999,
                                          randomize_samples=self.cfg_d['randomize_samples'],
                                          ds_filters_d=self.cfg_d['ds_filters_d'])

        print(' Empezando entrenamiento:')
        print('  - n_samples:', self.cfg_d['n_samples'])
        print('  - steps_per_epoch:', self.cfg_d['n_samples']//self.cfg_d['batch_size'])
        
        model.model.fit_generator(sampler_trn,
                                  steps_per_epoch=math.ceil(self.cfg_d['n_samples']/self.cfg_d['batch_size']),
                                  epochs=self.cfg_d['n_epochs'],
                                  verbose=1,
                                  callbacks=[tbCallBack, svCallBack],
##                                  validation_data=None,
##                                  validation_steps=None,
##                                  validation_freq=1,
                                  class_weight=self.cfg_d['cw'],
                                  max_queue_size=10,
                                  workers=1,
                                  use_multiprocessing=False,
                                  shuffle=False,
                                  initial_epoch=0)


                                  
##        for mfcc, phn in sampler_trn:
##            h     = self.model.fit(mfcc, phn, batch_size=self.cfg_d['batch_size'], epochs=1, verbose=False, class_weight=self.cfg_d['cw'], callbacks=[tbCallBack])

        return None


    def predict(x):
        y_ = self.model.predict(x, batch_size=1024, verbose=False)
        return y_


    def summary(self):
        self.model.summary()
        return None
        

if __name__ == '__main__':
    if os.name == 'nt':
        ds_path = r'G:\Downloads\timit'
    else:
        ds_path = '/media/sergio/EVO970/UNIR/TFM/code/data_sets/TIMIT'

        

    ds_cfg_d = {'ds_path':ds_path,
                'use_all_phonemes':True,
                'ds_norm':(0.0, 10.0),
                'remake_samples_cache':False,
                   'random_seed':0,
                'ds_cache_name':'timit_cache.pickle',
                'phn_mfcc_cache_name':'phn_mfcc_cache.h5py',
                'verbose':True,

                'sample_rate':16000,

                'pre_emphasis':0.97,
                'hop_length': 40,
                'win_length':400,
                'n_mels':128,
                'n_mfcc':40,
                'mfcc_normaleze_first_mfcc':True,
                'mfcc_norm_factor':0.01,
                'mean_abs_amp_norm':0.003,
                'clip_output':True}



    timit = TIMIT(ds_cfg_d)

    
    model_cfg_d = {'model_name':'phn_model_epoch={epoch:03d}.kmodel',
                   'model_path':'./models_phn',
                   'input_shape':(timit.n_mfcc,),
                   'n_output':timit.n_phn,
                   'n_hidden_v':[256,256,128],
                   'hidden_activation_type':'tanh',
                   'output_activation':'softmax',
                   'opt_hp_d': {'lr':1e-3,'beta_1':0.9,'beta_2':0.999,'epsilon':1e-8,'decay':0.0},
                   'log_dir':'./Graph',

                   'ds_filters_d':{'ds_type':'TRAIN'},
                   'randomize_samples':True,
                   'use_class_weight':True,
                   'n_epochs': 1000,
                   'batch_size': 512}




    model = phn_classificator_model(timit)


    model.set_config_d(model_cfg_d)

    model.summary()

    model.train()
