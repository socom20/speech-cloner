import numpy as np
import matplotlib.pyplot as plt
import os, sys
import math


import tensorflow as tf



from TIMIT_reader import TIMIT
from modules import prenet, CBHG


class encoder_spec_phn:
    def __init__(self, cfg_d={}, ds=None, mode='train', verbose=False):
        self.cfg_d = cfg_d
        self.ds    = ds

        self.mode = mode

        # Creo Modelo
        self._build_model( input_shape=self.cfg_d['input_shape'],
                           n_output   =self.cfg_d['n_output'],
                           encoder_num_banks=self.cfg_d['encoder_num_banks'],
                           num_highwaynet_blocks=self.cfg_d['num_highwaynet_blocks'],
                           dropout_rate=self.cfg_d['dropout_rate'],
                           is_training=self.cfg_d['is_training'],
                           scope=self.cfg_d['model_name'],
                           reuse=None )

        # Armo la funcion de costo
        self._make_loss(reuse=None)

        # Creo optimizador
        self._create_optimizer(reuse=None)

        
        
        # Inicio la sesion de tf
        self._create_tf_session()

        # Creamos Saver (se crea el merge del summary)
        self._create_saver()

        # Inicializo las variables
        self._initialize_variables()
        
        return None


    def _create_tf_session(self):
        tf_sess_config = tf.ConfigProto()
        tf_sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tf_sess_config)

        return None


        
    def _build_model(self, input_shape=(800, 256), n_output=48, encoder_num_banks=16, num_highwaynet_blocks=4, dropout_rate=0.5, is_training=True, scope="model", reuse=None):
        '''
        Args:
          inputs: A 2d tensor with shape of [N, T_x, E], with dtype of int32. Encoder inputs.
          is_training: Whether or not the layer is in training mode.
          scope: Optional scope for `variable_scope`
          reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.
        
        Returns:
          A collection of Hidden vectors. So-called memory. Has the shape of (N, T_x, E).
        '''

        # Inputs para el modelo
        self.inputs = tf.placeholder(tf.float32, (None,)+input_shape)# (N, T, E)
        # Targets para el modelo
        self.target = tf.placeholder(tf.float32, (None, n_output) )  # (N, T, O)

        
        embed_size  = input_shape[-1]
        inputs = self.inputs
        with tf.variable_scope(scope, reuse=reuse): 
            # Encoder pre-net
            prenet_out = prenet(inputs, None, embed_size, dropout_rate, is_training, scope="prenet", reuse=None) # (N, T_x, E/2)
            
            # Encoder CBHG 
            CBHG_out = CBHG(prenet_out, embed_size, encoder_num_banks, num_highwaynet_blocks, dropout_rate, is_training, scope="CBHG", reuse=None) # (N, T_x, E)


            # Classificator
            y_logits     = tf.layers.dense(CBHG_out, n_output, activation=None, name="y_logits")  # (N, T, O)   tf.nn.relu
            y_pred       = tf.nn.softmax(y_logits, name='y_pred')  # (N, T, O)
            y_pred_class = tf.to_int32(tf.argmax(y_logits, axis=-1), name='y_pred_class')  # (N, T)


        self.y_pred       = y_pred
        self.y_pred_class = y_pred_class
        self.y_logits     = y_logits
        
        return None
    

    def _make_loss(self, reuse=None):
        with tf.variable_scope('loss',reuse=reuse):
            self.loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.target,
                                                                                   logits=self.y_logits), axis=-1)
        tf.summary.scalar('{}/loss'.format(self.mode), self.loss)
        return None

    def _create_optimizer(self, reuse=None):
        with tf.variable_scope('opt',reuse=reuse):
            self.learning_rate       = tf.Variable(self.cfg_d['learning_rate'], trainable=False, dtype=tf.float32, name='learning_rate')
            self.learning_rate_start = tf.Variable(self.cfg_d['learning_rate'], trainable=False, dtype=tf.float32, name='learning_rate_start')
            self.learning_rate_decay = tf.Variable(self.cfg_d['decay'],         trainable=False, dtype=tf.float32, name='learning_rate_decay')

            self.global_step = tf.Variable(0, trainable=False, dtype=tf.int32, name='global_step')
            
            self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                              beta1=self.cfg_d['beta1'],
                                              beta2=self.cfg_d['beta2'],
                                              epsilon=self.cfg_d['epsilon'])

            self.train_step = self.opt.minimize(loss=self.loss,
                                                global_step=self.global_step)
            
            self.lr_decay_op = tf.assign(self.learning_rate, self.learning_rate_start / (1. + self.learning_rate_decay * tf.cast(self.global_step, tf.float32)))

            tf.summary.scalar('{}/learning_rate'.format(self.mode), self.learning_rate)
        return None

    

    def _initialize_variables(self):
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)
        return None



    def _create_saver(self):
        self.saver = tf.train.Saver()
        self.summary_merged = tf.summary.merge_all()
        
        if self.mode == 'train':
            self.train_writer = tf.summary.FileWriter(self.cfg_d['log_dir'] + '/train', self.sess.graph)
        elif self.mode == 'test':
            self.test_writer = tf.summary.FileWriter(self.cfg_d['log_dir'] + '/test')
        else:
            raise Exception(' - ERROR, self.mode={} not implemented'.format(self.mode))

        return None


    def save(self, save_path=None):
        if save_path is None:
            save_path = self.cfg_d['model_path']+'model.ckpt'
            
        self.saver.save(self.sess, save_path)

        return None

    def restore(self, save_path=None):
        if save_path is None:
            save_path = self.cfg_d['model_path']+'model.ckpt'
            
        self.saver.restore(self.sess, save_path)
        return None


    def exec_train_step(self, inputs, target):
        
        loss, global_step, train_step, summary_merged = self.sess.run([self.loss, self.global_step, self.train_step, self.summary_merged], feed_dict={self.inputs:inputs, self.target:target})

        self.train_writer.add_summary(summary_merged, train_step)
        
        return (loss, global_step, train_step)


    def train(self):
        self.cfg_d['n_samples_trn'] = self.ds.get_ds_filter( self.cfg_d['ds_trn_filter_d'] ).sum()
            

        sampler_trn = self.ds.window_sampler(batch_size=self.cfg_d['batch_size'],
                                             n_epochs=999999,
                                             randomize_samples=self.cfg_d['randomize_samples'],
                                             ds_filter_d=self.cfg_d['ds_trn_filter_d'])


        sampler_val = self.ds.window_sampler(batch_size=self.cfg_d['batch_size'],
                                             n_epochs=999999,
                                             randomize_samples=self.cfg_d['randomize_samples'],
                                             ds_filter_d=self.cfg_d['ds_val_filter_d'])
        
        

        print(' Startin Training ...')
        self.i_epoch = 0
        for mfcc, phn_v in sampler_trn:
            loss, global_step, train_step = self.exec_train_step(mfcc, phn_v)

            print(' - {}: {}'.format(train_step, loss) )
            
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
                'n_timesteps':800,
                
                'n_mels':128,
                'n_mfcc':40,
                'mfcc_normaleze_first_mfcc':True,
                'mfcc_norm_factor':0.01,
                'mean_abs_amp_norm':0.003,
                'clip_output':True}



    timit = None #TIMIT(ds_cfg_d)

    
    model_cfg_d = {'model_name':'phn_model_epoch={epoch:03d}.kmodel',
                   'model_path':'./models_phn',
                   'input_shape':(ds_cfg_d['n_timesteps'], ds_cfg_d['n_mfcc']),
                   'n_output':61,

                   'encoder_num_banks':16,
                   'num_highwaynet_blocks':4,
                   'dropout_rate':0.5,
                   'is_training':True,

                   'model_name':'encoder',

                   'learning_rate': 1e-3,
                   'beta1':0.9,
                   'beta2':0.999,
                   'epsilon':1e-8,
                   'decay':0.0,

                   
                   'log_dir':'./Graph',

                   'ds_trn_filter_d':{'ds_type':'TRAIN'},
                   'ds_val_filter_d':{'ds_type':'TEST'},
                   'ds_tst_filter_d':{'ds_type':'TEST'},
                   'randomize_samples':True,
                   
                   'n_epochs': 1000,
                   'batch_size': 32}




    

    np.random.seed(0)

    x = np.random.random( (32,) + model_cfg_d['input_shape'] )
    y = np.random.random( (32, model_cfg_d['n_output']) )
    y[np.arange(y.shape[0]), np.argmax(y,axis=1)] = 1.0
    y[y != 1] = 0.0
                      
    model = encoder_spec_phn(model_cfg_d, timit)
    model.exec_train_step(x, y)
##    model.train()


