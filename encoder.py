import numpy as np
import matplotlib.pyplot as plt
import os, sys
import math
from collections import namedtuple

import tensorflow as tf


from TIMIT_reader import TIMIT
from modules import prenet, CBHG
from aux_func import *


class encoder_spec_phn:
    def __init__(self, cfg_d={}, ds=None):
        self.cfg_d = cfg_d
        self.ds    = ds

        self.i_global_step = 0
        self.i_epoch       = 0
        self.summary_v     = []
        

        # Creo Modelo
        self._build_model( input_shape=self.cfg_d['input_shape'],
                           n_output   =self.cfg_d['n_output'],
                           embed_size=self.cfg_d['embed_size'],
                           num_conv_banks=self.cfg_d['num_conv_banks'],
                           num_highwaynet_blocks=self.cfg_d['num_highwaynet_blocks'],
                           dropout_rate=self.cfg_d['dropout_rate'],
                           is_training=self.cfg_d['is_training'],
                           scope=self.cfg_d['model_name'],
                           use_Cudnn=self.cfg_d['use_Cudnn'],
                           use_lstm=self.cfg_d['use_lstm'],
                           reuse=None )

        
        # Armo la funcion de costo
        self._build_loss(reuse=None)

        # Armo las metricas
        self._build_metric(reuse=None)


        if self.cfg_d['is_training']:
            # Creo optimizador
            self._build_optimizer(reuse=None)


        # Inicio la sesion de tf
        self._create_tf_session()

        # Creamos Saver (se crea el merge del summary)
        self._create_saver()

        # Si no entrenamos retiramos todas las variables de la collection TRAINABLE_VARIABLES
        if not self.cfg_d['is_training']:
            for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.cfg_d['model_name']):
                tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES).remove(v)

        
        # Inicializo las variables
        self._initialize_variables()
        
        return None


    def _create_tf_session(self):
        tf_sess_config = tf.ConfigProto()
        tf_sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tf_sess_config)

        return None


        
    def _build_model(self, input_shape=(800, 256), n_output=48, embed_size=None, num_conv_banks=16, num_highwaynet_blocks=4, dropout_rate=0.5, is_training=True, scope="model", use_Cudnn=False, use_lstm=False, reuse=None):
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

        if embed_size is None:
            embed_size  = input_shape[-1]
            
        with tf.variable_scope(scope, reuse=reuse):
            # Inputs para el modelo
            inputs = tf.placeholder(tf.float32, (None,)+tuple(input_shape), name='inputs')# (N, T, E)
            
            # Targets para el modelo
            target = tf.placeholder(tf.float32, (None, input_shape[0], n_output), name='target' )  # (N, T, O)
            
            # Encoder pre-net
            prenet_out = prenet(inputs, None, embed_size, dropout_rate, is_training, scope="prenet", reuse=None) # (N, T_x, E/2)
            
            # Encoder CBHG 
            CBHG_out = CBHG(prenet_out, embed_size, num_conv_banks, num_highwaynet_blocks, dropout_rate, is_training, scope="CBHG", use_Cudnn=use_Cudnn, use_lstm=use_lstm, reuse=None) # (N, T_x, E)


            # Classificator
            y_logits     = tf.layers.dense(CBHG_out, n_output, activation=None, name="y_logits")  # (N, T, O)   tf.nn.relu
            y_pred       = tf.nn.softmax(y_logits, name='y_pred')  # (N, T, O)
            y_pred_class = tf.to_int32(tf.argmax(y_logits, axis=-1), name='y_pred_class')  # (N, T)


        # Hago que las variables sean parte del objeto modelo
        self.inputs       = inputs
        self.target       = target

        self.CBHG_out     = CBHG_out
        self.y_pred       = y_pred
        self.y_pred_class = y_pred_class
        self.y_logits     = y_logits
        
        return None

    def get_input(self):
        return self.inputs


    def get_outputs(self):
        output_nt = namedtuple('output_nt', 'y_pred y_pred_class y_logits CBHG_out')
        return output_nt(self.y_pred, self.y_pred_class, self.y_logits, self.CBHG_out)


    def _build_loss(self, reuse=None):
        with tf.variable_scope('loss', reuse=reuse):
            self.loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.target,
                                                                                   logits=self.y_logits), name='cross_entropy')
            
        self.summary_v += [tf.summary.scalar('metric/loss', self.loss)]
        return None


    def _build_metric(self, reuse=None):
        with tf.variable_scope('metric', reuse=reuse):
            labels      = tf.to_int32(tf.argmax(self.target, axis=-1), name='labels')      # dims  [N, T]
            predictions = tf.to_int32(tf.argmax(self.y_pred, axis=-1), name='predictions') # dims  [N, T]

            
            self.acc = tf.contrib.metrics.accuracy(labels=labels, predictions=predictions, name='accuracy')
            self.mse = tf.reduce_mean( tf.squared_difference(self.y_pred, self.target),    name='mean_squared_error')

            num_classes = self.cfg_d['n_output']
            self.batch_confusion     = tf.confusion_matrix(labels=tf.reshape(labels, [-1]), predictions=tf.reshape(predictions, [-1]), num_classes=num_classes, dtype=tf.float32, name='batch_confusion_matrix')
            self.batch_confusion_img = tf.reshape(self.batch_confusion, [1, num_classes, num_classes, 1], name='batch_confusion_img')
            
        self.summary_v += [tf.summary.scalar('metric/acc', self.acc),
                           tf.summary.scalar('metric/mse', self.mse),
                           tf.summary.image('metric/batch_conf_img', self.batch_confusion_img)]
        return None

    
    def _build_optimizer(self, reuse=None):
        with tf.variable_scope('opt',reuse=reuse):
            self.learning_rate       = tf.Variable(self.cfg_d['learning_rate'], trainable=False, dtype=tf.float32, name='learning_rate')
            self.learning_rate_start = tf.Variable(self.cfg_d['learning_rate'], trainable=False, dtype=tf.float32, name='learning_rate_start')
            self.learning_rate_decay = tf.Variable(self.cfg_d['decay'],         trainable=False, dtype=tf.float32, name='learning_rate_decay')

            self.global_step = tf.Variable(0, trainable=False, dtype=tf.int32, name='global_step')
            self.i_epoch_tf  = tf.Variable(0, trainable=False, dtype=tf.int32, name='epoch')
            
            self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                              beta1=self.cfg_d['beta1'],
                                              beta2=self.cfg_d['beta2'],
                                              epsilon=self.cfg_d['epsilon'])

            
            # https://www.tensorflow.org/api_docs/python/tf/contrib/layers/batch_norm
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_step = self.opt.minimize(loss=self.loss,
                                                    global_step=self.global_step)
            
            self.lr_decay_op = tf.assign(self.learning_rate, self.learning_rate_start / (1. + self.learning_rate_decay * tf.cast(self.i_epoch_tf, tf.float32)))

            self.i_epoch_inc_op = tf.assign(self.i_epoch_tf, self.i_epoch_tf + 1)
                                            
            self.summary_v += [tf.summary.scalar('learning_rate',       self.learning_rate),
                               tf.summary.scalar('global_step',          self.global_step),
                               tf.summary.scalar('i_epoch_tf',          self.i_epoch_tf),
                               tf.summary.scalar('learning_rate_decay', self.learning_rate_decay),
                               tf.summary.scalar('learning_rate_start', self.learning_rate_start)]

            
        return None

    

    def _initialize_variables(self):
        self.init_v = []
        self.init_v.append( tf.global_variables_initializer() )
        self.init_v.append( tf.local_variables_initializer() )
        self.sess.run(self.init_v)
        return None



    def _create_saver(self):
        self.saver = tf.train.Saver(max_to_keep=9999, keep_checkpoint_every_n_hours=0.5)
        
        self.summary_merged = tf.summary.merge(self.summary_v)
        if self.cfg_d['is_training']:
            self.trn_writer = tf.summary.FileWriter(self.cfg_d['log_dir'] + '/trn', graph=self.sess.graph)
            self.val_writer = tf.summary.FileWriter(self.cfg_d['log_dir'] + '/val')
            
##        elif self.cfg_d['is_training']:
##            self.tst_writer = tf.summary.FileWriter(self.cfg_d['log_dir'] + '/tst')
##        else:
##            raise Exception(' - ERROR, self.mode={} not implemented'.format(self.mode))

        return None


    def save(self, save_path=None, i_checkpoint=None, verbose=True):
        if save_path is None:
            save_path = '{}/{}'.format(self.cfg_d['model_path'], self.cfg_d['model_name'])

        if i_checkpoint is None:
            i_checkpoint = self.i_global_step

        self.saver.save(self.sess, save_path, int(i_checkpoint))

        if verbose:
            print(' Saved: "{}"'.format(self.saver.last_checkpoints[-1]))

        return None


    def restore(self, save_path=None, i_checkpoint=None):
        if save_path is None:
            if i_checkpoint is None:
                save_path = tf.train.latest_checkpoint(self.cfg_d['model_path'])
            else:
                save_path = '{}/{}-{}'.format(self.cfg_d['model_path'], self.cfg_d['model_name'], int(i_checkpoint))
            
        try:
            self.saver.restore(self.sess, save_path)
            print('Restored: "{}"'.format(save_path))
        except:
            print(' Model not found: {}'.format(save_path), file=sys.stderr)
            sys.exit(1)

            
        return None


    def exec_train_step(self, inputs, target):
        
        ret = self.sess.run([self.loss,
                             self.acc,
                             self.mse,
                             self.global_step,
                             self.train_step,
                             self.summary_merged], feed_dict={self.inputs:inputs, self.target:target})
        
        loss, acc, mse, global_step, train_step, summary_merged = ret

        self.i_global_step = global_step
        self.trn_writer.add_summary(summary_merged, self.i_global_step)
        
        return (loss, acc, mse, global_step, train_step)

    

    def exec_calc_metrics(self, inputs, target, summary_mode='validation'):

        ret = self.sess.run([self.acc,
                             self.mse,
                             self.loss,
                             self.batch_confusion_img,
                             self.global_step,
                             self.summary_merged], feed_dict={self.inputs:inputs, self.target:target})

        
        acc, mse, loss, batch_confusion_img, global_step, summary_merged = ret


        self.i_global_step = global_step
        if summary_mode == 'train':
            self.val_writer.add_summary(summary_merged, self.i_global_step)
        elif summary_mode == 'validation':
            self.val_writer.add_summary(summary_merged,  self.i_global_step)
        elif summary_mode == 'test':
            self.tst_writer.add_summary(summary_merged,  self.i_global_step)            
        else:
            raise Exception(' - ERROR, summary_mode={} not implemented'.format(summary_mode))
            
        return acc, mse, loss

    
    def train(self):
        if not self.cfg_d['is_training']:
            raise Exception('Model is not in training model')
        
        self.cfg_d['n_samples_trn'] = self.ds.get_ds_filter( self.cfg_d['ds_trn_filter_d'] ).sum()

        self.cfg_d['n_steps_epoch_trn'] = self.cfg_d['n_samples_trn']//self.cfg_d['batch_size']

        self.sampler_trn = self.ds.window_sampler(batch_size=self.cfg_d['batch_size'],
                                                  n_epochs=99999999,
                                                  randomize_samples=self.cfg_d['randomize_samples'],
                                                  ds_filter_d=self.cfg_d['ds_trn_filter_d'])


        self.sampler_val = self.ds.window_sampler(batch_size=self.cfg_d['batch_size'],
                                                  n_epochs=99999999,
                                                  randomize_samples=self.cfg_d['randomize_samples'],
                                                  ds_filter_d=self.cfg_d['ds_val_filter_d'])

        self.iter_val  = iter(self.sampler_val)
        
        print(' Starting Training ...')
        print(' n_samples_trn:    ', self.cfg_d['n_samples_trn'])
        print(' n_steps_epoch_trn:', self.cfg_d['n_steps_epoch_trn'])
        print(' batch_size:       ', self.cfg_d['batch_size'])
        print(' n_epochs:         ', self.cfg_d['n_epochs'])
        input('Press --ENTER--')


        # Refresh: i_epoch lr 
        self.i_epoch, self.lr = self.sess.run( [self.i_epoch_tf, self.lr_decay_op] )
        
        for mfcc_trn, phn_v_trn in self.sampler_trn:
            loss, acc, mse, global_step, train_step = self.exec_train_step(mfcc_trn, phn_v_trn)
                        
            print(' - i_epoch={}   global_step={}   loss_trn={:6.3f}  acc_trn={:6.3f}  mse_trn={:6.3f}'.format(self.i_epoch, global_step, loss, acc, mse) )

            if (global_step/self.cfg_d['n_steps_epoch_trn']) % self.cfg_d['save_each_n_epochs'] == 0:
                # new epoch
                print(' Saving, epoch={} ...'.format(self.i_epoch))
                self.save()
                mfcc_val, phn_v_val = next(self.iter_val)
                acc_val, mse_val, loss_val = self.exec_calc_metrics(mfcc_val, phn_v_val)
                print(' - i_epoch={}   global_step={}   loss_val={:6.3f}  acc_val={:6.3f}   mse_val={:6.3f}'.format(self.i_epoch, int(global_step), loss_val, acc_val, mse_val) )

                


            if global_step % self.cfg_d['n_steps_epoch_trn'] == 0:
                _, self.i_epoch, self.lr = self.sess.run( [self.i_epoch_inc_op, self.i_epoch_tf, self.lr_decay_op] )
                
                if self.i_epoch >= self.cfg_d['n_epochs']:
                    break


        print(' End of Training !!!')
        return None


    def predict(self, x, batch_size=32):
        y_pred_v = []
        
        for i_s in range(0, x.shape[0], batch_size):
            x_batch = x[i_s:min(i_s+batch_size, x.shape[0])]
            y_pred = self.sess.run(self.y_pred, {self.inputs:x_batch} )
            y_pred_v.append(y_pred)

        return np.concatenate(y_pred_v, axis=0)


    def run(self, var, feed_dict={}):
        return self.sess.run(var, feed_dict)

    
    def eval_acc(self, ds_sampler, n_batchs=100):
        n_c = 0
        n_t = 0
        for i_batch, (mfcc_batch, phn_v_batch) in enumerate(ds_sampler):
            
            y_pred = self.sess.run(self.y_pred, {self.inputs: mfcc_batch})
            y_dec  = np.argmax( y_pred, axis=-1)
            y_true = np.argmax( phn_v_batch, axis=-1)
##            f = ( np.abs(mfcc_batch).sum(axis=-1) > np.finfo(np.float32).eps )
            n_c += (y_dec == y_true).sum()
            n_t += y_dec.size
            acc = n_c/n_t
            print('acc[{:4d}] = {:5.03f}'.format(int(n_t), acc))
            
        return acc, n_t



if __name__ == '__main__':
    if os.name == 'nt':
        ds_path = r'G:\Downloads\timit'
    else:
        ds_path = '../data_sets/TIMIT'

        

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
                'n_fft':None,
                'window':'hann',
                'mfcc_normaleze_first_mfcc':True,
                'calc_mfcc_derivate':False,
                
                'mfcc_norm_factor':0.01,
                'M_dB_norm_factor':0.01,
                'P_dB_norm_factor':0.01,
                
                
                'mean_abs_amp_norm':0.003,
                'clip_output':True}


    
    model_cfg_d = {'model_name':'phn_model',
                   
                   'input_shape':(ds_cfg_d['n_timesteps'], (2 if ds_cfg_d['calc_mfcc_derivate'] else 1)*ds_cfg_d['n_mfcc']),
                   'n_output':61,
                   
                   'embed_size':64, # Para la prenet. Se puede aumentar la dimension. None (usa la cantidad n_mfcc)
                   'num_conv_banks':4,
                   'num_highwaynet_blocks':2,
                   'dropout_rate':0.2,
                   'is_training':True,
                   'use_Cudnn':False, # sys.platform!='win32', # Solo cuda para linux
                   'use_lstm':False,

                   'model_name':'encoder',

                   'learning_rate':5.0e-3,
                   'decay':1.0e-2,
                   
                   'beta1':0.9,
                   'beta2':0.999,
                   'epsilon':1e-8,
                   
                   'ds_trn_filter_d':{'ds_type':'TRAIN'},
                   'ds_val_filter_d':{'ds_type':'TEST'},
                   'ds_tst_filter_d':{'ds_type':'TEST'},
                   'randomize_samples':True,
                   
                   'n_epochs':        99999,
                   'batch_size':       32,
                   'val_batch_size':   128,
                   'save_each_n_epochs':3,

                   'log_dir':'./enc_stats_dir',
                   'model_path':'./enc_6_ckpt'}


    model_cfg_d = load_cfg_d('./hp/encoder_cfg_d.json')
##    save_cfg_d(ds_cfg_d,    './hp/ds_cfg_d.json')
##    save_cfg_d(model_cfg_d, './hp/encoder_cfg_d.json')

    if True:
        timit = TIMIT(ds_cfg_d)
    else:
        timit = None
        np.random.seed(0)
        x = np.random.random( (32, model_cfg_d['input_shape'][0], model_cfg_d['input_shape'][1]) )
        y = np.random.random( (32, model_cfg_d['input_shape'][0], model_cfg_d['n_output']) )
        for i in range(y.shape[0]):
            y[i,np.arange(y[i].shape[0]), np.argmax(y[i], axis=-1)] = 1.0

        y[y != 1] = 0.0

                      
    model = encoder_spec_phn(model_cfg_d, timit)
##    model.exec_train_step(x, y)
##    model.train()

    model.restore()
    model.eval_acc(timit.window_sampler(ds_filter_d={'ds_type':'TEST'}) )


    
    
    

