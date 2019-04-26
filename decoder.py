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
from aux_func import *

class decoder_specs:
    def __init__(self, cfg_d={}, ds=None, encoder=None):
        self.cfg_d = cfg_d
        self.ds    = ds

        self.i_global_step = 0
        self.i_epoch       = 0
        self.summary_v     = []

        self.encoder = encoder

        # Creo Modelo
        self._build_model( reuse=None )

        
        # Armo la funcion de costo
        self._build_loss(reuse=None)

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
        if self.encoder is None:
            tf_sess_config = tf.ConfigProto()
            tf_sess_config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=tf_sess_config)
        else:
            self.sess = self.encoder.sess
            
        return None


        
    def _build_model(self, reuse=None):
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


        with tf.variable_scope(self.cfg_d['model_name'], reuse=reuse):
            if self.encoder is None:
                self.inputs = tf.placeholder(tf.float32, (None,)+tuple(self.cfg_d['input_shape']), name='inputs')
                inputs = self.inputs
            else:
                self.inputs = self.encoder.get_input()

                enc_o = self.encoder.get_outputs()
                inputs = enc_o.y_pred  # Usamos softmax que tiene menos informacion del hablante.
                 
                assert self.encoder.y_logits.shape.as_list()[1:] == list(self.cfg_d['input_shape']), 'ERROR, input_shape no coincide con la dimensión de salida del encoder.'
                

            with tf.variable_scope('step1', reuse=reuse): # self.cfg_d['steps_v'][0]['']  self.cfg_d['']
                step_d = self.cfg_d['steps_v'][0]
                
                if step_d['embed_size'] is None:
                    embed_size = self.cfg_d['input_shape'][-1]
                else:
                    embed_size = step_d['embed_size']

                # Encoder pre-net
                prenet_out = prenet(inputs=inputs,
                                    num_units=None,
                                    embed_size=embed_size,
                                    dropout_rate=self.cfg_d['dropout_rate'],
                                    is_training=self.cfg_d['is_training'],
                                    scope="prenet",
                                    reuse=reuse) # (N, T_x, E/2)
                
                # Encoder CBHG 
                CBHG_out = CBHG(inputs=prenet_out,
                                embed_size=embed_size,
                                num_conv_banks=step_d['num_conv_banks'],
                                num_highwaynet_blocks=step_d['num_highwaynet_blocks'],
                                dropout_rate=self.cfg_d['dropout_rate'],
                                is_training=self.cfg_d['is_training'],
                                scope="CBHG",
                                use_CudnnGRU=self.cfg_d['use_CudnnGRU'],
                                reuse=reuse) # (N, T_x, E)



                
                self.y_mel      = tf.layers.dense(CBHG_out, step_d['n_output'], activation=None, name="y_logits")                       # (N, T_x, n_mel)
                self.target_mel = tf.placeholder(tf.float32, (None, self.cfg_d['input_shape'][0], step_d['n_output']), name='target' )  # (N, T_x, n_mel)
            

            with tf.variable_scope('step2', reuse=reuse):
                step_d = self.cfg_d['steps_v'][1]

                if step_d['embed_size'] is None:
                    embed_size = CBHG_out.shape.as_list()[-1]
                else:
                    embed_size = step_d['embed_size']


                if False:
                    inputs_step2 = tf.concat([CBHG_out, self.inputs], axis=-1)
                else:
                    inputs_step2 = CBHG_out
                
                # Encoder pre-net
                prenet_out = prenet(inputs=inputs_step2,
                                    num_units=None,
                                    embed_size=embed_size,
                                    dropout_rate=self.cfg_d['dropout_rate'],
                                    is_training=self.cfg_d['is_training'],
                                    scope="prenet",
                                    reuse=reuse) # (N, T_x, E/2)
                
                # Encoder CBHG 
                CBHG_out = CBHG(inputs=prenet_out,
                                embed_size=embed_size,
                                num_conv_banks=step_d['num_conv_banks'],
                                num_highwaynet_blocks=step_d['num_highwaynet_blocks'],
                                dropout_rate=self.cfg_d['dropout_rate'],
                                is_training=self.cfg_d['is_training'],
                                scope="CBHG",
                                use_CudnnGRU=self.cfg_d['use_CudnnGRU'],
                                reuse=reuse) # (N, T_x, E)

                
                self.y_stft      = tf.layers.dense(CBHG_out, step_d['n_output'], activation=None, name="y_logits")                       # (N, T_x, n_stft=n_fft//2+1)
                self.target_stft = tf.placeholder(tf.float32, (None, self.cfg_d['input_shape'][0], step_d['n_output']), name='target' )  # (N, T_x, n_stft=n_fft//2+1)
        
        return None
    



    def _build_loss(self, reuse=None):
        with tf.variable_scope('loss', reuse=reuse):
            self.mel_loss  = self.cfg_d['mel_loss_weight'] * tf.reduce_mean( tf.squared_difference( self.y_mel, self.target_mel ),   name='mel_loss' )
 
            self.stft_loss = self.cfg_d['stft_loss_weight'] * tf.reduce_mean( tf.squared_difference( self.y_stft, self.target_stft ), name='stft_loss' )


            self.loss = tf.log(self.mel_loss) + tf.log(self.stft_loss)

        
        self.summary_v += [tf.summary.scalar('dec_metric/mel_loss', self.mel_loss),
                           tf.summary.scalar('dec_metric/stft_loss', self.stft_loss),
                           tf.summary.scalar('dec_metric/loss', self.loss)]
        return None


    
    def _build_optimizer(self, reuse=None):
        with tf.variable_scope('dec_opt',reuse=reuse):
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
            
##        elif self.mode == 'test':
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


    def exec_train_step(self, inputs, target_mel, target_stft):
        
        ret = self.sess.run([self.mel_loss,
                             self.stft_loss,
                             self.loss,
                             self.global_step,
                             self.train_step,
                             self.summary_merged],
                            
                            feed_dict={self.inputs:inputs,
                                       self.target_mel:target_mel,
                                       self.target_stft:target_stft})
        
        mel_loss, stft_loss, loss, global_step, train_step, summary_merged = ret

        self.i_global_step = global_step
        self.trn_writer.add_summary(summary_merged, self.i_global_step)
        
        return (mel_loss, stft_loss, loss, global_step, train_step)

    

    def exec_calc_metrics(self, inputs, target_mel, target_stft, summary_mode='validation'):

        ret = self.sess.run([self.mel_loss,
                             self.stft_loss,
                             self.loss,
                             self.global_step,
                             self.summary_merged], feed_dict={self.inputs:inputs,
                                                              self.target_mel:target_mel,
                                                              self.target_stft:target_stft})


        
        mel_loss, stft_loss, loss, global_step, summary_merged = ret


        self.i_global_step = global_step
        if summary_mode == 'train':
            self.val_writer.add_summary(summary_merged, self.i_global_step)
        elif summary_mode == 'validation':
            self.val_writer.add_summary(summary_merged,  self.i_global_step)
        elif summary_mode == 'test':
            self.tst_writer.add_summary(summary_merged,  self.i_global_step)            
        else:
            raise Exception(' - ERROR, summary_mode={} not implemented'.format(summary_mode))
            
        return mel_loss, stft_loss, loss

    
    def train(self):
        self.cfg_d['n_samples_trn'] = self.ds.get_n_windows(self.cfg_d['ds_prop_val'])[0]

        self.cfg_d['n_steps_epoch_trn'] = self.cfg_d['n_samples_trn']//self.cfg_d['batch_size']

        self.sampler_trn = self.ds.spec_window_sampler(batch_size=self.cfg_d['batch_size'],
                                                       n_epochs=99999999,
                                                       randomize_samples=self.cfg_d['randomize_samples'],
                                                       sample_trn=True,
                                                       prop_val=self.cfg_d['ds_prop_val'])


        self.sampler_val = self.ds.spec_window_sampler(batch_size=self.cfg_d['batch_size'],
                                                       n_epochs=99999999,
                                                       randomize_samples=self.cfg_d['randomize_samples'],
                                                       sample_trn=False,
                                                       prop_val=self.cfg_d['ds_prop_val'])

        self.iter_val  = iter(self.sampler_val)
        
        print(' Starting Training ...')
        print(' n_samples_trn:    ', self.cfg_d['n_samples_trn'])
        print(' n_steps_epoch_trn:', self.cfg_d['n_steps_epoch_trn'])
        print(' batch_size:       ', self.cfg_d['batch_size'])
        print(' n_epochs:         ', self.cfg_d['n_epochs'])
        input('Press --ENTER--')


        # Refresh: i_epoch lr 
        self.i_epoch, self.lr = self.sess.run( [self.i_epoch_tf, self.lr_decay_op] )
        
        for mfcc_trn, mel_trn, stft_trn in self.sampler_trn:
            mel_loss_trn, stft_loss_trn, loss_trn, global_step, train_step = self.exec_train_step(mfcc_trn, mel_trn, stft_trn)
                        
            print(' - i_epoch={}   global_step={}   mel_loss_trn={:6.3f}  stft_loss_trn={:6.3f}  loss_trn={:6.3f}'.format(self.i_epoch, global_step, mel_loss_trn, stft_loss_trn, loss_trn) )

            if (global_step/self.cfg_d['n_steps_epoch_trn']) % self.cfg_d['save_each_n_epochs'] == 0:
                # new epoch
                print(' Saving, epoch={} ...'.format(self.i_epoch))
                self.save()
                mfcc_val, mel_val, stft_val = next(self.iter_val)
                mel_loss_val, stft_loss_val, loss_val = self.exec_calc_metrics(mfcc_val, mel_val, stft_val)
                print(' - i_epoch={}   global_step={}   mel_loss_val={:6.3f}   stft_loss_val={:6.3f}   loss_val={:6.3f}'.format(self.i_epoch, int(global_step), mel_loss_val, stft_loss_val, loss_val) )

                


            if global_step % self.cfg_d['n_steps_epoch_trn'] == 0:
                _, self.i_epoch, self.lr = self.sess.run( [self.i_epoch_inc_op, self.i_epoch_tf, self.lr_decay_op] )
                
                if self.i_epoch >= self.cfg_d['n_epochs']:
                    break


        print(' End of Training !!!')
        return None


    def predict(self, x, batch_size=32):
        y_mel_v  = []
        y_stft_v = []
        
        for i_s in range(0, x.shape[0], batch_size):
            x_batch = x[i_s:min(i_s+batch_size, x.shape[0])]
            y_mel, y_stft = self.sess.run([self.y_mel, self.y_stft], {self.inputs:x_batch} )
            
            y_mel_v.append(y_mel)
            y_stft_v.append(y_stft)

        predict_nt = namedtuple('predict', 'y_mel y_stft')
        
        ret_nt = predict_nt(np.concatenate(y_mel_v, axis=0),
                            np.concatenate(y_stft_v, axis=0))
        return ret_nt


    def run(self, var, feed_dict={}):
        return self.sess.run(var, feed_dict)

    def get_input_shape(self):
        return tuple(self.inputs.shape.as_list()[1:])
    
    def eval_acc(self, ds_iterator, n_batchs=100):
        n_c = 0
        n_t = 0
        for i_batch in range(n_batchs):
            mfcc_batch, phn_v_batch = next(ds_iterator)
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
##    if os.name == 'nt':
##        ds_path = r'G:\Downloads\timit'
##    else:
##        ds_path = '/media/sergio/EVO970/UNIR/TFM/code/data_sets/TIMIT'

    timit_ds_cfg_d = load_cfg_d('./hp/ds_enc_cfg_d.json')


    if os.name == 'nt':
        ds_path = r'G:\Downloads\TRG\L. Frank Baum/The Wonderful Wizard of Oz'
    else:
        ds_path = '/media/sergio/EVO970/UNIR/TFM/code/data_sets/TRG/L. Frank Baum/The Wonderful Wizard of Oz'

        
    target_ds_cfg_d = {'ds_path':ds_path,
                       'sample_rate':timit_ds_cfg_d['sample_rate'],  #Frecuencia de muestreo los archivos de audio Hz
                       'exclude_files_with':['Oz-01', 'Oz-25'],
                       'ds_cache_name':'AH_target_cache.pickle',
                       'verbose':True,
                       'spec_cache_name':'spec_cache.h5py',

                       'ds_norm':(0.0, 1.0),
                       'remake_samples_cache':False,
                       'random_seed':         None,
                        
                       'pre_emphasis': timit_ds_cfg_d['pre_emphasis'],
                       
                       'hop_length_ms': timit_ds_cfg_d['hop_length_ms'], # 2.5ms = 40c | 5.0ms = 80c (@ 16kHz)
                       'win_length_ms': timit_ds_cfg_d['win_length_ms'], # 25.0ms = 400c (@ 16kHz)
                       'n_timesteps':   timit_ds_cfg_d['n_timesteps'], # 800ts*(win_length_ms=2.5ms)= 2000ms  Cantidad de hop_length_ms en una ventana de prediccion.
                       
                       'n_mels': timit_ds_cfg_d['n_mels'],
                       'n_mfcc': timit_ds_cfg_d['n_mfcc'],
                       'n_fft':  timit_ds_cfg_d['n_fft'], # None usa n_fft=win_length
                        
                       'window':                    timit_ds_cfg_d['window'],
                       'mfcc_normaleze_first_mfcc': timit_ds_cfg_d['mfcc_normaleze_first_mfcc'],
                       'mfcc_norm_factor':          timit_ds_cfg_d['mfcc_norm_factor'],
                       'calc_mfcc_derivate':        timit_ds_cfg_d['calc_mfcc_derivate'],
                       'M_dB_norm_factor':          timit_ds_cfg_d['M_dB_norm_factor'],
                       'P_dB_norm_factor':          timit_ds_cfg_d['P_dB_norm_factor'],
                        
                       'mean_abs_amp_norm': timit_ds_cfg_d['mean_abs_amp_norm'],
                       'clip_output':       timit_ds_cfg_d['clip_output']}

    save_cfg_d(target_ds_cfg_d, './hp/ds_dec_cfg_d.json')
        
    enc_cfg_d = load_cfg_d('./hp/encoder_cfg_d.json')
    enc_cfg_d['is_training'] = False


    n_stft = (target_ds_cfg_d['n_fft'] or target_ds_cfg_d.get('win_length') or (int(target_ds_cfg_d['win_length_ms']*target_ds_cfg_d['sample_rate']/1000.0))) // 2 + 1
    dec_cfg_d = {'model_name':'decoder',
                   
                 'input_shape':(enc_cfg_d['input_shape'][0], enc_cfg_d['n_output']),
                 
                 'steps_v':[{'embed_size':256, # Para la prenet. Se puede aumentar la dimension. None (usa la cantidad n_mfcc)
                             'num_conv_banks':8,
                             'num_highwaynet_blocks':8,
                             'n_output':target_ds_cfg_d['n_mels']},
                            
                            {'embed_size':256, # Para la prenet. Se puede aumentar la dimension. None (usa la cantidad n_mfcc)
                             'num_conv_banks':8,
                             'num_highwaynet_blocks':8,
                             'n_output': n_stft}],
                   
                  'dropout_rate':0.2,
                  'is_training':True,
                  'use_CudnnGRU':True, # sys.platform!='win32', # Solo cuda para linux

                 'learning_rate':1.0e-3,
                 'decay':5.0e-3,
                   
                 'beta1':0.9,
                 'beta2':0.999,
                 'epsilon':1e-8,

                 'mel_loss_weight': 400,
                 'stft_loss_weight':400,
                   
                 'ds_prop_val':0.3,
                 'randomize_samples':True,
                   
                 'n_epochs':        99999,
                 'batch_size':        128,
                 'val_batch_size':    128,
                 'save_each_n_epochs':  2,

                 'log_dir':   './dec_stats_dir',
                 'model_path':'./dec_ckpt'}

    save_cfg_d(dec_cfg_d, './hp/decoder_cfg_d.json')

    
    trg_spk = TARGET_spk(target_ds_cfg_d)

    
    encoder = encoder_spec_phn(cfg_d=enc_cfg_d, ds=None)

##    if True:
##        encoder.restore()
##        timit   = TIMIT(timit_ds_cfg_d)
##        encoder.eval_acc(timit.window_sampler(ds_filter_d={'ds_type':'TEST'}) )

    decoder = decoder_specs(cfg_d=dec_cfg_d, ds=trg_spk, encoder=encoder)
    encoder.restore()

##    # Restauro entrenamiento pausado
##    decoder.restore()
##    # Asigno nuevo lr_decay
##    decoder.run(tf.assign(decoder.learning_rate_decay, dec_cfg_d['decay']))
    
    decoder.train()



##    if True:
##        y = np.random.random( (32,) + decoder.get_input_shape())
##        y_pred = decoder.predict(y)
##
##        mfcc, mel, stft = next(iter( trg_spk.spec_window_sampler() ))
##        
##        decoder.exec_calc_metrics(mfcc, mel, stft)
