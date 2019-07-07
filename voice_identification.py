import numpy as np
import matplotlib.pyplot as plt
import os, sys
import keras

from TIMIT_reader import TIMIT


def create_model(input_shape=(400, 80), n_output=8):
    model = keras.Sequential()

    model.add( keras.layers.Reshape(input_shape+(1,), input_shape=input_shape) )

    model.add( keras.layers.Conv2D(filters=32,  kernel_size=5, activation='relu' ) )

    model.add( keras.layers.MaxPooling2D(pool_size=2) )

    model.add( keras.layers.Conv2D(filters=64,  kernel_size=3, activation='relu' ) )

    model.add( keras.layers.MaxPooling2D(pool_size=2) )

    model.add( keras.layers.Flatten() )
    
    model.add( keras.layers.BatchNormalization() )

    model.add( keras.layers.Dense(128, activation='relu') )

    model.add( keras.layers.Dense(512, activation='relu') )
    
    model.add( keras.layers.Dense(n_output, activation='softmax') )

    opt = keras.optimizers.Adam(lr=1e-4)
    
    model.compile(opt, 'categorical_crossentropy', metrics=['categorical_accuracy'])

    model.summary()
    return model








if __name__ == '__main__':
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

    spk_id_v =['ABC0', 'ABW0', 'ADC0', 'ADD0', 'ADG0', 'AEB0', 'AEM0', 'AEO0',
               'AFM0', 'AHH0', 'AJC0', 'AJP0', 'AJW0', 'AKB0', 'AKR0', 'AKS0',
               'ALK0', 'ALR0', 'APB0', 'APV0', 'ARC0', 'ARW0', 'ASW0', 'AWF0',
               'BAR0', 'BAS0', 'BBR0', 'BCG0', 'BCG1', 'BCH0', 'BDG0', 'BEF0',
               'BGT0', 'BJK0', 'BJL0', 'BJV0', 'BLV0', 'BMA0', 'BMA1', 'BMH0',
               'BMJ0', 'BML0', 'BNS0', 'BOM0', 'BPM0', 'BSB0', 'BTH0', 'BWM0',
               'BWP0', 'CAE0', 'CAG0', 'CAJ0', 'CAL0', 'CAL1', 'CAU0', 'CCS0',
               'CDC0', 'CDD0', 'CDR0', 'CDR1', 'CEF0', 'CEG0', 'CEM0', 'CEW0',
               'CFT0', 'CHH0', 'CHL0', 'CJF0', 'CJS0', 'CKE0', 'CLK0', 'CLM0',
               'CLT0', 'CMB0', 'CMG0', 'CMH0', 'CMH1', 'CMJ0', 'CMM0', 'CMR0',
               'CPM0', 'CRC0', 'CRE0', 'CRH0', 'CRZ0', 'CSH0', 'CSS0', 'CTH0',
               'CTM0', 'CTT0', 'CTW0', 'CXM0', 'CYL0', 'DAB0', 'DAC0', 'DAC1',
               'DAC2', 'DAS0', 'DAS1', 'DAW0', 'DAW1', 'DBB0', 'DBB1', 'DBP0',
               'DCD0', 'DCM0', 'DDC0', 'DED0', 'DEF0', 'DEM0', 'DFB0', 'DHC0',
               'DHL0', 'DHS0', 'DJH0', 'DJM0', 'DKN0', 'DKS0', 'DLB0', 'DLC0',
               'DLC1', 'DLC2', 'DLD0', 'DLF0', 'DLH0', 'DLM0', 'DLR0', 'DLR1',
               'DLS0', 'DMA0', 'DML0', 'DMS0', 'DMT0', 'DMY0', 'DNC0', 'DNS0',
               'DPB0', 'DPK0', 'DPS0', 'DRB0', 'DRD0', 'DRD1', 'DRM0', 'DRW0',
               'DSC0', 'DSJ0', 'DSS0', 'DSS1', 'DTB0', 'DTD0', 'DVC0', 'DWA0',
               'DWD0', 'DWH0', 'DWK0', 'DWM0', 'DXW0', 'EAC0', 'EAL0', 'EAR0',
               'ECD0', 'EDR0', 'EDW0', 'EEH0', 'EFG0', 'EGJ0', 'EJL0', 'EJS0',
               'ELC0', 'EME0', 'ERS0', 'ESD0', 'ESG0', 'ESJ0', 'ETB0', 'EWM0',
               'EXM0', 'FER0', 'FGK0', 'FMC0', 'FRM0', 'FWK0', 'FXS0', 'FXV0',
               'GAF0', 'GAG0', 'GAK0', 'GAR0', 'GAW0', 'GCS0', 'GDP0', 'GES0',
               'GJC0', 'GJD0', 'GJF0', 'GLB0', 'GMB0', 'GMD0', 'GMM0', 'GRL0',
               'GRP0', 'GRT0', 'GRW0', 'GSH0', 'GSL0', 'GWR0', 'GWT0', 'GXP0',
               'HBS0', 'HES0', 'HEW0', 'HIT0', 'HJB0', 'HLM0', 'HMG0', 'HMR0',
               'HPG0', 'HRM0', 'HXL0', 'HXS0', 'ILB0', 'ISB0', 'JAC0', 'JAE0',
               'JAI0', 'JAR0', 'JAS0', 'JBG0', 'JBR0', 'JCS0', 'JDA0', 'JDC0',
               'JDE0', 'JDG0', 'JDH0', 'JDM0', 'JDM1', 'JDM2', 'JEB0', 'JEB1',
               'JEE0', 'JEM0', 'JEN0', 'JES0', 'JFC0', 'JFH0', 'JFR0', 'JHI0',
               'JHK0', 'JJB0', 'JJG0', 'JJJ0', 'JJM0', 'JKL0', 'JKR0', 'JLB0',
               'JLG0', 'JLG1', 'JLM0', 'JLN0', 'JLR0', 'JLS0', 'JMA0', 'JMD0',
               'JMG0', 'JMM0', 'JMP0', 'JPG0', 'JPM0', 'JPM1', 'JRA0', 'JRB0',
               'JRE0', 'JRF0', 'JRG0', 'JRH0', 'JRH1', 'JRK0', 'JRP0', 'JRP1',
               'JSA0', 'JSJ0', 'JSK0', 'JSP0', 'JSR0', 'JSW0', 'JTC0', 'JTH0',
               'JVW0', 'JWB0', 'JWB1', 'JWG0', 'JWS0', 'JWT0', 'JXA0', 'JXL0',
               'JXM0', 'JXP0', 'KAA0', 'KAG0', 'KAH0', 'KAJ0', 'KAM0', 'KCH0',
               'KCL0', 'KDB0', 'KDD0', 'KDE0', 'KDR0', 'KDT0', 'KDW0', 'KES0',
               'KFB0', 'KJL0', 'KJO0', 'KKH0', 'KLC0', 'KLC1', 'KLH0', 'KLN0',
               'KLR0', 'KLS0', 'KLS1', 'KLT0', 'KLW0', 'KMS0', 'KRG0', 'KSR0',
               'KXL0', 'LAC0', 'LAG0', 'LAS0', 'LBC0', 'LBW0', 'LEH0', 'LEL0',
               'LET0', 'LHD0', 'LIH0', 'LJA0', 'LJB0', 'LJC0', 'LJD0', 'LJG0',
               'LJH0', 'LKD0', 'LKM0', 'LLL0', 'LMA0', 'LMC0', 'LMK0', 'LNH0',
               'LNS0', 'LNT0', 'LOD0', 'LSH0', 'LTM0', 'MAA0', 'MAB0', 'MAB1',
               'MAF0', 'MAG0', 'MAH0', 'MAH1', 'MAM0', 'MAR0', 'MBG0', 'MBS0',
               'MCC0', 'MCM0', 'MDB0', 'MDB1', 'MDG0', 'MDH0', 'MDM0', 'MDM1',
               'MDM2', 'MDS0', 'MEA0', 'MEB0', 'MEM0', 'MGC0', 'MGD0', 'MGG0',
               'MGK0', 'MJB0', 'MJB1', 'MJF0', 'MJR0', 'MJU0', 'MKC0', 'MKF0',
               'MLD0', 'MLM0', 'MMH0', 'MML0', 'MPG0', 'MPM0', 'MRP0', 'MSM0',
               'MVP0', 'MWB0', 'MWH0', 'MWS0', 'MWS1', 'MXS0', 'NET0', 'NJM0',
               'NKL0', 'NLP0', 'NLS0', 'NMR0', 'NTB0', 'NTW0', 'PAB0', 'PAB1',
               'PAC0', 'PAD0', 'PAF0', 'PAM0', 'PAM1', 'PAR0', 'PAS0', 'PAZ0',
               'PCS0', 'PDF0', 'PEB0', 'PFU0', 'PGH0', 'PGL0', 'PGR0', 'PGR1',
               'PJF0', 'PKT0', 'PLB0', 'PLS0', 'PMB0', 'PMY0', 'PPC0', 'PRB0',
               'PRD0', 'PRK0', 'PRT0', 'PSW0', 'PWM0', 'RAB0', 'RAB1', 'RAI0',
               'RAM0', 'RAM1', 'RAV0', 'RBC0', 'RCG0', 'RCS0', 'RCW0', 'RCZ0',
               'RDD0', 'RDM0', 'RDS0', 'REB0', 'REE0', 'REH0', 'REH1', 'REM0',
               'RES0', 'REW0', 'REW1', 'RFK0', 'RFL0', 'RGG0', 'RGM0', 'RGS0',
               'RHL0', 'RJB0', 'RJB1', 'RJH0', 'RJM0', 'RJM1', 'RJM3', 'RJM4',
               'RJO0', 'RJR0', 'RJS0', 'RJT0', 'RKM0', 'RKO0', 'RLD0', 'RLJ0',
               'RLJ1', 'RLK0', 'RLL0', 'RLR0', 'RMB0', 'RMG0', 'RMH0', 'RML0',
               'RMS0', 'RMS1', 'RNG0', 'ROA0', 'RPC0', 'RPC1', 'RPP0', 'RRE0',
               'RRK0', 'RSO0', 'RSP0', 'RTC0', 'RTJ0', 'RTK0', 'RVG0', 'RWA0',
               'RWS0', 'RWS1', 'RXB0', 'SAG0', 'SAH0', 'SAH1', 'SAK0', 'SAS0',
               'SAT0', 'SAT1', 'SBK0', 'SCN0', 'SDB0', 'SDC0', 'SDH0', 'SDJ0',
               'SDS0', 'SEM0', 'SEM1', 'SES0', 'SFH0', 'SFH1', 'SFV0', 'SGF0',
               'SJG0', 'SJK0', 'SJK1', 'SJS0', 'SJS1', 'SJW0', 'SKC0', 'SKL0',
               'SKP0', 'SLB0', 'SLB1', 'SLS0', 'SMA0', 'SMC0', 'SMM0', 'SMR0',
               'SMS0', 'SMS1', 'SPM0', 'SRG0', 'SRH0', 'SRR0', 'SSB0', 'STF0',
               'STK0', 'SVS0', 'SXA0', 'TAA0', 'TAB0', 'TAJ0', 'TAS0', 'TAS1',
               'TAT0', 'TAT1', 'TBC0', 'TBR0', 'TBW0', 'TCS0', 'TDB0', 'TDP0',
               'TDT0', 'TEB0', 'TER0', 'THC0', 'TJG0', 'TJM0', 'TJS0', 'TJU0',
               'TKD0', 'TKP0', 'TLB0', 'TLC0', 'TLG0', 'TLH0', 'TLS0', 'TMG0',
               'TML0', 'TMN0', 'TMR0', 'TMT0', 'TPF0', 'TPG0', 'TPP0', 'TPR0',
               'TQC0', 'TRC0', 'TRR0', 'TRT0', 'TWH0', 'TWH1', 'TXS0', 'UTB0',
               'VFB0', 'VJH0', 'VKB0', 'VLO0', 'VMH0', 'VRW0', 'WAC0', 'WAD0',
               'WAR0', 'WBT0', 'WCH0', 'WDK0', 'WEM0', 'WEW0', 'WGR0', 'WJG0',
               'WRE0', 'WRP0', 'WSB0', 'WSH0', 'WVW0', 'ZMB0']

    
    ds_filter_trn_d={'split_d':{'split_key':'spk_id', 'split_props_v':(0.8,0.9), 'split_type':'trn'},
                     'spk_id':spk_id_v}

    ds_filter_val_d={'split_d':{'split_key':'spk_id', 'split_props_v':(0.8,0.9), 'split_type':'val'},
                     'spk_id':spk_id_v}

    ds_filter_tst_d={'split_d':{'split_key':'spk_id', 'split_props_v':(0.8,0.9), 'split_type':'tst'},
                     'spk_id':spk_id_v}

    
    timit.get_ds_filter(ds_filter_trn_d).sum()
    model = create_model(input_shape=(400, 201), n_output=len(spk_id_v))

##    r = next(iter(timit.speaker_spec_sampler(32, n_epochs=1, ds_filter_d=ds_filter_d)))
##    for mfcc, mel_dB, power_dB, class_oh in zip(*r):
##        print(class_oh)
##        timit.spec_show(mel_dB)

    
    
    n_epochs = 1000
    batch_size = 32

    val_flow = iter( timit.speaker_spec_sampler(batch_size, n_epochs=n_epochs, ds_filter_d=ds_filter_val_d) )

    trn_loss_v = []
    trn_acc_v  = []
    val_loss_v = []
    val_acc_v  = []

    best_val_acc = 0.0
    
    i_step = 0
    for mfcc_v, mel_dB_v, power_dB_v, class_oh_v in timit.speaker_spec_sampler(batch_size, n_epochs=n_epochs, ds_filter_d=ds_filter_trn_d):
        hist_trn_d = model.fit(power_dB_v, class_oh_v,  batch_size=batch_size, epochs=1, verbose=False).history

        val_mfcc_v, val_mel_dB_v, val_power_dB_v, val_class_oh_v = next(val_flow)
        hist_val_v = model.evaluate(val_power_dB_v, val_class_oh_v, batch_size=512, verbose=False)

        trn_loss_v.append(hist_trn_d['loss'][0])
        trn_acc_v.append( hist_trn_d['categorical_accuracy'][0])
        val_loss_v.append(hist_val_v[0])
        val_acc_v.append( hist_val_v[1])
        
        print(' - i_step={:4d}  -  trn_loss={:5.03f}  -  trn_acc={:5.03f}  -  val_loss={:5.03f}  -  val_acc={:5.03f}'.format(i_step,
                                                                                                                             trn_loss_v[-1], trn_acc_v[-1],
                                                                                                                             val_loss_v[-1], val_acc_v[-1]))

        if len(val_acc_v) > 10 and np.mean(val_acc_v[-10]) > best_val_acc:
            best_w = model.get_weights()
            best_val_acc = np.mean(val_acc_v[-10])

            print(best_val_acc, 'Salvando<<<<<<<<<<<<<<<<<<<<<')
            
        i_step += 1


    if 1:
        plt.plot(trn_acc_v, 'r-')
        plt.plot(val_acc_v, 'b-')
        plt.show()





        
            




        
