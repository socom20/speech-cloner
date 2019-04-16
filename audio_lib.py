import numpy as np
import matplotlib.pyplot as plt
import os, sys

import librosa
import librosa.display

import pickle


def calc_PHN_target(y, phn_v, phn_conv_d, hop_length=40, win_length=400):
    n_samples = int(y.shape[0] / hop_length) + 1

    half_n_fft = win_length//2 #Esto es porque usamos por defecto un modo center en la STFT (hacemos un pad de n_fft//2 en cada extremo de la señal)

    
    target_v = []
    i_phn = 0
    for i_s in range(n_samples):
        i_win_s = i_s*hop_length - half_n_fft
        i_win_e = i_s*hop_length + win_length - half_n_fft
        
        while phn_v[i_phn][1] <= i_win_s and i_phn+1 < len(phn_v):
            i_phn += 1

        delta_phn_a = min(phn_v[i_phn][1], i_win_e) - max(phn_v[i_phn][0], i_win_s)

        if i_phn+1 < len(phn_v):
            delta_phn_b = min(phn_v[i_phn+1][1], i_win_e) - max(phn_v[i_phn+1][0], i_win_s)

            if delta_phn_a >= delta_phn_b:
                target_v.append(phn_conv_d[phn_v[i_phn][2]])
            else:
##                print('es b', delta_phn_a, delta_phn_b)
                target_v.append(phn_conv_d[phn_v[i_phn+1][2]])
        else:
            target_v.append(phn_conv_d[phn_v[i_phn][2]])

##        if i_s > 2 and (target_v[-1] != target_v[-2]).any():
##            print(delta_phn_a, delta_phn_b)
            
    target_v = np.array(target_v, dtype=np.int32)
##    assert len(target_v) == n_samples, 'ERROR len(target_v) != n_samples, no se pudo hacer bien et target'
    
    return target_v


        
def calc_MFCC_input(y, sr=16000, pre_emphasis=0.97, hop_length=40, win_length=400, n_mels=128, n_mfcc=40, mfcc_normaleze_first_mfcc=True, mfcc_norm_factor=0.01, mean_abs_amp_norm=0.003, clip_output=True):
    """ Calcula MFCC de la onda de entrada para usarlo como input.
        A la señal de entrada y se le aplica primero un filtro de pre enfasis.
        Posteriormente se calcula el espectrograma MFCC
        
        y            : Señal de entrada
        sr           : Frecuencia de muestreo de la señal de entrada
        pre_emphasis : Parametro de pre enfasis. Si pre_emphasis==0.0, no se calucla este filtro.
        hop_length   : salto temperoal entre frames del espectro de salida
        win_length   : Ancho de la ventana usado para calcular la FFT
        n_mels       : Cantidad de intervalos Mel utilizados para el calculo del espectrograma Mel
        n_mfcc       : Cantidad de intervalos para la DCT utilizados para el calculo del espectrograma MFCC
        mfcc_normaleze_first_mfcc : Normaliza el primer coeficiente cepstral restando la componente cte inicial
        mean_abs_amp_norm : escala la salida MFCC con este factor.
        clip_output  : El MFCC de salida estará entre -1.0 y 1.0

        return : MFCC con shape (n_steps, n_mfcc)
        """

    if mean_abs_amp_norm != 1.0:
        y = (mean_abs_amp_norm/np.abs(y).mean()) * y
        
    
    if pre_emphasis != 0.0:
        y_preem = np.append(y[0], y[1:] - pre_emphasis * y[:-1])
    else:
        y_preem = y

    n_fft = win_length

##    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
##    # Calculando Parcialmente
##    # Calculamos la STFT
##    F = librosa.core.stft(y_preem,
##                          n_fft=n_fft,
##                          hop_length=hop_length,
##                          win_length=win_length,
##                          window='hann', # Aplica filtro de Hanning antes de calcular la FFT
##                          center=True,
##                          pad_mode='reflect')  # Salida shape = (1 + n_fft/2, t)
##
##    # Dejamos solo el modulo de la STFT
##    F = np.abs(F)
##    
##    # Calculamos la potencia
##    P = F ** 2 #/ n_fft
##
##    # Calculo los filtros mel para componer el espectrograma Mel
##    M = librosa.filters.mel(sr,
##                            n_fft,
##                            n_mels,
##                            fmin=0.0, fmax=None,
##                            htk=False,
##                            norm=1) # 1 divide the triangular mel weights by the width of the mel band (area normalization)
##
##    # Calculo el espectrograma Mel de la señal de entrada:
##    M_spec = M @ P
##
##    # Paso el espectrograma mel a frecuencias logaritmicas (amplitudes a dB)
##    M_spec_dB = librosa.core.amplitude_to_db(M_spec)
##
##    
##    # Calculo de los filtros para componer la DCT
##    D = librosa.filters.dct(n_mfcc, n_mels)
##
##    # Finalmente Calculo los MFCC 
##    MFCC = D @ M_spec_dB
##    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Usando solamente la libreria librosa:
    M_spec    = librosa.feature.melspectrogram(y_preem,
                                               sr,
                                               S=None,
                                               n_fft=n_fft,
                                               hop_length=hop_length,
                                               power=2.0,
                                               n_mels=n_mels)
    
    M_spec_dB = librosa.core.amplitude_to_db(M_spec)
    MFCC      = librosa.feature.mfcc(y=y_preem,
                                     sr=sr,
                                     S=M_spec_dB,
                                     n_mfcc=n_mfcc,
                                     dct_type=2,
                                     norm='ortho')
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #



    
    # Traspongo para que el primer axis sea el tiempo
    MFCC      = MFCC.T
    M_spec    = M_spec.T
    M_spec_dB = M_spec_dB.T
    
    if mfcc_normaleze_first_mfcc:
        MFCC[:,0] -= MFCC[0,0]

    if mfcc_norm_factor != 1.0:
        MFCC = mfcc_norm_factor * MFCC

    if clip_output:
        MFCC = np.clip(MFCC, -1.0, 1.0)
    
    return MFCC.astype(np.float32)




if __name__ == '__main__':
    
    import sounddevice as sd
    
    if os.name == 'nt':
        ds_path = r'G:\Downloads\timit'
    else:
        ds_path = '/media/sergio/EVO970/UNIR/TFM/code/data_sets/TIMIT'


        
    y, sr = librosa.load(ds_path + '/TEST/DR2/FDRD1/SA1.WAV')
##    y, sr = librosa.load(ds_path + '/TRAIN/DR7/FLET0/SX277.WAV')

    
    MFCC = calc_MFCC_input(y)

##    librosa.display.specshow(MFCC.T, sr=sr, x_axis='time', cmap='viridis')
##    plt.show()

    for i in range(MFCC.shape[-1]):
        plt.plot(MFCC[:,i])

    plt.show()
    
##    sd.play(y, sr, blocking=True)
##    sd.play(y_preem, sr, blocking=True)

    
    
