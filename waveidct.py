from sigprocess import pre_emphasis,audio2frame,spectrum_power
import numpy as np
from scipy.fftpack.realtransforms import idct,dct
from scipy.fftpack import fft,ifft


def IDCTcoefficient(sig, samplerate=16000, win_length=0.025, win_step=0.01, pre_emphasis_coeff=0.97):
    '''
    计算初始IDCT系数
    :param sig:
    :param samplerate:
    :param win_length:
    :param win_step:
    :param pre_emphasis_coeff:
    :return:
    '''

    #预处理
    signal = pre_emphasis(sig, pre_emphasis_coeff)
    #分帧
    frames = audio2frame(signal, win_length * samplerate, win_step * samplerate)  # 得到帧数组
    #加窗
    frames *= np.hamming(int(round(win_length * samplerate)))  # 加窗
    #DCT
    dctfeat = dct(frames, type=2, axis=1, norm='ortho')  # 进行离散余弦变换
    #为0无法取对数
    #dctfeat[dctfeat < 1e-30] = 1e-30
    dctfeatnoz = np.where(dctfeat == 0, np.finfo(float).eps, dctfeat) #对为0的地方调整为eps，这样便于进行对数处理
    dctfeatnoz=np.absolute(dctfeatnoz)
    #取对数
    logfeat = np.log(dctfeatnoz)
    del dctfeatnoz
    #取IDCT
    idctfeat = idct(logfeat, type=2, axis=1, norm='ortho')  # 进行反离散余弦变换

    return idctfeat


def FFTcoefficient(sig, samplerate=16000, win_length=0.025, win_step=0.01, pre_emphasis_coeff=0.97,NFFT=512):
    '''
    计算初始IDCT系数
    :param sig:
    :param samplerate:
    :param win_length:
    :param win_step:
    :param pre_emphasis_coeff:
    :return:
    '''

    #预处理
    signal = pre_emphasis(sig, pre_emphasis_coeff)
    #分帧
    frames = audio2frame(signal, win_length * samplerate, win_step * samplerate)  # 得到帧数组
    #加窗
    frames *= np.hamming(int(round(win_length * samplerate)))  # 加窗
    #FFT
    fftfeat =  spectrum_power(frames,NFFT)  # 进行快速傅里叶变换 得到幅值系数
    feat = np.where(fftfeat == 0, np.finfo(float).eps, fftfeat)

    #TODO  滤波

    feat = np.log(fftfeat)

    feat = dct(feat, type=2, axis=1, norm='ortho')



    return feat