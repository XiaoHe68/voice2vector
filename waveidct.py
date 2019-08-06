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



if __name__=="__main__":
    import scipy.io.wavfile as wav
    import matplotlib.pyplot as plt
    import vad


    pointnum=3200

    (rate,sig)=wav.read("/Volumes/WDBlueTM/timit_feature/timit_wav/test/dr1/faks0/sa1.wav")
    sig=vad.vad_power_fast(sig,0.01*rate,4000,0.02*rate)

    # plt.subplot(4, 1, 1)
    # plt.title("sig")
    # plt.plot(range(pointnum), sig[:pointnum])



    feat = dct(np.array(sig[:pointnum]).reshape(1,-1), type=2, axis=1, norm='ortho')

    # plt.subplot(4, 1, 2)
    # plt.title("DCT")
    # plt.plot(range(feat.shape[1]), feat[0])


    # feat=np.power(feat,2)
    feat = abs(feat)
    feat = np.where(feat == 0, np.finfo(float).eps, feat)
    feat=np.log(feat)

    # plt.subplot(4, 1, 3)
    # plt.title("Log")
    # # plt.plot(range(175),feat[0][25:200])
    # plt.plot(range(feat.shape[1]), feat[0])

    feat=idct(feat)

    # plt.subplot(4,1,4)
    plt.title("IDCT")
    # plt.plot(range(175),feat[0][25:200])
    plt.plot(range(feat.shape[1]), abs(feat[0]))

    # plt.subplot(4, 1, 3)
    # plt.plot(range(feat.shape[1]), feat[0]-np.array( sig[:pointnum]))
    plt.show()



    # x = np.linspace(-0, 1, 10000)
    # a = np.sin(2*np.pi*x)#+np.cos(2*np.pi*x) +np.sin(8*np.pi*x)+np.cos(2*np.pi*x)+np.sin(np.pi*x)+np.sin(6*np.pi*x)
    # plt.subplot(4, 1, 1)
    # plt.plot(x,a)
    #
    # feat = dct(a.reshape(1,-1), type=2, axis=1, norm='ortho')
    # plt.subplot(4, 1, 2)
    # plt.plot(x, feat[0])
    # feat=abs(feat)
    # feat = np.log(feat)
    # feat=idct(feat)
    # plt.subplot(4, 1, 3)
    # plt.plot(x, feat[0])
    # plt.show()
    # lst=[50,100,150,200,250,300,350,400,450,500,600,700,800,900,1000]
    # for ii in lst:
    #     print(16000/ii)





