import scipy.io.wavfile as wav
from python_speech_features import mfcc as psf_mfcc,delta,fbank as psf_fbank
from waveidct import IDCTcoefficient as idctcoeff
import os
import columu_cluster
import numpy as np
from columu_cluster import _merge_feat
import utils
from sigprocess import pre_emphasis,audio2frame,spectrum_power
from vad import vad_power_fast,vad_power_fast_no


class voice_feature(object):
    p_win_len=0.25
    p_win_step=0.05
    p_nfft=512
    p_pre_emphasis_coeff=0.97
    p_nfilt=26
    p_vad=True

    def __readwav(self,wavefile):

        (rate, sig)=wav.read(wavefile)
        if self.p_vad:
            return (rate,vad_power_fast_no(sig,winstep=int(self.p_win_step*rate),threshold=4000
                                        ,winlen=int(rate*self.p_win_len),wavefile=wavefile))
        else:
            return (rate, sig)
        
    
    # def __init__(self):
    #     self.set_feat_type(featname)


    def set_feat_type(self,featname):
        if hasattr(self, 'f_' + featname):
            self.current_feat = featname
            self.current_func = getattr(self, 'f_' + featname)


    def f_mfcc(self,wavefile):
        (rate, sig) = self.__readwav(wavefile)
        return psf_mfcc(sig, rate, self.p_win_len, self.p_win_step, nfft=self.p_nfft, nfilt=self.p_nfilt,
                        preemph=self.p_pre_emphasis_coeff)

    def f_mfcc_delta(self,wavefile):
        (rate, sig) = self.__readwav(wavefile)
        feat = psf_mfcc(sig, rate, self.p_win_len, self.p_win_step, nfft=self.p_nfft, nfilt=self.p_nfilt,
                        preemph=self.p_pre_emphasis_coeff)
        delta_1 = delta(feat, N=1)
        return np.hstack([feat, delta_1])

    def f_mfcc_delta_delta(self,wavefile):
        (rate, sig) = self.__readwav(wavefile)
        feat = psf_mfcc(sig, rate, self.p_win_len, self.p_win_step, nfft=self.p_nfft, nfilt=self.p_nfilt,
                        preemph=self.p_pre_emphasis_coeff)
        delta_1 = delta(feat, N=1)
        delta_2 = delta(delta_1, N=1)
        return np.hstack([feat, delta_1, delta_2])

    def f_fbank(self,wavefile,kargs):
        ndim=kargs.get('ndim',self.p_nfilt)
        (rate, sig) = self.__readwav(wavefile)
        feat, energy = psf_fbank(sig, rate, self.p_win_len, self.p_win_step, nfft=self.p_nfft,
                                 preemph=self.p_pre_emphasis_coeff,nfilt=ndim)
        return feat

    def f_logfbank(self,wavefile,kargs):
        ndim = kargs.get('ndim', self.p_nfilt)
        feat=self.f_fbank(wavefile,{'ndim':ndim})
        return np.log(feat)

    def f_xvector(self,wavefile,kargs):
        kpath=kargs.get('kpath','')
        k=kargs.get('ndim',15)
        assert os.path.isfile(kpath),'kpath not exit'
        assert os.path.isfile(wavefile),'wav not exit'
        (rate, sig) = self.__readwav(wavefile)
        feat1 = idctcoeff(sig,rate,self.p_win_len,self.p_win_step,self.p_pre_emphasis_coeff)
        return _merge_feat(feat1,kpath,k)

    def f_xvector_from_feat(self,featfile,k=15,kpath=''):
        assert  os.path.isfile(kpath)
        assert os.path.isfile(featfile)
        assert k>0


    def f_idct(self,wavefile):
        assert os.path.isfile(wavefile)
        (rate, sig) = self.__readwav(wavefile)
        return idctcoeff(sig, rate, self.p_win_len, self.p_win_step, self.p_pre_emphasis_coeff)


    def f_spectrum_power(self,wavefile):
        assert os.path.isfile(wavefile)
        (rate, sig) = self.__readwav(wavefile)
        high_freq = rate / 2  # 计算音频样本的最大频率
        signal = pre_emphasis(sig, self.p_pre_emphasis_coeff)  # 对原始信号进行预加重处理
        frames = audio2frame(signal,self.p_win_len* rate,self.p_win_step* rate)  # 得到帧数组
        frames *= np.hamming(int(round(self.p_win_len * rate)))  # 加窗
        spec_power = spectrum_power(frames, self.p_nfft)  # 得到每一帧FFT以后的能量谱
        return spec_power


    def f_hdcc(self,wavefile):
        (rate, sig) = self.__readwav(wavefile)
        feat1 = idctcoeff(sig, rate, self.p_win_len, self.p_win_step, self.p_pre_emphasis_coeff)
        row, clm = feat1.shape
        hzrange = 1000.0
        spl = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000]
        melspl = [utils.hz2mel(xx) for xx in spl]
        splhzhigh = utils.hz2mel(hzrange)
        xishu = [x * clm / splhzhigh for x in melspl]
        feat = np.zeros((row, len(spl) - 1))
        for i in range(len(xishu) - 1):
            st = int(xishu[i])
            ed = int(xishu[i + 1])
            bst = feat1[:, st:ed]
            liclist1 = np.sum(bst, axis=1)
            if len(liclist1) == 0:
                continue
            feat[:, i] = liclist1
        return feat


    def f_hdcc_from_feat(self,idctcoeff_feat):
        row, clm = idctcoeff_feat.shape
        hzrange = 1000.0
        spl = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000]
        melspl = [utils.hz2mel(xx) for xx in spl]
        splhzhigh = utils.hz2mel(hzrange)
        xishu = [x * clm / splhzhigh for x in melspl]
        feat = np.zeros((row, len(spl) - 1))
        for i in range(len(xishu) - 1):
            st = int(xishu[i])
            ed = int(xishu[i + 1])
            bst = idctcoeff_feat[:, st:ed]
            liclist1 = np.sum(bst, axis=1)
            if len(liclist1) == 0:
                continue
            feat[:, i] = liclist1
        return feat




if __name__=='__main__':
    wavefile='/Volumes/WDBlueTM/timit_feature/timit_wav/test/dr1/faks0/sa1.wav'
    (rate, sig) = self.__readwav(wavefile)
    idctcoeff(sig, rate,0.025, 0.005)
    