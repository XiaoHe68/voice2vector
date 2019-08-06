from sklearn.decomposition import PCA
from feature_extract import voice_feature
from utils import exist_dir
import os
import numpy
from scipy import signal
import random

vf=voice_feature()
vf.p_vad=True
vf.p_win_step=10
vf.p_win_len=30
vf.p_nfft=512

def get_PCA(featDic,dicinfo):
    minlen=numpy.inf

    for k in dicinfo:
        if dicinfo[2]<minlen:
            minlen=dicinfo[2]

    # newDataList = numpy.zeros((len(featDic.keys())*minlen,column))
    newDataDic = dict()
    # idx=0
    for pf in featDic:
        pca = PCA(n_components=minlen, copy=False)  # 直接指定降维维度
        # newData=pca.fit_transform(featDic[pf].T).T
        # newDataList[idx*minlen:(idx+1)*minlen,:]=pca.fit_transform(featDic[pf].T).T
        # idx+=1

        newDataDic[pf]=pca.fit_transform(featDic[pf].T).T
        print(pf,pca.explained_variance_)

    return newDataDic

def get_input_feat(filename):
    return vf.f_spectrum_power(filename)


# 产生以人名为key，mfcc为值的Dic
def CreatFeatureDic(folder,generatefunc,extension='.wav'):
    '''

    :param folder: 数据源路径
    :param extension: 文件格 .wav
    :param myway:
    :param decmean:
    :param win:
    :param winstep:
    :param NFFT:
    :return:
    '''
    dicinfo={}
    dicfeature= {}
    for root, dirs, files in os.walk(folder):
        files = sorted(files)
        for f in files:
            if f[-4:] != extension:
                continue
            froot = root + '/' + f
            # pn为说话者名字或标识，fn为句子唯一标识
            pn, fn = generatefunc(froot)
            tmpfeature = get_input_feat(froot)
            if dicinfo != None:
                if pn not in dicinfo:
                    dicinfo[pn] = []
                    dicinfo[pn].append([])
                    dicinfo[pn].append([])
                    dicinfo[pn].append(0)
                dicinfo[pn][0].append(fn)
                dicinfo[pn][1].append([dicinfo[pn][2], dicinfo[pn][2] + tmpfeature.shape[0]])
                dicinfo[pn][2] += tmpfeature.shape[0]

            if pn not in dicfeature:
                dicfeature[pn] = tmpfeature
                print(root, pn)
            else:
                dicfeature[pn] = numpy.concatenate((dicfeature[pn], tmpfeature))

    return get_PCA(dicfeature,dicinfo)

def conv2(data,kenerl):
    # numpy.convolve(a,v,)
    return signal.convolve2d(data, kenerl, boundary='symm', mode='same')

def creatN1(folder, generatefunc):

    N2=10
    listKerel=[]
    for i in range(N2):
        listKerel.append(2 *numpy.random.randn(3, 3) - 1)
    iptdic=CreatFeatureDic(folder, generatefunc, extension='.wav')
