import numpy as np

def vad_power_fast(sig,winstep,threshold,winlen,step=None,wavefile=''):
    winstep=int(winstep)
    winlen=int(winlen)

    minwin=winlen+winstep
    if step is None:
        step=int(winstep)
    sig_power=np.square(sig)
    l=len(sig)
    assert l>winstep,'frame too short!'
    st=0
    #idxlst=[]
    idx=[]
    p_start=False
    st_flag=0
    while st<l:
        power=np.sum(sig_power[st:st+winstep])/winstep
        lst=min(l-1,st+winstep)
        if power<threshold and sig_power[st]<threshold and sig_power[lst]<threshold:
            if p_start:
                p_start=False
                if st+step-st_flag<minwin:
                    continue
                number= (st + step-st_flag)//winstep
                lst=min(l-1,st_flag+winstep*number)
                #idxlst.append([st_flag, lst])
                idx.extend(list(range(st_flag,lst)))
        else:
            if not p_start:
                p_start=True
                st_flag = st
        st+=step
    if p_start:
        if l- st_flag > minwin:
            number = (l - st_flag) // winstep
            #idxlst.append([st_flag, st_flag+winlen*number])
            lst = min(l, st_flag+winstep*number)
            idx.extend(list(range(st_flag, lst)))

    return sig[idx]

def vad_power_fast_no(sig,winstep,threshold,winlen,step=None,wavefile=''):
    minwin=winlen+winstep
    if step is None:
        step=winstep
    sig_power=np.square(sig)
    l=len(sig)
    assert l>winstep,'frame too short!'
    st=0
    #idxlst=[]
    idx=[]
    p_start=False
    st_flag=0
    while st<l:
        power=np.sum(sig_power[st:st+winstep])/winstep
        lst=min(l-1,st+winstep)
        if power<threshold and sig_power[st]<threshold and sig_power[lst]<threshold:
            if p_start:
                p_start=False
                if st+step-st_flag<minwin:
                    continue
                lst=min(l-1,st+step)
                #idxlst.append([st_flag, lst])
                idx.extend(list(range(st_flag,lst)))
        else:
            if not p_start:
                p_start=True
                st_flag = st
        st+=step
    if p_start:
        if l- st_flag > minwin:
            #idxlst.append([st_flag, st_flag+winlen*number])
            lst = min(l, st)
            idx.extend(list(range(st_flag, lst)))

    return sig[idx]


def vad_power_v(sig,winlen_vad,threshold,step_vad=None):
    if step_vad is None:
        step=winlen_vad
    sig_power=np.square(sig)
    l=len(sig)
    assert l>winlen_vad,'frame too short!'
    st=0
    idx=[]
    power=np.inf
    while st<l:
        power=np.sum(sig_power[st:st+winlen_vad])/winlen_vad
        lst=min(l-1,st+winlen_vad)
        if power<threshold and sig_power[st]<threshold and sig_power[lst]<threshold:
            idx.extend([st,st+step_vad])
        st+=step_vad

    if power < threshold:
        lst = min(l - 1, st+step_vad)
        idx.extend([lst, l])
    if len(idx)>=4:
        idx=idx[1:-1]
    else:
        idx=[0,l]

    return idx


def sample_vad(sig,winlen,step,threshold,winlen_vad,step_vad):
    vad_power_v(sig)

    # sig[idx]=0
    # return sig




if __name__=='__main__':
    import scipy.io.wavfile as wav
    import matplotlib.pyplot as plt
    wl=-1
    wavefile = '/Volumes/WDBlueTM/timit_feature/timit_wav/train/dr2/mefg0/sa1.wav'
    wavefile='/Volumes/WDBlueTM/timit_feature/timit_wav/test/dr1/faks0/sa1.wav'
    (rate, sig) = wav.read(wavefile)
    tmpsig = np.copy(sig[:wl]).astype(int)
    sig_vad=vad_power_fast(tmpsig,winstep=int(0.015*rate),threshold=4000
                                        ,winlen=int(rate*0.03))
    plt.subplot(2,1,1)
    plt.plot(range(len(sig[:wl])),(sig[:wl]))

    # for x in lst:
    #     plt.axvline(x)
    plt.subplot(2, 1, 2)
    plt.plot(range(len(sig_vad)), (sig_vad))
    print('压缩率：',1-(len(sig[:wl])-len(sig_vad))/len(sig[:wl]))
    plt.show()

#0.9153412006157003
#0.8921138654317624