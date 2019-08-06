import os
import numpy
import columu_cluster
import utils
import distance
import json
from utils import exist_dir


# 产生以人名为key，特征为值的Dic
def CreatFeatureDic(folder,feat_extract,generatefunc,ndim,kpath=None,extension='.wav',
                    saveFeature=False,saveFolder=None,featname=None):
    exist_dir(saveFolder)
    assert os.path.isdir(saveFolder),'输出路径不存在！'

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
            if ndim:
                tmpfeature = get_feat(feat_extract,froot,ndim=ndim,kpath=kpath)
            else:
                tmpfeature = get_feat(feat_extract, froot,kpath=kpath)

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

    if ndim is None:
        onefile_prefix = featname
    else:
        onefile_prefix=featname+'_'+str(ndim)

    with open(saveFolder + '/' + onefile_prefix + '_p_s_position.json', 'w') as fw:
        json.dump(dicinfo, fw)

    if saveFeature:
        if saveFolder is None:
            print('保存路径有误！')
        else:
            for k in dicfeature:
                numpy.save(saveFolder + '/' + k, dicfeature[k])
    return dicfeature


def Compress2OneFiel(feat_extract_info,db_indfo,featname,ndim=None,kpath=None):

    wavefolder=db_indfo.wavefolder
    outfolder=db_indfo.outfolder
    generatefunc=db_indfo.generatefunc
    extension=db_indfo.extension

    if ndim is None:
        onefile_prefix = featname
    else:
        onefile_prefix=featname+'_'+str(ndim)
    
    dicfeat=CreatFeatureDic(wavefolder,featname=onefile_prefix,feat_extract=feat_extract_info,extension=extension,generatefunc=generatefunc
                        ,saveFolder=outfolder,ndim=ndim,kpath=kpath)
    utils.feature2oneFile(dicfeat,outfolder,onefile_prefix=onefile_prefix)


def creat_kpath(idct_onefeatfile,outfolder,disfunc,onefile_prefix,savek):
    import time
    st=time.time()
    columu_cluster.merge_from_onefile(idct_onefeatfile, foldername=outfolder, savewhenk=savek, disfunc=
    disfunc,onefile_prefix=onefile_prefix)
    print('creat kpath for '+onefile_prefix+' success! use time ',time.time()-st)


def feat_x(db,x,savek):
    if x=='evector':
        creat_kpath(db.pre_data, db.outfolder, distance.distEclud, x, savek)
    elif x=='cvector':
        creat_kpath(db.pre_data, db.outfolder, distance.cos_sim_package, x, savek)
    elif x=='rvector':
        creat_kpath(db.pre_data, db.outfolder, distance.corrcoef, x, savek)
    else:
        raise ValueError('x shoud be one of evector,cvector,rvector')


def creat_xvector(pre_data,outfolder,disfunc,onefile_prefix,savek,pnumber):
    import time
    onefile_prefix=onefile_prefix+'_'+str(pnumber)
    st=time.time()
    feat=numpy.load(pre_data)

    file_pfx='/'.join(pre_data.split('/')[:-1])+'/'+'.'.join(pre_data.split('/')[-1].split('.')[0:-1])
    print(file_pfx)
    postionfile=file_pfx+ '_p_position.json'
    postion_s_file=file_pfx+ '_p_s_position.json'

    if not os.path.isfile(postionfile):
        raise FileNotFoundError(postionfile+',Not Found!')

    with open(postionfile, 'r') as fr:
        p_position = json.load(fr)
    with open(postion_s_file, 'r') as fr:
        p_s_position = json.load(fr)

    person = sorted(p_position.items(), key=lambda x: x[1][0])
    print(person[pnumber-1][1][1])
    columu_cluster.merge_from_matrx(feat[:person[pnumber-1][1][1],:], foldername=outfolder, savewhenk=savek, disfunc=
    disfunc,onefile_prefix=onefile_prefix)

    with open(outfolder + '/' + onefile_prefix + '_p_position.json', 'w') as fw:
        dic=dict(person[:pnumber])
        json.dump(dic, fw)

    with open(outfolder + '/' + onefile_prefix + '_p_s_position.json', 'w') as fw:
        dicinfo=dict()
        for k in dic:
            dicinfo[k]=p_s_position[k]
        json.dump(dicinfo, fw)

    print('creat '+onefile_prefix+' success! use time ',time.time()-st)


def feat_xvector(pre_data,outfolder,x,savek,pnumber):
    if x=='evector':
        creat_xvector(pre_data, outfolder, distance.distEclud, x, savek,pnumber)
    elif x=='cvector':
        creat_xvector(pre_data, outfolder, distance.cos_sim_package, x, savek,pnumber)
    elif x=='rvector':
        creat_xvector(pre_data, outfolder, distance.corrcoef, x, savek,pnumber)
    else:
        raise ValueError('x shoud be one of evector,cvector,rvector')

def get_feat_xvector(fe,wavefile,outputfile,x,ndim):

    fe.set_feat_type('idct')
    feat=fe.current_func(wavefile)
    disfunc=None
    if x=='evector':
        disfunc=distance.distEclud
    elif x=='cvector':
        disfunc = distance.cos_sim_package
    elif x=='rvector':
        disfunc = distance.corrcoef
    else:
        raise ValueError('x shoud be one of evector,cvector,rvector')

    outfolder='/'.join(outputfile.split('/')[:-1])
    onefile_prefix='.'.join(outputfile.split('/')[-1].split('.')[:-1])

    columu_cluster.merge_from_matrx(feat, foldername=outfolder, savewhenk=[ndim], disfunc=
    disfunc, onefile_prefix=onefile_prefix)



def get_feat(fe,wavefile,**kwargs):
    if 'wavefile' in fe.current_func.__code__.co_varnames:
        if fe.current_func.__code__.co_argcount==2:
            return fe.current_func(wavefile)
        else:
            return fe.current_func(wavefile, kwargs)
    else:
        raise ValueError('传入参数不正确')


def save_feat(feat,savepath,fmt):
    if fmt=='txt':
        numpy.savetxt(savepath,feat,'%.3f',delimiter='\t')
    else:
        numpy.save(savepath,feat)


