import os
import numpy
import columu_cluster
import utils
import distance
from utils import exist_dir


# 产生以人名为key，mfcc为值的Dic
def CreatFeatureDic(folder,feat_extract,generatefunc,ndim,kpath=None,extension='.wav',saveFeature=False,saveFolder=None):
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
		onefile_prefix = feat_extract.current_feat
	else:
		onefile_prefix=feat_extract.current_feat+'_'+str(ndim)

	with open(saveFolder + '/' + onefile_prefix + '_p_s_position.json', 'w') as fw:
		import json
		json.dump(dicinfo, fw)

	if saveFeature:
		if saveFolder is None:
			print('保存路径有误！')
		else:
			for k in dicfeature:
				numpy.save(saveFolder + '/' + k, dicfeature[k])
	return dicfeature


def Compress2OneFiel(feat_extract_info,db_indfo,ndim=None,kpath=None):

	wavefolder=db_indfo.wavefolder
	outfolder=db_indfo.outfolder
	generatefunc=db_indfo.generatefunc
	extension=db_indfo.extension

	if ndim is None:
		onefile_prefix = feat_extract_info.current_feat
	else:
		onefile_prefix=feat_extract_info.current_feat+'_'+str(ndim)
	
	dicfeat=CreatFeatureDic(wavefolder,feat_extract=feat_extract_info,extension=extension,generatefunc=generatefunc
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
		creat_kpath(db.idctcoefffile, db.outfolder, distance.distEclud, x, savek)
	elif x=='cvector':
		creat_kpath(db.idctcoefffile, db.outfolder, distance.cos_sim_package, x, savek)
	elif x=='rvector':
		creat_kpath(db.idctcoefffile, db.outfolder, distance.corrcoef, x, savek)
	else:
		raise ValueError('x shoud be one of evector,cvector,rvector')
	#提取x-vector 前必须先生成kpath


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


