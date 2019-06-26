import numpy as np
import json
import pickle

def exist_dir(path_folder):
	import os
	if os.path.isfile(path_folder):
		path_folder='/'.join(path_folder.split('/')[:-1])
	if path_folder[-1]!='/':
		path_folder+='/'
	if not os.path.isdir(path_folder):
		os.makedirs(path_folder)


def feature2memory(dicfeat):
	# 用于保存每个人的位置
	pfeatdic = {}
	totaln = 0
	row = 0
	for p in dicfeat:
		pfeat = dicfeat[p]
		m, n = pfeat.shape
		totaln = n
		pfeatdic[p] = [row, row + m]
		row += m
	feat = np.zeros((row, totaln))
	for kk in pfeatdic:
		feat[pfeatdic[kk][0]:pfeatdic[kk][1], :] = dicfeat[kk][:, :]
	return feat,pfeatdic

def feature2oneFile(dicfeat,outfolder,onefile_prefix):
	feat, pfeatdic=feature2memory(dicfeat)

	assert onefile_prefix!='' and onefile_prefix is not None

	try :
		exist_dir(outfolder)
		with open(outfolder + '/'+onefile_prefix+'_p_position.json', 'w') as fw:
			json.dump(pfeatdic, fw)
		print('saving feat file...')
		np.save(outfolder + '/'+onefile_prefix, feat)
		print('feat shape:',feat.shape)
	except Exception as err:
		print(err)


def feature2pickle(dicfeat,outfolder,onefile_prefix):
	feat, pfeatdic = feature2memory(dicfeat)
	m,n=feat.shape
	try :
		exist_dir(outfolder)
		with open(outfolder + '/'+onefile_prefix+'_p_position.json', 'w') as fw:
			json.dump(pfeatdic, fw)
		with open(outfolder + '/' + onefile_prefix + 'pic', 'wb') as fw:
			pickle.dump([feat[:, f].flatten() for f in range(n)], fw)

	except Exception as err:
		print(err)


def recover_rout(targetk,routpath):
	'''
	得到本路径时传入的最小K值（即最小列col后的值）
	:param targetk: 要求的目标K列时聚类方式
	:param routpath:得到本路径时传入的最小K值（即最小列col后的值）
	:return: 返回sum顺序
	'''
	mink=int(routpath.split('/')[-1].split('.')[0].split('_')[-1])
	assert targetk>=mink
	lst=np.loadtxt(routpath,dtype='int')
	partlen=np.ones((targetk,))
	rlst=list(reversed(lst))
	idxlst=[]
	for i in rlst[targetk-mink:]:
		lsum=0
		idx=0
		if i==0:
			partlen[idx] += 1
		while lsum<i:
			lsum+=partlen[idx]
			if lsum==i:
				partlen[idx+1] += 1
			if lsum>i:
				partlen[idx] += 1
			idx += 1

	total=np.sum(partlen)
	rpartlen=list(reversed(partlen))
	for i in rpartlen:
		idxlst.append([int(total-i),int(total)])
		total -= i

	return list(reversed(idxlst))


def hz2mel(hz):
	'''把频率hz转化为梅尔频率
	参数说明：
	hz:频率
	'''
	return 2595 * np.log10(1 + hz / 700.0)


def mel2hz(mel):
	'''把梅尔频率转化为hz
	参数说明：
	mel:梅尔频率
	'''
	return 700 * (10 ** (mel / 2595.0) - 1)