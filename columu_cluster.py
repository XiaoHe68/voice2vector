import numpy as np
from utils import exist_dir,recover_rout
import pickle

def _merge_recover_rout(feat,foldername,onefile_prefix,savewhenk,endfluster=1):
	m,n=feat.shape
	routpath=foldername + '/' + onefile_prefix + '_kpath_' + str(endfluster) + '.txt'
	for k in savewhenk:
		tmpfeat=_merge_feat(feat,routpath,k)
		np.save(foldername + '/' + onefile_prefix + '_'+str(k), tmpfeat)
		stxt = foldername + '/'+onefile_prefix+'_routpath_k_' + str(k) + '.txt'
		rout = recover_rout(k, routpath)
		np.savetxt(stxt, rout, '%d')


def _merge_feat(feat,kpath,k):
	m,n=feat.shape
	routpath=kpath
	assert k>0
	tmpfeat = np.zeros((m, k))
	rout = recover_rout(k, routpath)
	for idx, col in enumerate(rout):
		tmpfeat[:, idx] = np.sum(feat[:, col[0]:col[1]], axis=1)
	return tmpfeat


def _merge_list(feat, totaln, foldername, savewhenk, disfunc,onefile_prefix,endfluster=1):
	if savewhenk and len(savewhenk)>0:
		k = min(endfluster,min(savewhenk))
	else:
		k=endfluster

	epoch = totaln - k
	routcluster = []
	import time
	dst = [0] * (totaln - 1)
	sumcols = range(totaln - 1)

	for l in range(epoch):
		st = time.time()
		print('Current column number：', totaln - l - 1, ',剩余合并次数：', epoch - l)
		for j in sumcols:
			dst[j] = disfunc(feat[j], feat[j + 1])
		minidx = np.argmin(dst)
		# print('minidx:',minidx)
		routcluster.append(minidx)

		feat[minidx] = feat[minidx] + feat[minidx + 1]
		if minidx == 0:
			sumcols = [0]
		if minidx == len(dst) - 1:
			sumcols = [minidx - 1]
		if minidx != 0 and minidx != len(dst) - 1:
			sumcols = [minidx - 1, minidx]
		del feat[minidx + 1]
		del dst[minidx]
		ed = time.time() - st
		print('current epoch use tiem：', ed, 'seconds')

		ck = totaln - l - 1
		if ck in savewhenk:
			exist_dir(foldername)
			print('save files when k：', ck)
			np.save(foldername + '/'+onefile_prefix + '_'+ str(ck), np.array(feat).T)

	routpath = foldername + '/'+onefile_prefix+'_kpath_' + str(k) + '.txt'
	np.savetxt(routpath, routcluster, '%d')
	print('kpath creat success!', routpath)

	for r in savewhenk:
		stxt = foldername + '/'+onefile_prefix+'_routpath_k_' + str(r) + '.txt'
		np.savetxt(stxt, recover_rout(r, routpath), '%d')


def merge_from_onefile(featpath, foldername, savewhenk, disfunc,onefile_prefix,kpath=False,endfluster=1):

	feat = np.load(featpath)
	if kpath:
		_merge_recover_rout(feat,foldername,onefile_prefix,savewhenk,endfluster)
	else:
		m,totaln=feat.shape
		print('merge_from_onefile,feat shape', feat.shape)
		tezheng = [feat[:,f].flatten() for f in range(totaln)]
		_merge_list(tezheng, totaln, foldername, savewhenk, disfunc=disfunc, onefile_prefix=onefile_prefix,
		                    endfluster=endfluster)


def merge_from_pickle(featfolder, savewhenk, disfunc,onefile_prefix,endfluster=1):
	try:
		with open(featfolder + '/' + onefile_prefix + 'pic', 'rb') as fw:
			feat=pickle.load(fw)
	except Exception as err:
		print('加载文件错误')
		raise err
	totaln=len(feat)
	_merge_list(feat, totaln, featfolder, savewhenk, disfunc=disfunc, onefile_prefix=onefile_prefix,
	                    endfluster=endfluster)


