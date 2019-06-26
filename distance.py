import numpy as np
from scipy import spatial


def distEclud(vecA, vecB):
	"""
	计算两向量的欧氏距离

	Args:
		vecA: 向量A
		vecB: 向量B
	Returns:
		欧式距离
	"""
	return np.sqrt(np.sum(np.power(vecA - vecB, 2)))


def dist_power(vecA ,vecB):
	return np.sum(np.power(vecA - vecB, 2))


def cos_sim_package(vector_a, vector_b):
	return spatial.distance.cosine(vector_a, vector_b)


def corrcoef(vecA, vecB):
	# vecA=np.array(vecA)
	# vecB=np.array(vecB)
	n = len(vecA)
	# 求和
	sum1 = np.sum(vecA)
	sum2 = np.sum(vecB)
	# 求乘积之和
	sum_pow_AB=np.sum(np.multiply(vecA, vecB))
	# 求平方和
	sum_pow1=np.sum(np.power(vecA,2))
	sum_pow2 = np.sum(np.power(vecB, 2))
	num = sum_pow_AB - (float(sum1) * float(sum2) / n)
	# 计算皮尔逊相关系数
	den = np.sqrt((sum_pow1 - float(sum1 ** 2) / n) * (sum_pow2 - float(sum2 ** 2) / n))
	if den==0:
		print('ERR')
		return 0

	return 1-(num / den)