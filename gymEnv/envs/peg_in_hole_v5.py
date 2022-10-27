import gym
import os
import math
from scipy.spatial.transform import Rotation as R
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
import copy

import sys
sys.path.append('../..')
from utils.util import read_dense_map, cosine_similarity



class PegInHole(gym.Env):
	def __init__(self, dense_map, beta=0.02, seed=1):
		super(PegInHole, self).__init__()
		'''
		Args:
			dense_map: [441,3]
		'''

		np.random.seed(seed)
		self.dense_map = dense_map
		self.beta = beta
		self.pca = PCA(n_components=3)


	def projection(self, target_pose):
		target_pos = target_pose[:2,3]
		# position projection: real world -> grid map
		# [-2mm,2mm] -> [0,20]
		dx = int(np.around(5*target_pos[0]+10))
		dy = int(np.around(5*target_pos[1]+10))
		return [dx, dy]


	def transform(self, rotMat, transVec):
		matrix = np.eye(4)
		matrix[:3,:3]=rotMat
		matrix[:3,3]=transVec
		return matrix


	def visualize(self, dx=0, dy=0):
		# dx, dy: prediction index
		# print('dxy', dx, dy)

		# dense map with gt
		self.dense_index_list.append(self.dense_index)
		ax = plt.subplot(2,3,1)
		ax.set_title('Dense with gt')
		plt.imshow(self.dense_map_reduced)
		for idx, dense_index in enumerate(self.dense_index_list):
			ax.scatter(dense_index%21, dense_index//21, 50, c='w', marker="+", alpha=(idx+1)/(len(self.dense_index_list)+1))
			ax.annotate(idx, (dense_index%21, dense_index//21))

		# # sparse map
		# ax = plt.subplot(2,3,2)
		# ax.set_title('Sparse map')
		# sparse_map_mask = self.sparse_map[:,:3].astype(np.bool)
		# sparse_map = self.pca.transform(self.sparse_map)
		# sparse_map = (sparse_map-self.dense_map_min)/(self.dense_map_max-self.dense_map_min)
		# sparse_map[~sparse_map_mask] = 0
		# plt.imshow(sparse_map.reshape(21,21,3))

		# dense map with prediction
		self.pred_index_list.append((dy, dx))
		ax = plt.subplot(2,3,3)
		ax.set_title('Dense with pred')
		plt.imshow(self.dense_map_reduced)
		for idx, pred_index in enumerate(self.pred_index_list):
			ax.scatter(pred_index[0], pred_index[1], 50, c='w', marker='+', alpha=(idx+1)/(len(self.pred_index_list)+1))
			ax.annotate(idx, pred_index)



		# ax = plt.subplot(2,3,5)
		# ft = self.pca.transform(self.ft[None,:])
		# dense_map_reduced = self.pca.transform(self.dense_map)
		# cos_sim = cosine_similarity(ft, dense_map_reduced)
		# plt.imshow(cos_sim.reshape(21,21), cmap='gray')


		# ax = plt.subplot(2,3,6)
		# dense_map = (self.dense_map- self.dense_map.min())/(self.dense_map.max()-self.dense_map.min())
		# plt.imshow(dense_map.reshape(21,21,6)[:,:,:3])

		plt.show()


	# def visualize(self, dx=0, dy=0):
	# 	# dx, dy: prediction index
	# 	# print('dxy', dx, dy)

	# 	# dense map with gt
	# 	self.dense_index_list.append(self.dense_index)
	# 	ax = plt.subplot(2,3,1)
	# 	ax.set_title('Dense with gt')
	# 	plt.imshow(self.dense_map.reshape(21,21,3))
	# 	for idx, dense_index in enumerate(self.dense_index_list):
	# 		ax.scatter(dense_index%21, dense_index//21, 10, c='w', marker="*", alpha=(idx+1)/(len(self.dense_index_list)+1))

	# 	# sparse map
	# 	ax = plt.subplot(2,3,2)
	# 	ax.set_title('Sparse map')
	# 	plt.imshow(self.sparse_map.reshape(21,21,3))

	# 	# dense map with prediction
	# 	ax = plt.subplot(2,3,3)
	# 	ax.set_title('Dense with pred')
	# 	plt.imshow(self.dense_map.reshape(21,21,3))
	# 	ax.scatter(dy, dx, 10, c='w', marker='*')

	# 	# cosine similarity
	# 	ax = plt.subplot(2,3,4)
	# 	ax.set_title('cosine similarity')
	# 	cos_sim = cosine_similarity(self.ft, self.dense_map)
	# 	plt.imshow(cos_sim.reshape(21,21), cmap='gray')

	# 	plt.show()




	def reset(self):
		# for debug
		dense_map_reduced = self.pca.fit_transform(self.dense_map)
		self.dense_map_max = dense_map_reduced.max()
		self.dense_map_min = dense_map_reduced.min()
		dense_map_reduced = (dense_map_reduced-self.dense_map_min)/(self.dense_map_max-self.dense_map_min)
		self.dense_map = copy.copy(dense_map_reduced)
		self.dense_map_reduced = dense_map_reduced.reshape(21,21,3)
		self.dense_index_list = []
		self.pred_index_list = []

		self.d = False
		self.r = 0
		self.stepCount = 0
		self.maxStep = 20
		self.sparse_map = np.zeros((441,3), dtype=np.float32)
		self.reset_dxy = [0,0]

		self.oriMat = self.transform(np.eye(3), [0,0,0])
		self.offset_translate = np.random.uniform(-2,2,2)
		self.offset_rotation = 0
		# print('reset offset', self.offset_translate)

		pose = self.transform(
			R.from_euler('z', self.offset_rotation, degrees=True).as_matrix(), 
			[self.offset_translate[0],self.offset_translate[1],0])
		init_pose = self.oriMat.dot(pose)

		# get dense index
		dx, dy = self.projection(init_pose)
		self.reset_dxy = [dx, dy]
		self.dense_index = dx*21+dy

		# get sparse index
		self.sparse_index = 10*21+10

		# print('\n')
		# print('reset')

		# get sparse map
		self.ft = copy.copy(self.dense_map[self.dense_index, :])
		# print('before', self.ft)
		self.ft += np.random.normal(loc=self.ft, scale=[1,1,1])*self.beta
		# print('after', self.ft)
		self.sparse_map[self.sparse_index] = self.ft

		return self.ft, self.dense_map, self.sparse_map, self.dense_index, self.sparse_index




	def step(self, a):
		# a: [dx(mm), dy(mm), dyaw(deg)] 
		self.stepCount += 1
		# print(self.stepCount)
		# print('stepCount:', self.stepCount)

		self.offset_translate += np.array(a[0:2])
		self.offset_rotation += 0

		# print('a', a)
		# print('offset', self.offset_translate)
		pose = self.transform(
			R.from_euler('z', self.offset_rotation, degrees=True).as_matrix(),
			[self.offset_translate[0],self.offset_translate[1],0])
		target_pose = self.oriMat.dot(pose)

		dx, dy = self.projection(target_pose)
		if dx>=0 and dx<=20 and dy>=0 and dy<=20:
			self.dense_index = dx*21+dy
			self.ft =copy.copy(self.dense_map[self.dense_index, :])
			# print('before', self.ft)
			self.ft += np.random.normal(loc=self.ft, scale=[1,1,1])*self.beta
			# print('after', self.ft)		

			dx_sparse = dx - self.reset_dxy[0]+10
			dy_sparse = dy - self.reset_dxy[1]+10
			if dx_sparse>=0 and dx_sparse<=20 and dy_sparse>=0 and dy_sparse<=20:
				self.sparse_index = dx_sparse*21+dy_sparse
				self.sparse_map[self.sparse_index, :] = self.ft
			else:
				self.d = True
				# print('test1')
		else:
			self.d = True
			# print('test2')


		if self.stepCount > self.maxStep:
			self.d = True
			# print('test3')
			
		# !!!!!!!!!
		# if abs(self.offset_translate[0]) <= 0.6 and abs(self.offset_translate[1]) <= 0.6:
		# 	self.d = True
		# 	self.r = 1-self.stepCount/100
			# print('test4')

		return self.ft, self.dense_map, self.sparse_map, self.dense_index, self.sparse_index, self.r, self.d


	def close(self):
		pass



if __name__ == '__main__':

	'''
	验证：
	1. state
		1.1: dense map 和 sparse map 空间位置，大小是否对应
		1.2: dense index 和 sparse index
	2: act,reward,done

	'''

	map_path = '/Users/xieliang/models/demos/map/0717'
	shape = 'square'
	pooling = True
	alpha = 2
	seed = 0
	beta = 0.1
	dense_map = read_dense_map(map_path, shape, pooling=pooling, alpha=alpha, seed=seed)

	env = gym.make('gymEnv:peg-in-hole-v5', dense_map=dense_map, beta=beta, seed=seed)
	env.reset()
	env.visualize()

	for i in range(100):
		# a = np.array([i,i,i])*0.1
		a = np.random.uniform(-2,2,size=(2))*0.3
		_, _, _,_, _, r, d = env.step(a)
		print('r, d', r, d)
		env.visualize()

		if d:
			env.reset()






