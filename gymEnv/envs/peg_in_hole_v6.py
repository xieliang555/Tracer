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
from utils.util import read_force_2d_map, cosine_similarity



class PegInHole(gym.Env):
	def __init__(self, dense_map, beta=0.02, seed=1):
		super(PegInHole, self).__init__()
		'''
		Args:
			dense_map: [441,6]
		'''
		np.random.seed(seed)
		self.dense_map = dense_map
		self.beta = beta


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



	def reset(self):
		self.d = False
		self.r = 0
		self.stepCount = 0
		self.maxStep = 20

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
		self.dense_index = dx*21+dy

		# print('\n')
		# print('reset')

		# get sparse map
		self.ft = copy.copy(self.dense_map[self.dense_index, :])

		# print('before', self.ft)
		# self.ft += np.random.normal(loc=self.ft, scale=[1,1,5,0.1,0.1,0.02])*self.beta
		# print('after', self.ft)

		return self.ft




	def step(self, a):
		# a: [dx(mm), dy(mm), dyaw(deg)] 
		self.stepCount += 1
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
			# self.ft += np.random.normal(loc=self.ft, scale=[1,1,5,0.1,0.1,0.02])*self.beta
			# print('after', self.ft)
		else:
			self.d = True
			# print('test2')

		if self.stepCount > self.maxStep:
			self.d = True
			# print('test3')
			
		# !!!!!!!!!
		if abs(dx-10) <=3 and abs(dy-10)<=3:
			self.d = True
			self.r = 1-self.stepCount/100
			# print('test4', self.r)

		return self.ft, self.r, self.d


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

	map_path = '~/models/demos/map/0717'
	shape = 'square'
	pooling = True
	alpha = 0.2
	seed = 0
	beta = 0.5
	dense_map = read_force_2d_map(map_path, shape, pooling=pooling, alpha=alpha, seed=seed)

	env = gym.make('gymEnv:peg-in-hole-v4', dense_map=dense_map, beta=beta, seed=seed)
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






