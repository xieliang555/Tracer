import os 
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA



class CustomDataset(Dataset):
	def __init__(self, dense_map, sparse_map, dense_index, sparse_index, ep_len):
		'''
		Args:
			dense_map: [N,441,6]
			sparse_map: [N,441,6]
			dense_index: [N]
			sparse_index: [N]
			ep_len:[N]
		'''
		self.dense_map = dense_map
		self.sparse_map = sparse_map
		self.dense_index = dense_index
		self.sparse_index = sparse_index
		self.ep_len = ep_len


	def __len__(self):
		return len(self.dense_index)

	def __getitem__(self, idx):
		dense_map = torch.tensor(self.dense_map[idx], dtype=torch.float32)
		sparse_map = torch.tensor(self.sparse_map[idx], dtype=torch.float32)
		sparse_index = torch.tensor(self.sparse_index[idx], dtype=torch.long)
		label = torch.tensor(self.dense_index[idx], dtype=torch.long)
		ep_len = torch.tensor(self.ep_len[idx], dtype=torch.long)
		return dense_map, sparse_map, sparse_index, label, ep_len



def vis_dense_map(dense_map):
    dense_map = dense_map.reshape(21,21,6)
    # print(dense_map.shape)
    i = 0
    plt.subplot(2,3,1)
    grid_map = dense_map[:,:,i+0]
    plt.imshow((grid_map-grid_map.min())/(grid_map.max()-grid_map.min()))
    plt.subplot(2,3,2)
    grid_map = dense_map[:,:,i+1]
    plt.imshow((grid_map-grid_map.min())/(grid_map.max()-grid_map.min()))
    plt.subplot(2,3,3)
    grid_map = dense_map[:,:,i+2]
    plt.imshow((grid_map-grid_map.min())/(grid_map.max()-grid_map.min()))
    plt.subplot(2,3,4)
    grid_map = dense_map[:,:,i+3]
    plt.imshow((grid_map-grid_map.min())/(grid_map.max()-grid_map.min()))
    plt.subplot(2,3,5)
    grid_map = dense_map[:,:,i+4]
    plt.imshow((grid_map-grid_map.min())/(grid_map.max()-grid_map.min()))
    plt.subplot(2,3,6)
    grid_map = dense_map[:,:,i+5]
    plt.imshow((grid_map-grid_map.min())/(grid_map.max()-grid_map.min()))
    plt.show()



def read_force_2d_map(map_path, shape, pooling=False, alpha=0, seed=0):
	fx = pd.read_excel(os.path.join(map_path, 'fx.xlsx'), sheet_name=shape).values
	fy = pd.read_excel(os.path.join(map_path, 'fy.xlsx'), sheet_name=shape).values
	fz = pd.read_excel(os.path.join(map_path, 'fz.xlsx'), sheet_name=shape).values
	tx = pd.read_excel(os.path.join(map_path, 'tx.xlsx'), sheet_name=shape).values
	ty = pd.read_excel(os.path.join(map_path, 'ty.xlsx'), sheet_name=shape).values
	tz = pd.read_excel(os.path.join(map_path, 'tz.xlsx'), sheet_name=shape).values
	dense_map = np.stack((fx,fy,fz,tx,ty,tz), -1)

	# pooling the grid map
	if pooling:
		poolLayer = nn.AvgPool2d(kernel_size=3,stride=1,padding=1)
		dense_map = torch.as_tensor(dense_map, dtype=torch.float32)
		for i in range(6):
			dense_map[:,:,i] = poolLayer(
				dense_map[:,:,i].unsqueeze(0).unsqueeze(1)).squeeze(0).squeeze(0)
		dense_map = dense_map.numpy()

	# adding noise
	np.random.seed(seed)
	dense_map += np.random.normal(loc=dense_map, scale=[1,1,5,0.1,0.1,0.02])*alpha

	# vis_dense_map(dense_map)
	
	dense_map = dense_map.reshape(441,6)

	# z-score norm the grid map
	#grid_map = grid_map.reshape(-1,6)
	#mean = np.mean(grid_map.reshape(-1,6), axis=1, keepdims=True)
	#std = np.std(grid_map.reshape(-1,6), axis=1, keepdims=True)
	#grid_map = (grid_map-mean) / std
	#grid_map = grid_map.reshape(21,21,6)

	return dense_map



def read_force_map_30d(map_path, shape, pooling=False, alpha=0, seed=0):
	'''
		dense_map_30d: [441, 30]
	'''
	dense_map_30d = []
	for i in range(5):
		fx = pd.read_excel(os.path.join(map_path, str(i)+'-'+'fx.xlsx'), sheet_name=shape).values
		fy = pd.read_excel(os.path.join(map_path, str(i)+'-'+'fy.xlsx'), sheet_name=shape).values
		fz = pd.read_excel(os.path.join(map_path, str(i)+'-'+'fz.xlsx'), sheet_name=shape).values
		tx = pd.read_excel(os.path.join(map_path, str(i)+'-'+'tx.xlsx'), sheet_name=shape).values
		ty = pd.read_excel(os.path.join(map_path, str(i)+'-'+'ty.xlsx'), sheet_name=shape).values
		tz = pd.read_excel(os.path.join(map_path, str(i)+'-'+'tz.xlsx'), sheet_name=shape).values
		dense_map = np.stack((fx,fy,fz,tx,ty,tz), -1)

		# pooling the grid map
		if pooling:
			poolLayer = nn.AvgPool2d(kernel_size=3,stride=1,padding=1)
			dense_map = torch.as_tensor(dense_map, dtype=torch.float32)
			for i in range(6):
				dense_map[:,:,i] = poolLayer(
					dense_map[:,:,i].unsqueeze(0).unsqueeze(1)).squeeze(0).squeeze(0)
			dense_map = dense_map.numpy()

		# adding noise
		np.random.seed(seed)
		dense_map += np.random.normal(loc=dense_map, scale=[1,1,5,0.1,0.1,0.02])*alpha
		
		dense_map = dense_map.reshape(441,6)
		dense_map_30d.append(dense_map)

	return np.concatenate(dense_map_30d, -1)



def read_dense_map(map_path, shape, pooling=False, alpha=0, seed=0):
	fx = pd.read_excel(os.path.join(map_path, 'fx.xlsx'), sheet_name=shape).values
	fy = pd.read_excel(os.path.join(map_path, 'fy.xlsx'), sheet_name=shape).values
	fz = pd.read_excel(os.path.join(map_path, 'fz.xlsx'), sheet_name=shape).values
	tx = pd.read_excel(os.path.join(map_path, 'tx.xlsx'), sheet_name=shape).values
	ty = pd.read_excel(os.path.join(map_path, 'ty.xlsx'), sheet_name=shape).values
	tz = pd.read_excel(os.path.join(map_path, 'tz.xlsx'), sheet_name=shape).values
	dense_map = np.stack((fx,fy,fz,tx,ty,tz), -1)

	# pooling the grid map
	if pooling:
		poolLayer = nn.AvgPool2d(kernel_size=3,stride=1,padding=1)
		dense_map = torch.as_tensor(dense_map, dtype=torch.float32)
		for i in range(6):
			dense_map[:,:,i] = poolLayer(
				dense_map[:,:,i].unsqueeze(0).unsqueeze(1)).squeeze(0).squeeze(0)
		dense_map = dense_map.numpy()

	# dimension reduced
	dense_map = dense_map.reshape(441,6)
	pca = PCA(n_components=3)
	dense_map_reduced = pca.fit_transform(dense_map)

	# adding noise
	np.random.seed(seed)
	dense_map_reduced += np.random.normal(loc=dense_map_reduced, scale=[1,1,1])*alpha
	dense_map_noise = (dense_map_reduced-dense_map_reduced.min())/(dense_map_reduced.max()-dense_map_reduced.min())

	return dense_map_noise




def cosine_similarity(ft, grid_map):
	cos = nn.CosineSimilarity(dim=-1)
	x1 = torch.tensor(grid_map, dtype=torch.float32)
	x2 = torch.tensor(ft, dtype=torch.float32)
	output = cos(x1, x2).view(-1)

	# mean norm
	# output = (output-output.mean()) / output.std()
	output = (output-output.min()) / (output.max()-output.min())

	return output



if __name__ == '__main__':
	# evaluate dense map
	# map_path = '/home/xieliang/models/demos/map/0815_30d'
	map_path = '/home/xieliang/models/demos/map/0717'
	shape = 'square'
	pooling = True
	alpha = 2
	seed = 1
	# dense_map = read_force_map_30d(map_path, shape, pooling, alpha, seed=seed)
	dense_map = read_force_2d_map(map_path, shape, pooling, alpha, seed=seed)
	vis_dense_map(dense_map)

	# pca = PCA(n_components=3)
	# dense_map =pca.fit_transform(dense_map)
	# dense_map = (dense_map -dense_map.min()) / (dense_map.max()-dense_map.min())

	# # plt.subplot(1,2,3)
	# plt.imshow(dense_map.reshape(21,21,3))
	# plt.show()






