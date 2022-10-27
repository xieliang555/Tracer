import gym
import pybullet as p
import pybullet_data
import os
import math
from scipy.spatial.transform import Rotation as R
import numpy as np
import matplotlib.pyplot as plt




class PegInHole(gym.Env):
	def __init__(self, isGUI, holeType, seed=1):
		super(PegInHole, self).__init__()
		if isGUI:
			p.connect(p.GUI)
		else:
			p.connect(p.DIRECT)
		p.setAdditionalSearchPath(pybullet_data.getDataPath())
		
		self.base = np.array([1,1,0])
		self.table = p.loadURDF('table/table.urdf', basePosition=self.base+[0,0,0],
			useFixedBase=1)

		self.panda_peg = p.loadURDF(os.path.join(os.path.dirname(
			os.path.realpath(__file__)),"urdf/"+holeType+"/robot-peg/peg.urdf"), 
			baseOrientation=p.getQuaternionFromEuler([0,0,math.pi]),
			basePosition=self.base+[0.5,0.,0.625], globalScaling=1, useFixedBase=1)

		# !!!!!!!!!!!!1
		self.hole = p.loadURDF(os.path.join(os.path.dirname(
			os.path.realpath(__file__)), "urdf/"+holeType+"/base/base.urdf"), 
			baseOrientation=p.getQuaternionFromEuler([0,0,0]),
			basePosition=self.base+[0,0,0.625+0.066], globalScaling=1.1, useFixedBase=1) 

		self.pandaNumDofs = 7
		self.ftJointIndex = 7
		self.endEffectorIndex = 9

		# p.setGravity(9.8,0,0)

		# !!!!!
		# dx, dy
		self.action_space = gym.spaces.Box(low=np.float32(-2.0),high=np.float32(2.0), shape=(2,), dtype=np.float32)

		p.enableJointForceTorqueSensor(self.panda_peg, self.ftJointIndex, enableSensor=True)
		p.resetDebugVisualizerCamera(cameraDistance=0.1, cameraYaw=30, 
			cameraPitch=-40, cameraTargetPosition=self.base+[0, 0, 0.625+0.066])

		np.random.seed(seed)

		# debug table coordinates
		# tablePos, tableOrn = p.getBasePositionAndOrientation(self.hole)
		# tablePos += np.array([0,0,0.625])



	def robot_init(self):
		self.jointPositions=[
			0.327, 0.369, -0.293, -2.383, 0.261, 2.726, 2.17]
		index = 0
		for j in range(p.getNumJoints(self.panda_peg)):
			p.changeDynamics(self.panda_peg, j, linearDamping=0, angularDamping=0)
			info = p.getJointInfo(self.panda_peg, j)
			jointName = info[1]
			jointType = info[2]
			if jointType == p.JOINT_REVOLUTE:
				p.resetJointState(self.panda_peg, j, self.jointPositions[index])
				index=index+1


	def position_control(self, num_step, target_pose):
		target_pos = target_pose[:3,3]
		target_orn = np.array(R.from_matrix(target_pose[:3,:3]).as_quat())
		ft_list = []
		for t in range(num_step):
			jointPoses = p.calculateInverseKinematics(
				self.panda_peg, self.endEffectorIndex, target_pos, target_orn)
			for i in range(self.pandaNumDofs):
				p.setJointMotorControl2(self.panda_peg, i, 
						p.POSITION_CONTROL, jointPoses[i], force=5.*240.)
			p.stepSimulation()

			# The force/torque will measure reaction forces
			ft = -np.array(p.getJointState(self.panda_peg, self.ftJointIndex)[2])
			ft_list.append(ft)

		ft_list = np.array(ft_list)

		ft_avg = np.mean(ft_list[-50:,:], axis=0)
		return ft_avg, ft_list


	def get_encoder_gt(self):
		cur_pos_world, cur_orn_world = p.getLinkState(self.panda_peg, self.endEffectorIndex)[0:2]
		cur_pose_world = self.transform(R.from_quat(cur_orn_world).as_matrix(), cur_pos_world)
		cur_pose_tcp = np.linalg.inv(self.oriMat).dot(cur_pose_world)

		cur_pos_tcp = cur_pose_tcp[:3,3]
		# print('cur_pos_tcp', cur_pose_tcp[:3,3])

		# print('dx', 5*cur_pos_tcp[0]*1000+10)
		# print('dy', 5*cur_pos_tcp[1]*1000+10)

		dx = np.around(5*cur_pos_tcp[0]*1000)+10
		dy = np.around(5*cur_pos_tcp[1]*1000)+10
		encoder_gt = [int(dx), int(dy)]
		return encoder_gt


	def transform(self, rotMat, transVec):
		matrix = np.eye(4)
		matrix[:3,:3]=rotMat
		matrix[:3,3]=transVec
		return matrix


	def reset(self):
		self.d = False
		self.r = 0
		self.stepCount = 0
		self.maxStep = 10

		# print('reset')
		self.robot_init()

		self.ori_position = np.array([0,0,0.625+0.07-0.00306])+self.base
		self.ori_orientation = p.getQuaternionFromEuler(
			np.array([180,0,-90])*math.pi/180)
		self.oriMat = self.transform(R.from_quat(self.ori_orientation).as_matrix(),
			self.ori_position)
		
		# # !!!!!!!!!
		# random end-effector pose initialization in the ft/endEffector coordinate system
		self.offset = np.array([0.00, 0.00])
		# self.offset = 0.001*np.random.uniform(-1.5,1.5,2)

		pose = self.transform(np.eye(3), [self.offset[0],self.offset[1],0])
		init_ee_pose = self.oriMat.dot(pose)

		ft, _ = self.position_control(550, init_ee_pose)
		encoder_gt = self.get_encoder_gt()

		# ft coordinate system
		ftPos, ftOrn = p.getLinkState(self.panda_peg, self.endEffectorIndex)[0:2]
		p.addUserDebugLine(ftPos, R.from_quat(ftOrn).apply([1,0,0])+ftPos, lineColorRGB=[1,0,0], lineWidth=2)
		p.addUserDebugLine(ftPos, R.from_quat(ftOrn).apply([0,1,0])+ftPos, lineColorRGB=[0,1,0], lineWidth=2)
		p.addUserDebugLine(ftPos, R.from_quat(ftOrn).apply([0,0,1])+ftPos, lineColorRGB=[0,0,1], lineWidth=2)

		# print('init_ee_pose_gt', ftPos)
		# print('init_ee_pose', init_ee_pose[:3,3])
		
		# ftPos, ftOrn = p.getLinkState(self.panda_peg, self.endEffectorIndex)[0:2]
		# p.addUserDebugLine(ftPos, R.from_quat(ftOrn).apply([1,0,0])+ftPos, lineColorRGB=[1,0,0], lineWidth=2)
		# p.addUserDebugLine(ftPos, R.from_quat(ftOrn).apply([0,1,0])+ftPos, lineColorRGB=[0,1,0], lineWidth=2)
		# p.addUserDebugLine(ftPos, R.from_quat(ftOrn).apply([0,0,1])+ftPos, lineColorRGB=[0,0,1], lineWidth=2)
		return ft, encoder_gt


	def step(self, a):
		# a: [dx(mm), dy(mm), dyaw(deg)] in the ft/endEffector coordinate system
		# eePos, eeOrn = p.getLinkState(self.panda_peg, self.endEffectorIndex)[0:2]
		# eeMat = self.transform(R.from_quat(eeOrn).as_matrix(),eePos)
		# pose = self.transform(
		# 	R.from_euler('z', a[2], degrees=True).as_matrix(),
		# 	np.array([a[0],a[1],0])*0.001)
		
		self.stepCount += 1

		self.offset += np.array(a[0:2])*0.001
		pose = self.transform(
			R.from_euler('z', 0, degrees=True).as_matrix(),
			[self.offset[0],self.offset[1],0])

		target_ee_pose = self.oriMat.dot(pose)

		# print('target_pos', target_ee_pose[:3,3])
		ft, ft_list = self.position_control(550, target_ee_pose)
		# print('target_pos_gt', p.getLinkState(self.panda_peg, self.endEffectorIndex)[0])

		# ftPos, ftOrn = p.getLinkState(self.panda_peg, self.ftJointIndex)[0:2]
		# p.addUserDebugLine(ftPos, R.from_quat(ftOrn).apply([1,0,0])+ftPos, lineColorRGB=[1,0,0], lineWidth=2)
		# p.addUserDebugLine(ftPos, R.from_quat(ftOrn).apply([0,1,0])+ftPos, lineColorRGB=[0,1,0], lineWidth=2)
		# p.addUserDebugLine(ftPos, R.from_quat(ftOrn).apply([0,0,1])+ftPos, lineColorRGB=[0,0,1], lineWidth=2)


		# plt.subplot(2,3,1)
		# plt.plot(ft_list[:,0])
		# plt.subplot(2,3,2)
		# plt.plot(ft_list[:,1])
		# plt.subplot(2,3,3)
		# plt.plot(ft_list[:,2])
		# plt.subplot(2,3,4)
		# plt.plot(ft_list[:,3])
		# plt.subplot(2,3,5)
		# plt.plot(ft_list[:,4])
		# plt.subplot(2,3,6)
		# plt.plot(ft_list[:,5])
		# plt.show()

		encoder_gt = self.get_encoder_gt()


		if self.stepCount > self.maxStep:
			self.d = True
		if encoder_gt[0]>20 or encoder_gt[0]<0 or encoder_gt[1]>20 or encoder_gt[1]<0:
			self.d = True
		if abs(encoder_gt[0]-10)<5 and abs(encoder_gt[1]-10)<5:
			self.d = True
			self.r = 1-self.stepCount/100

		return ft, self.r, self.d, encoder_gt


	def close(self):
		p.disconnect()






