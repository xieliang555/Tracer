import gym
import pybullet as p 
import pybullet_data
import pkgutil
import math
import numpy as np
import os
import time
from scipy.spatial.transform import Rotation as R



class PegInHole(gym.Env):
	"""general peg-in-hole with RL"""
	def __init__(self, peg_type=None, seed=0):
		super(PegInHole, self).__init__()
		# connecting to pybullet
		print(peg_type)
		if 0:
			p.connect(p.GUI)
		else:
			p.connect(p.DIRECT)
			# egl = pkgutil.get_loader('eglRenderer')
			# if egl:
			# 	p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
			# else:
			# 	p.loadPlugin("eglRendererPlugin")
				
		np.random.seed(seed)

		# loading objects
		self.panda_peg_id = p.loadURDF(os.path.join(os.path.dirname(os.path.realpath(__file__)),
			"mesh/"+peg_type+"/peg/peg.urdf"), basePosition=[0,0,0], useFixedBase=1,
			baseOrientation=p.getQuaternionFromEuler([0,0,math.pi]))
		self.hole_id = p.loadURDF(os.path.join(os.path.dirname(os.path.realpath(__file__)), 
			"mesh/"+peg_type+"/hole/hole.urdf"), basePosition=[0.603,0,0.6], globalScaling=0.05, useFixedBase=1,
			baseOrientation=p.getQuaternionFromEuler([0,-math.pi/2,math.pi/2]))

		# !!!!!!!!!!
		self.endEffectorIndex = 12
		# !!!!! 是11而不是12
		self.ftJointIndex = 11
		self.pandaNumDofs = 7
		# !!!!!!!!
		# p.setGravity(9.8,0,0)
		p.enableJointForceTorqueSensor(self.panda_peg_id, self.ftJointIndex, 1)
		p.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=-30, 
			cameraPitch=-30, cameraTargetPosition=[0.6,0,0.6])

		low = np.float32(0)
		high = np.float32(1)
		# !!!!!!!!
		self.observation_space = gym.spaces.Box(low, high, shape=(6,), dtype=np.float32)
		self.action_space = gym.spaces.Discrete(4)


	def robot_init(self):
		self.jointPositions=[
			2.94, -0.574, 0.103, -2.687, -0.218, 3.704, -0.544, 0.02, 0.02]
		index = 0
		for j in range(p.getNumJoints(self.panda_peg_id)):
			p.changeDynamics(self.panda_peg_id, j, linearDamping=0, angularDamping=0)
			info = p.getJointInfo(self.panda_peg_id, j)
			jointName = info[1]
			jointType = info[2]
			if jointType == p.JOINT_PRISMATIC:
				p.resetJointState(self.panda_peg_id, j, self.jointPositions[index])
				index=index+1 
			if jointType == p.JOINT_REVOLUTE:
				p.resetJointState(self.panda_peg_id, j, self.jointPositions[index])
				index=index+1


	def position_control(self, num_step, target_pos):
		target_orn = p.getQuaternionFromEuler([math.pi/2, 0, math.pi/2])
		for t in range(num_step):
			jointPoses = p.calculateInverseKinematics(self.panda_peg_id, 
				self.endEffectorIndex, target_pos, target_orn)
			for i in range(self.pandaNumDofs):
				p.setJointMotorControl2(self.panda_peg_id, i, 
						p.POSITION_CONTROL, jointPoses[i], force=5.*240.)
			for i in [9,10]:
				p.setJointMotorControl2(self.panda_peg_id,i, p.POSITION_CONTROL, 0.02, force=10)
			p.stepSimulation()


	def hybrid_force_pos_control(self, target_pos, num_steps=100):
		kp,ki,kd = 0.000, 1, 0
		pid = PID(kp,ki,kd)
		transfom_factor = 0.000003
		setpoint = 10
		for t in range(num_steps):
			ft_wrench = -np.array(p.getJointState(self.panda_peg_id, self.ftJointIndex)[2])
			f_transform = R.from_euler('x', 180, degrees=True).inv().apply(ft_wrench[0:3])
			t_transform = R.from_euler('x', 180, degrees=True).inv().apply(ft_wrench[3:])
			ft_transform = np.concatenate((f_transform, t_transform), 0)
			force_err = pid.calc(ft_transform[2], setpoint)
			x_pos_err = force_err*transfom_factor
			x_cur = p.getLinkState(self.panda_peg_id, self.endEffectorIndex)[0][0]
			x_tar = x_cur + x_pos_err
			pos_tar = [x_tar, target_pos[1], target_pos[2]]
			self.position_control(1, pos_tar)
			# contact plane: 0.602
			if x_cur > 0.605:
				break


	def reset(self):
		self.robot_init()
		self.done = False
		self.maxStep = 100
		self.stepCount = 0
		self.offset = np.array([0.,0.,0.])
		random_pos = np.random.uniform(-0.005,0.005,2)
		self.target_pos = np.array([0.5, random_pos[0], 0.6+random_pos[1]])
		self.hybrid_force_pos_control(self.target_pos, num_steps=1000)
		ft_wrench =-np.array(p.getJointState(self.panda_peg_id, self.ftJointIndex)[2])

		pos, orn = p.getLinkState(self.panda_peg_id, self.endEffectorIndex)[0:2]
		x,y,z = R.from_quat(orn).apply([1,0,0]) + pos
		p.addUserDebugLine(pos,[x,y,z], lineColorRGB=[1,0,0])
		x,y,z = R.from_quat(orn).apply([0,1,0]) + pos
		p.addUserDebugLine(pos,[x,y,z], lineColorRGB=[0,1,0])
		x,y,z = R.from_quat(orn).apply([0,0,1]) + pos
		p.addUserDebugLine(pos,[x,y,z], lineColorRGB=[0,0,1])

		pos,orn = p.getLinkState(self.panda_peg_id,self.ftJointIndex)[0:2]
		x,y,z = R.from_quat(orn).apply(R.from_euler('x', 180, degrees=True).apply([1,0,0]))+pos
		p.addUserDebugLine(pos, [x,y,z], lineColorRGB=[1,0,0])
		x,y,z = R.from_quat(orn).apply(R.from_euler('x', 180, degrees=True).apply([0,1,0]))+pos
		p.addUserDebugLine(pos,[x,y,z], lineColorRGB=[0,1,0])
		x,y,z = R.from_quat(orn).apply(R.from_euler('x', 180, degrees=True).apply([0,0,1]))+pos
		p.addUserDebugLine(pos,[x,y,z], lineColorRGB=[0,0,1])

		f_transform = R.from_euler('x', 180, degrees=True).inv().apply(ft_wrench[0:3])
		t_transform = R.from_euler('x', 180, degrees=True).inv().apply(ft_wrench[3:])
		ft_transform = np.concatenate((f_transform, t_transform), 0)

		return ft_transform


	def step(self, a):
		self.stepCount += 1

		if a == 0:
			self.offset += np.array([0.001, 0, 0])
		elif a == 1:
			self.offset += np.array([-0.001, 0, 0])
		elif a == 2:
			self.offset += np.array([0, 0.001, 0])
		elif a == 3:
			self.offset += np.array([0, -0.001, 0])

		pos, orn = p.getLinkState(self.panda_peg_id, self.endEffectorIndex)[0:2]
		target_pos = R.from_quat(orn).apply(R.from_euler('x', 180, degrees=True).apply(self.offset)) + pos
		self.hybrid_force_pos_control(target_pos, num_steps=100)
		
		ft_wrench = -np.array(p.getJointState(self.panda_peg_id, self.ftJointIndex)[2])
		f_transform = R.from_euler('x', 180, degrees=True).inv().apply(ft_wrench[0:3])
		t_transform = R.from_euler('x', 180, degrees=True).inv().apply(ft_wrench[3:])
		ft_transform = np.concatenate((f_transform, t_transform), 0)
		

		cur_pos = p.getLinkState(self.panda_peg_id, self.endEffectorIndex)[0]
		epi = (cur_pos[1]**2 + (cur_pos[2]-0.6)**2)**0.5
		if cur_pos[0] > 0.605 and epi < 0.004:
			self.done = True
			reward = 1-self.stepCount/self.maxStep
		elif self.stepCount >= self.maxStep:
			self.done = True
			reward = 0
		else:
			reward = 0

		info = {}
		return ft_transform, reward, self.done, info
		


	def close(self):
		p.disconnect()
