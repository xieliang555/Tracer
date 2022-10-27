from gym.envs.registration import register

register(
    id='peg-in-hole-v0',
    entry_point='gymEnv.envs.peg_in_hole_v0:PegInHole',
)

register(
    id='peg-in-hole-test-v1',
    entry_point='gymEnv.envs.peg_in_hole_test:PegInHole',
)

register(
    id='peg-in-hole-v1',
    entry_point='gymEnv.envs.peg_in_hole_v1:PegInHole',
)

register(
    id='peg-in-hole-v2',
    entry_point='gymEnv.envs.peg_in_hole_v2:PegInHole',
)

register(
    id='peg-in-hole-v3',
    entry_point='gymEnv.envs.peg_in_hole_v3:PegInHole',
)

register(
    id='peg-in-hole-v4',
    entry_point='gymEnv.envs.peg_in_hole_v4:PegInHole',
)

register(
    id='peg-in-hole-v5',
    entry_point='gymEnv.envs.peg_in_hole_v5:PegInHole',
)

register(
    id='peg-in-hole-v6',
    entry_point='gymEnv.envs.peg_in_hole_v6:PegInHole',
)



# peg-in-hole-v0: original

# peg-in-hole-test-v1: stacked ft wrench

# peg-in-hole-v1: 使用tcp末端坐标系作为参考

# peg-in-hole-v2: 2DoF RL

# peg-in-hole-v3: 3DoF RL

# peg-in-hole-v4: env constructed by grid map 

# peg-in-hole-v5 env constructed by grid map , PCA for 6-dim ft

# peg-in-hole-v6 env for tdm(transferable dynamic model)







