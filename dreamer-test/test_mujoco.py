import mujoco
import dm_control
from dm_control import suite
import numpy as np
import time

# 加载humanoid环境
env = suite.load(domain_name="humanoid", task_name="stand")

# 获取初始状态
timestep = env.reset()

# 执行随机动作
action_spec = env.action_spec()
random_action = np.random.uniform(
    action_spec.minimum, 
    action_spec.maximum, 
    size=action_spec.shape
)

# 执行多个步骤
print("MuJoCo和dm-control环境配置成功!")
print(f"初始观察空间形状: {timestep.observation['joint_angles'].shape}")

# 使用physics获取渲染图像
camera = dm_control.mujoco.engine.Camera(
    env.physics,
    height=480,
    width=640
)

for i in range(100):
    action = np.random.uniform(
        action_spec.minimum,
        action_spec.maximum,
        size=action_spec.shape
    )
    timestep = env.step(action)
    
    if i % 20 == 0:  # 每20步保存一帧
        pixels = camera.render()
        print(f"Step {i}, Reward: {timestep.reward}")
        print(f"渲染图像形状: {pixels.shape}")  # 应该是(480, 640, 3)

print("测试完成！")

