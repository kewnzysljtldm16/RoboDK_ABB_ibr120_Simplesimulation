import gym
from gym import spaces
import numpy as np
from robolink import *    # RoboDK API
from robodk import *      # Robot toolbox
import torch
from stable_baselines3 import SAC

# 连接到RoboDK
RDK = Robolink()

# 定义机械臂和其他对象
robot = RDK.Item('ABB IRB 120-3/0.6')
tool = RDK.Item('吸盘工具')
referencing_frame = RDK.Item('搬运坐标系')
red_part = RDK.Item('红色工件')
blue_part = RDK.Item('蓝色工件')
yellow_part = RDK.Item('黄色工件')
cover = RDK.Item('端盖')

# 定义目标点
home_1 = RDK.Item('home1')
pick_red_app_1 = RDK.Item('pick_red_app1')
put_red_app = transl(97, -87, 25) * rotz(pi) * rotx(pi)
put_red = transl(97, -87, 45) * rotz(pi) * rotx(pi)

# 定义抓取和放置点的关节角度
PICK_JOINTS = np.array([8.784176, 42.583797, -2.314112, 0.000000, 49.730315, 8.784176])
PUT_JOINTS = np.array([9.017088, 29.918624, 13.562401, 0.000000, 46.518975, 9.017088])

# 折扣因子
GAMMA = 1

# 初始化机械臂位置
def init():
    red_part.setPose(transl(453, 70, 75))
    yellow_part.setPose(transl(453, -52, 75))
    blue_part.setPose(transl(453, 10, 75))
    cover.setPose(transl(382, -48, 89))
    robot.setPoseTool(tool)
    robot.setPoseFrame(referencing_frame)
    robot.setSpeed(20, 20, -1, -1)

class PickPlaceEnv(gym.Env):
    def __init__(self):
        super(PickPlaceEnv, self).__init__()
        self.action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
        init()
        self.step_count = 0
        self.subtask = 1  # 初始位置到抓取位置为子任务1，抓取到放置位置为子任务2

    def reset(self):
        init()
        self.robot = robot
        self.robot.MoveJ(home_1)
        self.step_count = 0
        self.subtask = 1
        return self._get_obs()

    def _get_obs(self):
        joints = self.robot.Joints().tolist()
        red_position = list(red_part.Pose().Pos())
        observation = joints + red_position + [0] * (12 - len(joints) - len(red_position))
        return np.array(observation, dtype=np.float32)

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        current_joints = np.array(self.robot.Joints().tolist())
        new_joints = (current_joints + action).tolist()

        try:
            self.robot.MoveJ(new_joints)
        except TargetReachError:
            reward = -100
            done = True
            return self._get_obs(), reward, done, {}

        end_effector_pos = list(self.robot.PoseTool().Pos())
        red_pos = list(red_part.Pose().Pos())
        reward = -np.linalg.norm(np.array(end_effector_pos) - np.array(red_pos))

        if self.subtask == 1:
            if self._is_near_joints(PICK_JOINTS):
                self.robot.MoveL(pick_red_app_1)
                self.tool.AttachClosest()
                pause(2)
                reward = 100
                self.subtask = 2
        elif self.subtask == 2:
            reward = -np.linalg.norm(np.array(end_effector_pos) - np.array(put_red.Pos()))
            if self._is_near_joints(PUT_JOINTS):
                self.robot.MoveL(put_red_app)
                self.tool.DetachAll()
                pause(2)
                self.robot.MoveJ(home_1)
                reward = 200
                done = True
                return self._get_obs(), reward, done, {}

        self.step_count += 1
        reward *= GAMMA ** self.step_count
        done = False
        return self._get_obs(), reward, done, {}

    def _is_near_joints(self, target_joints, threshold=5.0):
        current_joints = np.array(self.robot.Joints().tolist())
        return np.linalg.norm(current_joints - target_joints) < threshold

    def render(self, mode='human'):
        pass

env = PickPlaceEnv()

# 使用SAC算法进行训练
print('device is : {}'.format(torch.cuda.is_available()))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SAC('MlpPolicy', env, verbose=1, device=device)
model.learn(total_timesteps=20000)
model.save("D:/work/learning/roboDK_learning/sac_robodk_red_pick")

# 测试训练好的模型
model = SAC.load("D:/work/learning/roboDK_learning/sac_robodk_red_pick")
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    if dones:
        obs = env.reset()
