from robolink import *  # RoboDK API
from robodk import *  # Robot toolbox
import numpy as np

RDK = Robolink()
pose = eye()
##########定义对象##########
robot = RDK.Item('ABB IRB 120-3/0.6')
tool = RDK.Item('吸盘工具')
referencing_frame = RDK.Item('搬运坐标系')
red_part = RDK.Item('红色工件')
blue_part = RDK.Item('蓝色工件')
yellow_part = RDK.Item('黄色工件')
cover = RDK.Item('端盖')

##########定义目标点##########
home_1 = RDK.Item('home1')
home_2 = RDK.Item('home2')
home_3 = RDK.Item('home3')
home_4 = RDK.Item('home4')
home_5 = RDK.Item('home5')
home_6 = RDK.Item('Target 13')  # 新的目标点
pick_red_app_1 = RDK.Item('pick_red_app1')
pick_red_app_2 = RDK.Item('pick_red_app2')
pick_red_app_3 = RDK.Item('pick_red_app3')
pick_red_app_4 = RDK.Item('pick_red_app4')

pick_red = transl(153, -80, 25) * rotz(pi) * rotx(pi)
put_red_app = transl(97, -87, 25) * rotz(pi) * rotx(pi)
put_red = transl(97, -87, 45) * rotz(pi) * rotx(pi)


##########定义初始化函数#########
def init():
    red_part.setPose(transl(453, 70, 75))
    yellow_part.setPose(transl(453, -52, 75))
    blue_part.setPose(transl(453, 10, 75))
    cover.setPose(transl(382, -48, 89))


##########仿真程序##########
init()
robot.setPoseTool(tool)
robot.setPoseFrame(referencing_frame)
robot.setSpeed(20, 20, -1, -1)
print(robot.Joints())

import time

import numpy as np


def pose_to_rot_mat(pose):
    # Convert pose to rotation matrix
    rot_mat = np.array([[pose[0, 0], pose[0, 1], pose[0, 2]],
                        [pose[1, 0], pose[1, 1], pose[1, 2]],
                        [pose[2, 0], pose[2, 1], pose[2, 2]]])
    return rot_mat


def pieper_inverse_kinematics(pose):
    # Extract position and rotation from pose
    position = pose.Pos()
    rotation = pose_to_rot_mat(pose)

    # Initialize joint angles
    joint_angles = [0] * 6

    # Calculate joint angles
    # Joint 1 (theta1)
    joint_angles[0] = np.arctan2(position[1], position[0])

    # Joint 3 (theta3)
    D = (position[0] ** 2 + position[1] ** 2 + (position[2] - 72) ** 2 - 70 ** 2 - 302 ** 2 - 270 ** 2) / (2 * 70 * 302)
    if -1 <= D <= 1:
        joint_angles[2] = np.arctan2(-np.sqrt(1 - D ** 2), D)
    else:
        # Handle cases where D is out of valid range
        joint_angles[2] = 0  # Set joint angle to default value

    # Joint 2 (theta2)
    alpha = np.arctan2(position[2] - 72, np.sqrt(position[0] ** 2 + position[1] ** 2))
    beta = np.arctan2(302 * np.sin(joint_angles[2]), 70 + 302 * np.cos(joint_angles[2]))
    joint_angles[1] = np.pi / 2 - alpha - beta

    # Joint 4, 5, 6 (theta4, theta5, theta6)
    R03 = np.array([[np.cos(joint_angles[0]) * np.cos(joint_angles[1] + joint_angles[2]), -np.sin(joint_angles[0]),
                     np.cos(joint_angles[0]) * np.sin(joint_angles[1] + joint_angles[2])],
                    [np.sin(joint_angles[0]) * np.cos(joint_angles[1] + joint_angles[2]), np.cos(joint_angles[0]),
                     np.sin(joint_angles[0]) * np.sin(joint_angles[1] + joint_angles[2])],
                    [-np.sin(joint_angles[1] + joint_angles[2]), 0, np.cos(joint_angles[1] + joint_angles[2])]])

    R36 = np.dot(np.linalg.inv(R03), rotation)
    joint_angles[3] = np.arctan2(R36[2, 1], R36[2, 2])
    joint_angles[4] = -np.arctan2(-R36[2, 0], np.sqrt(R36[0, 0] ** 2 + R36[1, 0] ** 2))
    joint_angles[5] = np.arctan2(R36[1, 0], R36[0, 0])

    return joint_angles


robot_item_list = [robot]
joints_list = []
count = 0
# 从0度到90度，每秒变化10度
for angle in range(0, 91, 10):
    count += 1
    current_joint_angles = [angle, -0.000, 0.000, 0.000, 90.000, -0.000]
    joints_list.append(current_joint_angles)

    RDK.setJoints(robot_item_list, [current_joint_angles])

    # robodk自带的旋转矩阵计算
    if angle % 30 == 0:
        print("Current joint angles:\n", robot.Joints())
        print("Current Rotation Matrix:\n", Mat(robot.SolveFK(current_joint_angles)))

        # DH表得出数据进行选择矩阵计算
        # DH参数
        theta = current_joint_angles
        d = [290, 0, 0, 302, 0, 72]
        a = [0, 270, 70, 0, 0, 0]
        alpha = [np.radians(-90), np.radians(0), np.radians(-90), np.radians(0), np.radians(-90), np.radians(0)]


        # 定义选择矩阵函数
        def compute_transform_matrix(theta, d, a, alpha):
            return np.array([
                [np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
                [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
                [0, np.sin(alpha), np.cos(alpha), d],
                [0, 0, 0, 1]
            ])


        # 计算各个选择矩阵
        A_0_1 = compute_transform_matrix(theta[0], d[0], a[0], alpha[0])
        A_1_2 = compute_transform_matrix(theta[1], d[1], a[1], alpha[1])
        A_2_3 = compute_transform_matrix(theta[2], d[2], a[2], alpha[2])
        A_3_4 = compute_transform_matrix(theta[3], d[3], a[3], alpha[3])
        A_4_5 = compute_transform_matrix(theta[4], d[4], a[4], alpha[4])
        A_5_6 = compute_transform_matrix(theta[5], d[5], a[5], alpha[5])

        # 计算最终选择矩阵
        A_0_6 = np.dot(A_0_1, np.dot(A_1_2, np.dot(A_2_3, np.dot(A_3_4, np.dot(A_4_5, A_5_6)))))

# 使用逆运动学求解home_6的关节角度位置
home_6_pose = home_6.Pose()
home_6_xyzwpr = Pose_2_TxyzRxyz(home_6_pose)
target_joint_angles_DK = robot.SolveIK(home_6_pose)
target_joint_angles = pieper_inverse_kinematics(home_6.Pose())


print("Home 6 Pose (XYZWPR):\n", home_6_xyzwpr)

print("Pieper Home 6 Joint Angles:\n", target_joint_angles)

print("RoboDK Home 6 Joint Angles:\n", target_joint_angles_DK)

RDK.setJoints(robot_item_list, [target_joint_angles_DK])
print("**RoboDK Joint Angles:\n", robot.Joints())

time.sleep(2)
RDK.setJoints(robot_item_list, [target_joint_angles])
print("**Pieper Joint Angles:\n", robot.Joints())
