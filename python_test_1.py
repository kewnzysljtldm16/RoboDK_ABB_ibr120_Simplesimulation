from robolink import *    # RoboDK API
from robodk import *      # Robot toolbox
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

pick_red = transl(153,-80,25)*rotz(pi)*rotx(pi)
put_red_app = transl(97,-87,25)*rotz(pi)*rotx(pi)
put_red = transl(97,-87,45)*rotz(pi)*rotx(pi)

##########定义初始化函数#########
def init():
    red_part.setPose(transl(453,70,75))
    yellow_part.setPose(transl(453,-52,75))
    blue_part.setPose(transl(453,10,75))
    cover.setPose(transl(382,-48,89))

##########仿真程序##########
init()
robot.setPoseTool(tool)
robot.setPoseFrame(referencing_frame)
robot.setSpeed(20,20,-1,-1)
print(robot.Joints())

robot.MoveJ(home_1)
robot.MoveJ(pick_red_app_1)
robot.MoveL(pick_red)
tool.AttachClosest()
pause(0.2)
robot.MoveL(pick_red_app_1)
robot.MoveJ(put_red_app)
robot.MoveL(put_red)
tool.DetachAll()
pause(0.2)
robot.MoveL(put_red_app)
robot.MoveJ(home_1)

