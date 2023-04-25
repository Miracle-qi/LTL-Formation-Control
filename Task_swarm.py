#!/usr/bin/env python3
import sys
sys.path.append("../../../devel/lib/python3/dist-packages")

import rospy
import numpy as np
from nav_msgs.msg import Odometry
from quadrotor_msgs.msg import PositionCommand
from visualization_msgs.msg import Marker
from LTL_schedule.ltl_ctrl import TaskSchedule
from LTL_schedule.grid_display import GridDisplay
from FxT_control.FxT_Swarm_Ctrl import FxT_QP_swarm



class TaskInfo:

    def __init__(self, shape, unit, scale, form):
        self.shape = shape
        self.unit = unit
        self.scale = scale
        self.form_dict = form
        self.cur_state = 'X0_f0'
        self.cur_index = "Sinit"
        self.cur_pos = None
        self.cur_form = None
        self.goal_state = None
        self.goal_index = None
        self.goal_center = None

        self.goal_form = None

        self.form_topo = {'horizon': [[0.7, 0], [0.7, 0], [-1.4, 0]],
                          'vertical': [[0, 0.7], [0, 0.7], [0, -1.4]],
                          'triangle': [[1.6, 0], [-0.8, 1.4], [-0.8, -1.4]]}

    def find_next_task(self, next_state_index, next_state):
        self.goal_state = next_state
        pos_index = int(next_state[1:-3])
        form_index = int(next_state[-1])
        pos_x = self.scale * self.unit[0] * (pos_index % self.shape[0] + 1/2)
        pos_y = self.scale * self.unit[1] * (pos_index // self.shape[0] + 1/2)
        formation = list(self.form_dict.keys())[form_index]
        self.goal_index = next_state_index
        self.goal_center = (pos_x, pos_y)
        self.goal_form = formation
        return (pos_x, pos_y), formation

    def change_cur_state(self, cur_pos):
        self.cur_pos = cur_pos
        cur_center = [(cur_pos[0] + cur_pos[2] + cur_pos[4]) / 3,
                      (cur_pos[1] + cur_pos[3] + cur_pos[5]) / 3]
        dist = np.hypot(cur_center[0]-self.goal_center[0], cur_center[1]-self.goal_center[1])
        if (dist <= 0.2) and self.check_form(self.goal_form):
            self.cur_state = self.goal_state
            self.cur_index = self.goal_index
            self.cur_form = self.goal_form

    def check_form(self, goal_form):
        print("Cur pos: ", self.cur_pos)
        print("Goal form: ", goal_form)
        # print("Goal Index: ", self.goal_index)
        topo = self.form_topo[goal_form]
        r_accept = 0.1
        for n in range(len(topo)):
            _n = 2 * (n + 1) % (len(self.cur_pos))
            _x = self.cur_pos[_n] - self.cur_pos[2 * n]
            _y = self.cur_pos[_n+1] - self.cur_pos[2 * n + 1]
            if (_x - topo[n][0]) ** 2 + (_y - topo[n][1]) ** 2 >= r_accept ** 2:
                return False
        return True


class SwarmNode:

    def __init__(self, scale, space, unit, area_set, form_set):
        rospy.init_node('FxT_swarm', anonymous=True)
        self.controller = FxT_QP_swarm()
        self.area_set = area_set
        self.space = space
        self.scale = scale
        self.task_info = TaskInfo(space, unit, scale, form_set)
        self.ltl_tasks = TaskSchedule(space, unit, area_set, form_set)
        self.ltl_tasks.setup()
        self.ltl_dispalyer = GridDisplay(space, unit, area_set, form_set)
        self.ltl_dispalyer.plot_TLgraph3d(traj_show=False)

        agents_num = 3
        self.cur_state = np.zeros(2 * agents_num)
        self.cur_vel = np.zeros(2 * agents_num)

        self.form_topo = self.task_info.form_topo['triangle']
        self.odom_sub_0 = rospy.Subscriber('/odom_0', Odometry, self.callback_0)
        self.odom_sub_1 = rospy.Subscriber('/odom_1', Odometry, self.callback_1)
        self.odom_sub_2 = rospy.Subscriber('/odom_2', Odometry, self.callback_2)

        self.env_pub = rospy.Publisher('/obstacle', Marker, queue_size=10)
        self.cmd_pub_0 = rospy.Publisher('/cmd_0', PositionCommand, queue_size=10)
        self.cmd_pub_1 = rospy.Publisher('/cmd_1', PositionCommand, queue_size=10)
        self.cmd_pub_2 = rospy.Publisher('/cmd_2', PositionCommand, queue_size=10)

        self.battery_flag = 0

    def callback_0(self, odom_data):
        self.cur_state[0] = odom_data.pose.pose.position.x
        self.cur_state[1] = odom_data.pose.pose.position.y
        self.cur_vel[0] = odom_data.twist.twist.linear.x
        self.cur_vel[1] = odom_data.twist.twist.linear.y

    def callback_1(self, odom_data):
        self.cur_state[2] = odom_data.pose.pose.position.x
        self.cur_state[3] = odom_data.pose.pose.position.y
        self.cur_vel[2] = odom_data.twist.twist.linear.x
        self.cur_vel[3] = odom_data.twist.twist.linear.y

    def callback_2(self, odom_data):
        self.cur_state[4] = odom_data.pose.pose.position.x
        self.cur_state[5] = odom_data.pose.pose.position.y
        self.cur_vel[4] = odom_data.twist.twist.linear.x
        self.cur_vel[5] = odom_data.twist.twist.linear.y

    def run_1(self):
        goal_point = np.array([10, 10])
        goal_radius = 0.3
        obs_point = np.array([1, 1])
        obs_radius = 0.3
        Fix_T = 3
        self.controller.setup(goal_point, goal_radius, False, obs_point, obs_radius)
        opt_inputs = self.controller.solve(self.cur_state, self.cur_vel, self.form_topo, Fix_T)
        self.pub_cmd(opt_inputs)

    def run_2(self):
        print("battery_flag:", self.battery_flag)
        Fix_T = 3.5

        if self.task_info.cur_state == "X18_f2":
            self.battery_flag += 1
            print("Achieved Task!")
            print("Low Battery!")
        if self.task_info.cur_state == "X0_f0":
            self.battery_flag = 0
            print("Fully Charged!")

        battery = True if self.battery_flag <= 600 else False
        if battery == False:
            print("Low Battery!")

        next_ltl_state, dum = self.ltl_tasks.run(self.task_info.cur_index, battery)
        next_ltl_loc = dum['loc']

        goal_point, goal_form = self.task_info.find_next_task(next_ltl_state, next_ltl_loc)
        # print("Goal_point: ", goal_point)
        # print("Goal_form: ", goal_form)
        goal_radius = 0.2
        self.controller.setup(goal_point, goal_radius, False)
        # print("states: ", self.cur_state)
        form_topo = self.task_info.form_topo[goal_form]
        opt_inputs = self.controller.solve(self.cur_state, self.cur_vel, form_topo, Fix_T)
        self.pub_cmd(opt_inputs)
        print("cur_state: ", self.task_info.cur_state)
        print("goal_state: ", self.task_info.goal_state)
        self.env_display(self.scale)
        self.ltl_dispalyer.update(self.task_info.cur_state, self.task_info.goal_state)
        self.task_info.change_cur_state(self.cur_state)
        return batter_flag 


    def pub_cmd(self, opt_inputs):
        dt = 0.4
        Fixed_z = 1
        alpha = 0.

        cmd_0 = PositionCommand()
        cmd_0.position.x = self.cur_state[0]+ (alpha * self.cur_vel[0] +(1-alpha)*opt_inputs[0]) * dt
        cmd_0.position.y = self.cur_state[1]+ (alpha * self.cur_vel[1] +(1-alpha)*opt_inputs[1]) * dt
        cmd_0.position.z = Fixed_z
        # cmd_0.velocity.x = float('nan')
        # cmd_0.velocity.y = float('nan')
        # cmd_0.velocity.z = float('nan')
        # cmd_0.acceleration.x = float('nan')
        # cmd_0.acceleration.y = float('nan')
        # cmd_0.acceleration.z = float('nan')
        self.cmd_pub_0.publish(cmd_0)

        cmd_1 = PositionCommand()
        cmd_1.position.x = self.cur_state[2] + (alpha * self.cur_vel[2] + (1-alpha)*opt_inputs[2]) * dt
        cmd_1.position.y = self.cur_state[3] + (alpha * self.cur_vel[3] + (1-alpha)*opt_inputs[3]) * dt
        cmd_1.position.z = Fixed_z
        self.cmd_pub_1.publish(cmd_1)

        cmd_2 = PositionCommand()
        cmd_2.position.x = self.cur_state[4] + (alpha * self.cur_vel[4] + (1-alpha)*opt_inputs[4]) * dt
        cmd_2.position.y = self.cur_state[5] + (alpha * self.cur_vel[5] + (1-alpha)*opt_inputs[5]) * dt
        cmd_2.position.z = Fixed_z
        self.cmd_pub_2.publish(cmd_2)
        print("Current states:", self.cur_state)

    def env_display(self, scale):
        n = 0
        pos = [scale * self.space[0]/2, scale * self.space[1]/2, 0.01]
        shape = [scale * self.space[0], scale * self.space[1], 0.02]
        color = [0.5, 0.8, 0.8, 0.8]
        self.add_maker(n, "area" + str(n), pos, shape, color)
        for area, label in zip(self.area_set.keys(), self.area_set.values()):
            n += 1
            if label == "home":
                pos = [scale * sum(area[0])/2, scale * sum(area[1])/2, 0.01]
                shape = [scale * (area[0][1] - area[0][0]), scale * (area[1][1] - area[1][0]), 0.02]
                color = [0.8, 0.8, 0.5, 0.2]
                self.add_maker(n, "area_" + str(n), pos, shape, color)
            elif label == "bench":
                pos = [scale * sum(area[0])/2, scale * sum(area[1])/2, 0.01]
                shape = [scale * (area[0][1] - area[0][0]), scale * (area[1][1] - area[1][0]), 0.02]
                color = [0.8, 0.2, 0.5, 0.8]
                self.add_maker(n, "area_" + str(n), pos, shape, color)
            elif label == "obstacle":
                pos = [scale * sum(area[0])/2, scale * sum(area[1])/2, 0.6]
                shape = [scale * (area[0][1] - area[0][0]), scale * (area[1][1] - area[1][0]), 1.2]
                color = [1.0, 1.0, 0.2, 0.2]
                self.add_maker(n, "area_" + str(n), pos, shape, color)

    def add_maker(self, id, name, pos, shape, color):
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = rospy.Time.now()

        marker.ns = name
        marker.id = id
        marker.type = marker.CUBE
        marker.action = marker.ADD
        marker.pose.position.x = pos[0]
        marker.pose.position.y = pos[1]
        marker.pose.position.z = pos[2]
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = shape[0]
        marker.scale.y = shape[1]
        marker.scale.z = shape[2]
        marker.color.a = color[0]
        marker.color.r = color[1]
        marker.color.g = color[2]
        marker.color.b = color[3]
        self.env_pub.publish(marker)
    

if __name__ == "__main__":
    space = (5, 5) # The origin is (0,0) by default
    scale = 2
    unit = (1, 1)
    area_dict = {((0, 1), (0, 1)): 'home',
                 ((3, 4), (3, 4)): 'bench',
                 ((1, 2), (2, 4)): 'obstacle',
                 ((1, 4), (1, 2)): 'obstacle'}
    form_dict = {'horizon':  (2, 1, 1),
                 'vertical': (1, 2, 1),
                 'triangle': (2, 2, 2)}

    node = SwarmNode(scale, space, unit, area_dict, form_dict)
    batter_flag = 0 
    node.run_2()
    rospy.sleep(20)
    print("Set up!")
    while not rospy.is_shutdown():
        _flag = node.run_2()



