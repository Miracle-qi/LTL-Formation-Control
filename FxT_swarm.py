#!/usr/bin/env python3
import casadi as ca
import numpy as np
import rospy
from nav_msgs.msg import Odometry
# import FxT_control
from quadrotor_msgs.msg import PositionCommand


class FxT_QP_swarm:
    def __init__(self, num=3):
        self.x_dim = 2
        self.u_dim = 2
        self.r_dim = 3
        self.agents_num = num
        self.mu = 1.5
        self.max_u = 10

        self.opti = ca.Opti()
        self.U = self.opti.variable(self.u_dim * num)
        self.R = self.opti.variable(self.r_dim)
        self.X = self.opti.parameter(self.x_dim * num)
        self.alpha_1 = self.opti.parameter(1)
        self.alpha_2 = self.opti.parameter(1)
        self.form_topo = self.opti.parameter(num, 2)

        self.Q = ca.diag([0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 20, 20, 1])
        self.F = ca.DM([0, 0, 0, 0, 0, 0, 1, 1, 0])
        self.r_accept = 0.3

    def setup(self, goal_point, goal_radius, obs_flag, obs_point=(0., 0.), obs_radius=0.):
        self.opti = ca.Opti()
        self.U = self.opti.variable(self.u_dim * self.agents_num)
        self.R = self.opti.variable(self.r_dim)
        self.X = self.opti.parameter(self.x_dim * self.agents_num)
        self.alpha_1 = self.opti.parameter(1)
        self.alpha_2 = self.opti.parameter(1)
        self.form_topo = self.opti.parameter(self.agents_num, 2)

        gamma_1 = 1 + 1 / self.mu
        gamma_2 = 1 - 1 / self.mu

        Z = ca.vertcat(self.U, self.R)
        costs = ca.trace(Z.T @ self.Q @ Z) + self.F.T @ Z
        self.opti.minimize(costs)  # Note R[0] should be non-positive to make CLF satisfied

        self.opti.subject_to(self.U[:] <= self.max_u)
        self.opti.subject_to(self.U[:] >= -self.max_u)

        # Fixed-time CLF for Reaching
        x_c = (self.X[0] + self.X[2] + self.X[4]) / 3
        y_c = (self.X[1] + self.X[3] + self.X[5]) / 3
        v_x_c = (self.U[0] + self.U[2] + self.U[4]) / 3
        v_y_c = (self.U[1] + self.U[3] + self.U[5]) / 3
        h_g = (x_c - goal_point[0]) ** 2 + (y_c - goal_point[1]) ** 2 - goal_radius ** 2
        d_h_g = 2 * (x_c - goal_point[0]) * v_x_c + 2 * (y_c - goal_point[1]) * v_y_c
        self.opti.subject_to(d_h_g <= self.R[0] * h_g - self.alpha_1 * ca.fmax(0, h_g) ** gamma_1
                             - self.alpha_2 * ca.fmax(0, h_g) ** gamma_2)

        for n in range(self.agents_num):
            # Fixed-time CLF for Formation
            _n = 2 * (n + 1) % (2 * self.agents_num)
            _x = self.X[_n] - self.X[2 * n]
            _y = self.X[_n + 1] - self.X[2 * n + 1]
            _vx = self.U[_n] - self.U[2 * n]
            _vy = self.U[_n + 1] - self.U[2 * n + 1]
            _hg = (_x - self.form_topo[n, 0]) ** 2 + (_y - self.form_topo[n, 1]) ** 2 - self.r_accept ** 2
            _d_hg = 2 * (_x - self.form_topo[n, 0]) * _vx + 2 * (_y - self.form_topo[n, 1]) * _vy
            self.opti.subject_to(_d_hg <= self.R[1] * _hg - self.alpha_1 * ca.fmax(0, _hg) ** gamma_1
                                 - self.alpha_2 * ca.fmax(0, _hg) ** gamma_2)

            # CBF for Obstacle Avoidance
            if obs_flag:
                _hs = obs_radius**2 - (self.X[2*n]-obs_point[0])**2 - (self.X[2*n]-obs_point[1])**2
                _d_hs = - 2 * (self.X[2*n]-obs_point[0]) * self.U[2*n] - 2 * (self.X[2*n+1]-obs_point[1]) * self.U[2*n+1]
                self.opti.subject_to(_d_hs <= - self.R[2] * _hs)

        ipopt_options = {
            'verbose': False,
            "ipopt.tol": 1e-6,
            "ipopt.acceptable_tol": 1e-6,
            "ipopt.max_iter": 100,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.print_level": 0,
            "print_time": True
        }
        self.opti.solver('ipopt', ipopt_options)

    def solve(self, cur_state, form_topo, T_fix):
        cur_state = ca.DM(cur_state)
        self.opti.set_value(self.X, cur_state)
        self.opti.set_value(self.alpha_1, self.mu * ca.pi / (2 * T_fix))
        self.opti.set_value(self.alpha_2, self.mu * ca.pi / (2 * T_fix))
        self.opti.set_value(self.form_topo, np.array(form_topo))

        sol = self.opti.solve()
        print("\n self.optimal Input: \n", sol.value(self.U))

        return sol.value(self.U)


def formation(form_name):
    form_list = {'horizon': [[1, 0], [1, 0], [-2, 0]],
                 'vertical': [[0, 1], [0, 1], [0, -2]],
                 'triangle': [[2, 0], [-1, 2], [-1, -2]]}
    return form_list[form_name]


class SwarmNode:

    def __init__(self):
        rospy.init_node('FxT_swarm', anonymous=True)
        self.controller = FxT_QP_swarm()

        agents_num = 3
        self.cur_state = np.zeros(2 * agents_num)
        self.form_topo = formation('triangle')

        self.odom_sub_0 = rospy.Subscriber('/odom_0', Odometry, self.callback_0)
        self.odom_sub_1 = rospy.Subscriber('/odom_1', Odometry, self.callback_1)
        self.odom_sub_2 = rospy.Subscriber('/odom_2', Odometry, self.callback_2)

        self.cmd_pub_0 = rospy.Publisher('/cmd_0', PositionCommand, queue_size=10)
        self.cmd_pub_1 = rospy.Publisher('/cmd_1', PositionCommand, queue_size=10)
        self.cmd_pub_2 = rospy.Publisher('/cmd_2', PositionCommand, queue_size=10)


    def callback_0(self, odom_data):
        self.cur_state[0] = odom_data.pose.pose.position.x
        self.cur_state[1] = odom_data.pose.pose.position.y

    def callback_1(self, odom_data):
        self.cur_state[2] = odom_data.pose.pose.position.x
        self.cur_state[3] = odom_data.pose.pose.position.y

    def callback_2(self, odom_data):
        self.cur_state[4] = odom_data.pose.pose.position.x
        self.cur_state[5] = odom_data.pose.pose.position.y

    def run(self):
        goal_point = np.array([10, 10])
        goal_radius = 0.3
        obs_point = np.array([1, 1])
        obs_radius = 0.3
        Fix_T = 5
        self.controller.setup(goal_point, goal_radius, False, obs_point, obs_radius)

        Fixed_z = 2
        dt = 0.2
        opt_inputs = self.controller.solve(self.cur_state, self.form_topo, Fix_T)

        cmd_0 = PositionCommand()
        cmd_0.position.x = self.cur_state[0]+ opt_inputs[0] * dt
        cmd_0.position.y = self.cur_state[1]+ opt_inputs[1] * dt
        cmd_0.position.z = Fixed_z
        # cmd_0.velocity.x = float('nan')
        # cmd_0.velocity.y = float('nan')
        # cmd_0.velocity.z = float('nan')
        # cmd_0.acceleration.x = float('nan')
        # cmd_0.acceleration.y = float('nan')
        # cmd_0.acceleration.z = float('nan')
        self.cmd_pub_0.publish(cmd_0)

        cmd_1 = PositionCommand()
        cmd_1.position.x = self.cur_state[2] + opt_inputs[2] * dt
        cmd_1.position.y = self.cur_state[3] + opt_inputs[3] * dt
        cmd_1.position.z = Fixed_z
        self.cmd_pub_1.publish(cmd_1)

        cmd_2 = PositionCommand()
        cmd_2.position.x = self.cur_state[4] + opt_inputs[4] * dt
        cmd_2.position.y = self.cur_state[5] + opt_inputs[5] * dt
        cmd_2.position.z = Fixed_z
        self.cmd_pub_2.publish(cmd_2)
        print("Current states:", self.cur_state)


if __name__ == "__main__":
    node = SwarmNode()
    rospy.sleep(5)
    while not rospy.is_shutdown():
        node.run()
        rospy.sleep(0.1)



