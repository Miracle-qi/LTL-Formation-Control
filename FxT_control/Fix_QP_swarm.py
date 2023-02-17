import casadi as ca
import numpy as np
import matplotlib.pyplot as plt


class FxT_QP():
    def __int__(self, num=3):
        x_dim = 2
        u_dim = 2
        r_dim = 3
        self.agents_num = num

        self.opti = ca.Opti()
        self.U = self.opti.variable(u_dim * num)
        self.R = self.opti.variable(r_dim)
        self.X = self.opti.parameter(x_dim * num)
        self.alpha_1 = self.opti.parameter(1)
        self.alpha_2 = self.opti.parameter(1)
        self.form_topo = self.opti.parameter(num, 2)

        self.Q = ca.diag([0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 20, 20, 1])
        self.F = ca.DM([0, 0, 0, 0, 0, 0, 1, 1, 0])
        self.r_accept = 0.2
        self.mu = 1.5
        self.max_u = 3

    def setup(self, goal_point, goal_radius, obs_point, obs_radius):
        gamma_1 = 1 + 1 / self.mu
        gamma_2 = 1 - 1 / self.mu

        Z = ca.vertcat(self.U, self.R)
        costs = ca.trace(Z.T @ self.Q @ Z) + self.F.T @ Z
        self.opti.minimize(costs) # Note R[0] should be non-positive to make CLF satisfied

        self.opti.subject_to(self.U[:] <= max_u)
        self.opti.subject_to(self.U[:] >= -max_u)

        # Fixed-time CLF for Reaching
        x_c = (self.X[0] + self.X[2] + self.X[4]) / 3
        y_c = (self.X[1] + self.X[3] + self.X[5]) / 3
        v_x_c = (self.U[0] + self.U[2] + self.U[4]) / 3
        v_y_c = (self.U[1] + self.U[3] + self.U[5]) / 3
        h_g = (x_c-goal_point[0])**2 + (y_c-goal_point[1])**2 - goal_radius**2
        d_h_g = 2 * (x_c-goal_point[0]) * v_x_c + 2 * (y_c-goal_point[1]) * v_y_c
        self.opti.subject_to(d_h_g <= self.R[0] * h_g - self.alpha_1 * ca.fmax(0, h_g)**gamma_1
                             - self.alpha_2 * ca.fmax(0, h_g)**gamma_2)

        # Fixed-time CLF for Formation
        for n in range(self.agents_num):
            _x = self.X[2*(n+1)] - self.X[2*n]
            _y = self.X[2*(n+1)+1] - self.X[2*n+1]
            _vx = self.U[2*(n+1)] - self.U[2*n]
            _vy = self.U[2*(n+1)+1] - self.U[2*n+1]
            _h = (_x - self.form_topo[n, 0])**2 + (_y - self.form_topo[n, 1])**2 - self.r_accept**2
            _dh = 2 * (_x - self.form_topo[n, 0]) * _vx + 2 * (_y - self.form_topo[n, 1]) * _vy
            self.opti.subject_to(_dh <= self.R[1] * _h - self.alpha_1 * ca.fmax(0, _h)**gamma_1
                                 - self.alpha_2 * ca.fmax(0, _h)**gamma_2)


        # CBF for Obstacle Avoidance
        # h_s = obs_radius**2 - (X[0]-obs_point[0])**2 - (X[1]-obs_point[1])**2
        # d_h_s = - 2 * (X[0]-obs_point[0]) * U[0] - 2 * (X[1]-obs_point[1]) * U[1]
        # self.opti.subject_to(d_h_s <= - R[0] * h_s)

        ipopt_options = {
            'verbose': False,
            "ipopt.tol": 1e-4,
            "ipopt.acceptable_tol": 1e-4,
            "ipopt.max_iter": 100,
            "ipopt.warm_start_init_point": "yes",
            "print_time": True
        }
        self.opti.solver('ipopt', ipopt_options)

    def solve(self, cur_state, form_topo, T_fix):
        cur_state = ca.DM(cur_state)
        self.opti.set_value(self.X, cur_state)
        self.opti.set_value(self.alpha_1, self.mu * ca.pi / (2 * T_fix))
        self.opti.set_value(self.alpha_2, self.mu * ca.pi / (2 * T_fix))
        self.opti.set_value(self.form_topo, form_topo)

        sol = self.opti.solve()
        print("\n self.optimal Input: \n", sol.value(self.U))

        return sol.value(self.U)


def formation(form_name):

    form_list = {'horizon': [[1, 0], [1, 0], [1, 0]],
                 'vertical': [[0, 1], [0, 1], [0, 1]],
                 'triangle': [[1, 0], [-0.5, 1], [-0.5, -1]]}

    return form_list[form_name]


class simulator():

    def __init__(self, dt, num):
        self.dt = dt
        self.agents_num = num
        self.states = np.zeros(2 * num)
        self.states_traj = np.array([self.states])

    def update(self, input):
        self.states += self.dt * np.array([input])
        self.states_traj = np.append(self.states_traj, np.reshape([self.states]), axis=0)


if __name__ == "__main__":
    T = 0
    dt = 0.05
    agents_num = 3
    sim = simulator(dt, agents_num)
    controller = FxT_QP()

    goal_point = np.array([3, 3])
    goal_radius = 0.2
    obs_point = np.array([1, 1])
    obs_radius = 0.2
    controller.setup(goal_point, goal_radius, obs_point, obs_radius)

    form_topo = formation('triangle')
    Fix_T = 4

    plt.figure()
    while T <= (Fix_T * 1.5):
        opt_inputs = controller.solve(sim.states, form_topo, Fix_T)
        sim.update(opt_inputs)
        # Todo: print R[0] R[1]

        circle1 = plt.Circle(goal_point, radius=goal_radius, color='b')
        circle2 = plt.Circle(obs_point, radius=obs_radius, color='r')
        plt.gca().add_patch(circle1)
        plt.gca().add_patch(circle2)
        plt.xlim((-1, 4))
        plt.ylim((-1, 4))
        plt.plot(sim.states_traj[:, 0], sim.states_traj[:, 1], 'g')

        font = {'color': 'black', 'size': 14}
        text = "Fixed Time: " + str(Fix_T) + " Real Time: " + str(round(T, 2))
        plt.text(0.0, 3.5, text, fontdict=font)
        plt.pause(dt)
        plt.clf()
        T += dt
    plt.show()
