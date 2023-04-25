import casadi as ca
import numpy as np
import matplotlib.pyplot as plt



class FxT_QP_swarm:
    def __init__(self, num=3):
        self.x_dim = 2
        self.u_dim = 2
        self.r_dim = 3
        self.agents_num = num
        self.mu = 1.5
        self.max_u = 3

        self.opti = ca.Opti()
        self.U = self.opti.variable(self.u_dim * num)
        self.R = self.opti.variable(self.r_dim)
        self.X = self.opti.parameter(self.x_dim * num)
        self.V = self.opti.parameter(self.x_dim * num)
        self.alpha_1 = self.opti.parameter(1)
        self.alpha_2 = self.opti.parameter(1)
        self.form_topo = self.opti.parameter(num, 2)

        self.Q = ca.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 10, 1, 0.01])
        self.F = ca.DM([0, 0, 0, 0, 0, 0, 0.01, 0.01, 0])
        self.M = ca.diag(np.ones(self.u_dim * num) * 0.1)
        self.r_accept = 0.05

    def setup(self, goal_point, goal_radius, obs_flag, obs_point=(0., 0.), obs_radius=0.1):

        self.opti = ca.Opti()
        self.U = self.opti.variable(self.u_dim * self.agents_num)
        self.R = self.opti.variable(self.r_dim)
        self.X = self.opti.parameter(self.x_dim * self.agents_num)
        self.V = self.opti.parameter(self.x_dim * self.agents_num)
        self.alpha_1 = self.opti.parameter(1)
        self.alpha_2 = self.opti.parameter(1)
        self.form_topo = self.opti.parameter(self.agents_num, 2)

        gamma_1 = 1 + 1 / self.mu
        gamma_2 = 1 - 1 / self.mu

        Z = ca.vertcat(self.U, self.R)
        costs = ca.trace(Z.T @ self.Q @ Z) + self.F.T @ Z + ca.trace((self.U-self.V).T @ self.M @ (self.U-self.V))
        self.opti.minimize(costs)  # Note R[0] should be non-positive to make CLF satisfied

        self.opti.subject_to(self.U[:] <= self.max_u)
        self.opti.subject_to(self.U[:] >= -self.max_u)

        # Fixed-time CLF for Reaching
        x_c = (self.X[0] + self.X[2] + self.X[4]) / 3
        y_c = (self.X[1] + self.X[3] + self.X[5]) / 3
        v_x_c = (self.U[0] + self.U[2] + self.U[4]) / 3
        v_y_c = (self.U[1] + self.U[3] + self.U[5]) / 3
        h_g = (x_c - goal_point[0]) ** 2 + (y_c - goal_point[1]) ** 2 - goal_radius ** 2
        # h_g = (x_c - goal_point[0]) ** 2 + (y_c - goal_point[1]) ** 2
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

            min_dist = 0.5
            # CBF for Collision Aviodance
            _hc = min_dist ** 2 - _x ** 2 - _y ** 2
            _d_hc = - 2 * _x * _vx - 2 * _y * _vy
            self.opti.subject_to(_d_hc <= - self.R[2] * _hc)

            # CBF for Static Obstacle Avoidance
            if obs_flag:
                _hs = obs_radius**2 - (self.X[2*n]-obs_point[0])**2 - (self.X[2*n]-obs_point[1])**2
                _d_hs = - 2 * (self.X[2*n]-obs_point[0]) * self.U[2*n] - 2 * (self.X[2*n+1]-obs_point[1]) * self.U[2*n+1]
                self.opti.subject_to(_d_hs <= - self.R[2] * _hs)

        ipopt_options = {
            'verbose': False,
            "ipopt.tol": 1e-6,
            "ipopt.acceptable_tol": 1e-6,
            "ipopt.max_iter": 200,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.print_level": 0,
            "print_time": True
        }
        self.opti.solver('ipopt', ipopt_options)

    def solve(self, cur_pos, cur_vel, form_topo, T_fix):
        cur_pos = ca.DM(cur_pos)
        cur_vel = ca.DM(cur_vel)
        self.opti.set_value(self.X, cur_pos)
        self.opti.set_value(self.V, cur_vel)
        self.opti.set_value(self.alpha_1, self.mu * ca.pi / (2 * T_fix))
        self.opti.set_value(self.alpha_2, self.mu * ca.pi / (2 * T_fix))
        self.opti.set_value(self.form_topo, np.array(form_topo))

        sol = self.opti.solve()
        print("\n self.optimal Input: \n", sol.value(self.U))

        return sol.value(self.U)


def formation(form_name):
    form_topo = {'horizon': [[0.7, 0], [0.7, 0], [-1.4, 0]],
                 'vertical': [[0, 0.7], [0, 0.7], [0, -1.4]],
                 'triangle': [[1, 0], [-0.5, 0.8], [-0.5, -0.8]]}
    return form_topo[form_name]


class simulator:

    def __init__(self, dt, num):
        self.dt = dt
        self.agents_num = num
        self.states = np.zeros(2 * num)
        self.velocity = np.zeros(2 * num)
        self.states_traj = np.array([self.states])

    def update(self, input):
        self.velocity = input
        self.states += self.dt * np.array(input)
        self.states_traj = np.append(self.states_traj, np.array([self.states]), axis=0)

if __name__ == "__main__":
    T = 0
    dt = 0.03
    agents_num = 3
    sim = simulator(dt, agents_num)
    controller = FxT_QP_swarm()

    goal_point = np.array([2.5, 2.5])
    goal_radius = 0.15
    obs_point = np.array([1, 1])
    obs_radius = 0.3
    controller.setup(goal_point, goal_radius, True, obs_point, obs_radius)
    form_topo = formation('triangle')
    Fix_T = 4

    Animation_flag = True

    plt.figure()
    # plt.pause(10)

    # Display Setting
    lin_color = [0.2, 0.2, 0.8, 0.5]
    form_color = [0, 0, 0.8, 0.01]
    _time = [0.5, 1.0, 2.0, 3.0, 4.0]
    marker_time = [int(x/dt) for x in _time]
    his_shape = []

    while T <= (Fix_T * 1.5):
        form_color[3] = min(1, form_color[3] + dt * 1 / Fix_T)
        opt_inputs = controller.solve(sim.states, sim.velocity, form_topo, Fix_T)
        sim.update(opt_inputs)
        if Animation_flag:
            circle1 = plt.Circle(goal_point, radius=goal_radius, color="#9ACD32")
            circle2 = plt.Circle(obs_point, radius=obs_radius, color="#EE4000")
            plt.gca().add_patch(circle1)
            plt.gca().add_patch(circle2)
            plt.xlim((-0.5, 3.5))
            plt.ylim((-0.5, 3.5))
            plt.plot(sim.states_traj[:, 0], sim.states_traj[:, 1], color=lin_color)
            plt.plot(sim.states_traj[:, 2], sim.states_traj[:, 3], color=lin_color)
            plt.plot(sim.states_traj[:, 4], sim.states_traj[:, 5], color=lin_color)
            plt.plot([sim.states[0], sim.states[2], sim.states[4], sim.states[0]],
                     [sim.states[1], sim.states[3], sim.states[5], sim.states[1]], 'o--', color=lin_color)
            print(sim.states, sim.states[0:-1:2], sim.states[1:-1:2])
            plt.plot(sum(sim.states[0::2])/3, sum(sim.states[1::2])/3, '2', color=lin_color, markersize=10)

            print(int(100 * T), int(100 * dt))
            if int(T/dt) in marker_time:
                his_shape.append([[sim.states[0], sim.states[2], sim.states[4], sim.states[0]],
                                  [sim.states[1], sim.states[3], sim.states[5], sim.states[1]]])
            for s in his_shape:
                plt.plot(s[0], s[1], color=form_color, linestyle='dotted', linewidth=2)

            font = {'color': 'black', 'size': 14}
            text = "Fixed Time: " + str(Fix_T) + " Real Time: " + str(round(T, 2))
            plt.text(-0.3, 2.5, text, fontdict=font)
            plt.pause(dt)
            plt.clf()
        T += dt
    np.savetxt("./traj.txt", sim.states_traj)
    # plt.show()
