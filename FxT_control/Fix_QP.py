import casadi as ca
import numpy as np
import matplotlib.pyplot as plt


def FxT_QP(T_fix, cur_state, goal_point, goal_radius, obs_point, obs_radius):

    agent_num = 1
    x_dim = 2 * agent_num
    u_dim = 2 * agent_num
    r_dim = 2

    max_u = 3

    mu = 1.5
    gamma_1 = 1 + 1 / mu
    gamma_2 = 1 - 1 / mu
    alpha_1 = mu * ca.pi / (2 * T_fix)
    alpha_2 = mu * ca.pi / (2 * T_fix)

    opti = ca.Opti()

    U = opti.variable(u_dim)
    R = opti.variable(r_dim)
    X = opti.parameter(x_dim)

    Q = ca.diag([0.001, 0.001, 20, 1])
    F = ca.DM([0, 0, 1, 0])

    Z = ca.vertcat(U, R)
    costs = ca.trace(Z.T @ Q @ Z) + F.T @ Z
    opti.minimize(costs) # Note R[0] should be non-positive to make CLF satisfied

    opti.subject_to(U[0] <= max_u)
    opti.subject_to(U[0] >= -max_u)
    opti.subject_to(U[1] <= max_u)
    opti.subject_to(U[1] >= -max_u)

    # Fixed-time CLF
    h_g = (X[0]-goal_point[0])**2 + (X[1]-goal_point[1])**2 - goal_radius**2
    d_h_g = 2 * (X[0]-goal_point[0]) * U[0] + 2 * (X[1]-goal_point[1]) * U[1]
    opti.subject_to(d_h_g <= R[0] * h_g - alpha_1 * ca.fmax(0, h_g)**gamma_1 - alpha_2 * ca.fmax(0, h_g)**gamma_2)

    # CBF
    h_s = obs_radius**2 - (X[0]-obs_point[0])**2 - (X[1]-obs_point[1])**2
    d_h_s = - 2 * (X[0]-obs_point[0]) * U[0] - 2 * (X[1]-obs_point[1]) * U[1]
    opti.subject_to(d_h_s <= - R[0] * h_s)

    cur_state = ca.DM(cur_state)
    opti.set_value(X, cur_state)

    ipopt_options = {
        'verbose': False,
        "ipopt.tol": 1e-4,
        "ipopt.acceptable_tol": 1e-4,
        "ipopt.max_iter": 100,
        "ipopt.warm_start_init_point": "yes",
        "print_time": True
    }
    opti.solver('ipopt', ipopt_options)
    sol = opti.solve()
    print("\n Optimal Input: \n", sol.value(U))

    return sol.value(U)

class simulator():

    def __init__(self, dt):
        self.dt = dt
        self.states = np.zeros(2)
        self.states_traj = np.array([self.states])

    def update(self, input):
        self.states += self.dt * np.array([input[0], input[1]])
        self.states_traj = np.append(self.states_traj, np.array([self.states]), axis=0)


if __name__ == "__main__":

    T = 0
    dt = 0.05
    sim = simulator(dt)

    goal_point = np.array([3, 3])
    goal_radius = 0.2
    obs_point = np.array([1, 1])
    obs_radius = 0.2
    Fix_T = 4

    plt.figure()
    while T <= (Fix_T * 1.5):
        opt_inputs = FxT_QP(Fix_T, sim.states, goal_point, goal_radius, obs_point, obs_radius)
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
