import numpy as np
import matplotlib.pyplot as plt
from LTL_schedule.dis_task import TaskSchedule
from FxT_control.Fix_QP_swarm import FxT_QP_swarm, simulator


def formation(form_name):
    form_list = {'horizon': [[1, 0], [1, 0], [-2, 0]],
                 'vertical': [[0, 1], [0, 1], [0, -2]],
                 'triangle': [[1, 0], [-0.5, 1], [-0.5, -1]]}

    return form_list[form_name]


class task_info:

    def __init__(self):
        self.task_list = {'X0': {'goal_point': [2, 2], 'formation': 'vertical'},
                          'X1': {'goal_point': [2, 4], 'formation': 'horizon'},
                          'X2': {'goal_point': [4, 4], 'formation': 'horizon'},
                          'X3': {'goal_point': [6, 4], 'formation': 'horizon'},
                          'X4': {'goal_point': [6, 2], 'formation': 'triangle'}}
    def find_task(self, next_state):
        return self.task_list[next_state]

    def find_cur_state(self, cur_center):
        for state in self.task_list:
            _p = state[goal_point]
            cur_center





if __name__ == "__main__":

    ltl_tasks = TaskSchedule()
    ltl_tasks.setup()
    ltl_tasks.record()
    change_signal = True

    T = 0
    dt = 0.05
    agents_num = 3
    sim = simulator(dt, agents_num)
    controller = FxT_QP_swarm()


    Fix_T = 5

    plt.figure()
    cur_state = 0
    # plt.pause(5)
    while True:

        next_state = ltl_tasks.run(cur_state, change_signal)
        next_task = tasks(next_state)

        goal_point = next_task['goal_point']
        goal_radius = 0.1
        obs_point = np.array([1, 1])
        obs_radius = 0.2
        controller.setup(goal_point, goal_radius, obs_point, obs_radius)

        form_topo = formation(next_task['formation'])


        opt_inputs = controller.solve(sim.states, form_topo, Fix_T)
        sim.update(opt_inputs)
        # Todo: print R[0] R[1]

        circle1 = plt.Circle(goal_point, radius=goal_radius, color='b')
        circle2 = plt.Circle(obs_point, radius=obs_radius, color='r')
        plt.gca().add_patch(circle1)
        plt.gca().add_patch(circle2)
        plt.xlim((-1, 7))
        plt.ylim((-1, 7))
        plt.plot(sim.states_traj[:, 0], sim.states_traj[:, 1], 'g')
        plt.plot(sim.states_traj[:, 2], sim.states_traj[:, 3], 'g')
        plt.plot(sim.states_traj[:, 4], sim.states_traj[:, 5], 'g')

        plt.plot([sim.states[0], sim.states[2], sim.states[4], sim.states[0]],
                 [sim.states[1], sim.states[3], sim.states[5], sim.states[1]], 'g', linestyle = 'dotted')

        font = {'color': 'black', 'size': 14}
        text = "Fixed Time: " + str(Fix_T) + " Real Time: " + str(round(T, 2))
        plt.text(0.0, 3.5, text, fontdict=font)
        plt.pause(dt)
        plt.clf()


        T += dt
    plt.show()

