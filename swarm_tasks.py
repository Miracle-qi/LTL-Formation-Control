import numpy as np
import matplotlib.pyplot as plt
from LTL_schedule.dis_task import TaskSchedule
from FxT_control.Fix_QP_swarm import FxT_QP_swarm, simulator


def formation(form_name):
    form_list = {'horizon': [[0.5, 0], [0.5, 0], [-1, 0]],
                 'vertical': [[0, 0.5], [0, 0.5], [0, -1]],
                 'triangle': [[1, 0], [-0.5, 1], [-0.5, -1]]}

    return form_list[form_name]


class TaskInfo:

    def __init__(self):
        self.task_list = {'X0': {'goal_point': [1, 1], 'formation': 'vertical'},
                          'X1': {'goal_point': [2, 4], 'formation': 'triangle'},
                          'X2': {'goal_point': [2.5, 4], 'formation': 'horizon'},
                          'X3': {'goal_point': [5, 4], 'formation': 'horizon'},
                          'X4': {'goal_point': [7, 2], 'formation': 'triangle'}}
        self.cur_state = None
        self.goal_state = None

    def find_next_task(self, next_state):
        self.goal_state = next_state
        return self.task_list[next_state]

    def change_cur_state(self, cur_center):
        _p = self.task_list[self.goal_state]['goal_point']
        dist = np.hypot(cur_center[0]-_p[0], cur_center[1]-_p[1])
        if dist <= 0.3:
            self.cur_state = self.goal_state
            return True
        else:
            return False


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

    Fix_T = 6
    plt.figure()
    cur_state = 0
    task = TaskInfo()
    # plt.pause(5)
    while True:
        next_state, next_loc = ltl_tasks.run(cur_state, change_signal)
        next_task = task.find_next_task(next_loc)
        goal_point = next_task['goal_point']
        print("goal_point: ", goal_point)
        goal_radius = 0.2
        cur_center = [(sim.states[0] + sim.states[2] + sim.states[4]) / 3,
                      (sim.states[1] + sim.states[3] + sim.states[5]) / 3]
        if task.change_cur_state(cur_center):
            cur_state = next_state
        controller.setup(goal_point, goal_radius, False)

        form_topo = formation(next_task['formation'])
        print("form_topo: ", form_topo)
        print("states: ", sim.states)
        opt_inputs = controller.solve(sim.states, form_topo, Fix_T)
        sim.update(opt_inputs)
        # Todo: print R[0] R[1]

        circle1 = plt.Circle(goal_point, radius=goal_radius, color='b')
        # circle2 = plt.Circle(obs_point, radius=obs_radius, color='r')
        plt.gca().add_patch(circle1)
        # plt.gca().add_patch(circle2)
        plt.xlim((-1, 7))
        plt.ylim((-1, 7))
        plt.plot(sim.states_traj[:, 0], sim.states_traj[:, 1], 'g')
        plt.plot(sim.states_traj[:, 2], sim.states_traj[:, 3], 'g')
        plt.plot(sim.states_traj[:, 4], sim.states_traj[:, 5], 'g')

        plt.plot([sim.states[0], sim.states[2], sim.states[4], sim.states[0]],
                 [sim.states[1], sim.states[3], sim.states[5], sim.states[1]], 'g', linestyle = 'dotted')

        font = {'color': 'black', 'size': 14}
        text = "Fixed Time: " + str(Fix_T) + " Real Time: " + str(round(T, 2))
        plt.text(0.0, 6, text, fontdict=font)
        plt.pause(dt)
        plt.clf()
        T += dt
    plt.show()

