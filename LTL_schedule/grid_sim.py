import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
from ltl_ctrl import TaskSchedule
# from reactive_ltl_ctrl import TaskSchedule


class GridSim:

    def __init__(self, dt, num, space, unit_size, area_labels, form_labels):

        self.dt = dt
        self.agents_num = num
        self.states = np.zeros(2 * num)
        self.states_traj = np.array([self.states])

        self.space = space
        self.unit_size = unit_size
        self.area_labels = area_labels
        self.form_labels = form_labels

    def plot_map(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        color_dict = {'obstacle': 'r', 'home': 'y', 'bench': 'b'}
        for label, area in zip(self.area_labels.values(), self.area_labels.keys()):
            rect = plt.Rectangle((area[0][0], area[1][0]), area[0][1]-area[0][0],
                                 area[1][1]-area[1][0], color=color_dict[label])
            plt.gca().add_patch(rect)

        ticks = np.arange(0, self.space[0], self.unit_size[0])
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

        plt.xlim((0, self.space[0]))
        plt.ylim((0, self.space[1]))

        plt.grid()
        plt.show()
        plt.pause(0.05)
        plt.clf()

    def plot_TLgraph(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        color_dict = {'obstacle': 'r', 'home': 'y', 'bench': 'b'}
        for label, area in zip(self.area_labels.values(), self.area_labels.keys()):
            rect = plt.Rectangle((area[0][0], area[1][0]), area[0][1]-area[0][0],
                                 area[1][1]-area[1][0], color=color_dict[label])
            plt.gca().add_patch(rect)

        ticks = np.arange(0, self.space[0], self.unit_size[0])
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

        plt.xlim((0, self.space[0]))
        plt.ylim((0, self.space[1]))
        ltl_tasks = TaskSchedule(self.space, self.unit_size, self.area_labels, self.form_labels)
        ltl_tasks.setup()

        for t in ltl_tasks.transitions:
            x_from = int(ltl_tasks.state_dict[t[0]][1:-3])
            x_to = int(ltl_tasks.state_dict[t[1]][1:-3])
            p_from = np.array(ltl_tasks.sampling.get_pos(x_from))
            p_to = np.array(ltl_tasks.sampling.get_pos(x_to))
            l = p_to - p_from
            ax.quiver(p_from[0], p_from[1], l[0], l[1], color='black', angles='xy', scale_units='xy', scale=1)
            plt.plot(p_from[0], p_from[1], 'o')
        plt.grid()
        plt.show()
        plt.pause(0.05)
        plt.clf()

    def plot_TLgraph3d(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        ax.set_xlim3d(0, self.space[0])
        ax.set_ylim3d(0, self.space[1])
        ax.set_zlim3d(0, len(self.form_labels))

        color_dict = {'obstacle': 'r', 'home': 'y', 'bench': 'b'}
        for i in range(3):
            x = np.arange(0, self.space[0]+1, 1)
            y = np.arange(0, self.space[1]+1, 1)
            X, Y = np.meshgrid(x, y)
            Z = i * np.ones_like(X)
            ax.plot_surface(X, Y, Z, color='white', alpha=0.3, edgecolor='grey')
            for label, area in zip(self.area_labels.values(), self.area_labels.keys()):
                rect = plt.Rectangle((area[0][0], area[1][0]), area[0][1]-area[0][0],
                                     area[1][1]-area[1][0], color=color_dict[label], alpha=0.5)
                ax.add_patch(rect)
                art3d.pathpatch_2d_to_3d(rect, z=i, zdir="z")

        ltl_tasks = TaskSchedule(self.space, self.unit_size, self.area_labels, self.form_labels)
        ltl_tasks.setup()

        for t in ltl_tasks.transitions:
            X_from = ltl_tasks.state_dict[t[0]]
            X_to = ltl_tasks.state_dict[t[1]]
            x_from = int(X_from[1:-3])
            x_to = int(X_to[1:-3])
            p_from = np.array(ltl_tasks.sampling.get_pos(x_from))
            p_to = np.array(ltl_tasks.sampling.get_pos(x_to))
            z_from = int(X_from[-1])
            z_to = int(X_to[-1])
            l = p_to - p_from
            ax.quiver(p_from[0], p_from[1], z_from+0.1, l[0], l[1], z_to-z_from, arrow_length_ratio=0.15)
            plt.plot(p_from[0], p_from[1], z_from+0.1, 'o', color='r')

        ticks = np.arange(0, self.space[0], self.unit_size[0])
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        _s = 12
        ax.set_xlabel('X axis', fontsize= _s)
        ax.set_ylabel('Y axis', fontsize= _s)
        ax.set_zlabel('Formation', fontsize= _s)
        ax.text(0, 0, 0.5, "horizon", color='black', fontsize= _s+3)
        ax.text(0, 0, 1.5, "vertical", color='black', fontsize= _s+3)
        ax.text(0, 0, 2.5, "triangle", color='black', fontsize= _s+3)
        plt.grid()
        plt.show()
        plt.savefig('Figures/1.eps', format='eps')


    def plot_traj(self):
        plt.plot(self.states_traj[:, 0], self.states_traj[:, 1], 'g')
        plt.plot(self.states_traj[:, 2], self.states_traj[:, 3], 'g')
        plt.plot(self.states_traj[:, 4], self.states_traj[:, 5], 'g')

        plt.plot([self.states[0], self.states[2], self.states[4], self.states[0]],
                 [self.states[1], self.states[3], self.states[5], self.states[1]], 'g', linestyle = 'dotted')

    def update(self, input):
        self.states += self.dt * np.array(input)
        self.states_traj = np.append(self.states_traj, np.array([self.states]), axis=0)


if __name__ == "__main__":
    dt = 0.1
    agent_num = 3
    space = (5, 5) # The origin is (0,0) by default
    unit = (1, 1)
    area_dict = {((0, 1), (0, 1)): 'home',
                 ((3, 4), (3, 4)): 'bench',
                 ((1, 2), (2, 4)): 'obstacle',
                 ((1, 4), (1, 2)): 'obstacle'}
    form_dict = {'horizon':  (2, 1, 1),
                 'vertical': (1, 2, 1),
                 'triangle': (2, 2, 2)}
    sim = GridSim(dt, agent_num, space, unit, area_dict, form_dict)
    # sim.plot_map()
    sim.plot_TLgraph3d()
