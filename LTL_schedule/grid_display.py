import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d

try:
    from LTL_schedule.ltl_ctrl import TaskSchedule
except ImportError:
    from ltl_ctrl import TaskSchedule

# from reactive_ltl_ctrl import TaskSchedule


class GridDisplay:

    def __init__(self, space, unit_size, area_labels, form_labels):
        self.space = space
        self.unit_size = unit_size
        self.area_labels = area_labels
        self.form_labels = form_labels

        fig = plt.figure()
        self.ax = fig.add_subplot(projection='3d')
        self.ltl_tasks = TaskSchedule(self.space, self.unit_size, self.area_labels, self.form_labels)
        self.ltl_tasks.setup()

        self.x_from = None
        self.x_to = None
        self.cur_quiver = None

    def plot_TLgraph3d(self, traj_show):
        self.ax.set_xlim3d(0, self.space[0])
        self.ax.set_ylim3d(0, self.space[1])
        self.ax.set_zlim3d(0, len(self.form_labels))

        ticks = np.arange(0, self.space[0], self.unit_size[0])
        self.ax.set_xticks(ticks)
        self.ax.set_yticks(ticks)
        self.ax.set_xlabel('X self.axis')
        self.ax.set_ylabel('Y self.axis')
        self.ax.set_zlabel('Formation')
        self.ax.text(0, 0, 0.2, "horizon", color='black')
        self.ax.text(0, 0, 1.2, "vertical", color='black')
        self.ax.text(0, 0, 2.2, "triangle", color='black')

        color_dict = {'obstacle': 'r', 'home': 'y', 'bench': 'b'}
        for i in range(3):
            x = np.arange(0, self.space[0]+1, 1)
            y = np.arange(0, self.space[1]+1, 1)
            X, Y = np.meshgrid(x, y)
            Z = i * np.ones_like(X)
            self.ax.plot_surface(X, Y, Z, color='white', alpha=0.3, edgecolor='grey')
            for label, area in zip(self.area_labels.values(), self.area_labels.keys()):
                rect = plt.Rectangle((area[0][0], area[1][0]), area[0][1]-area[0][0],
                                     area[1][1]-area[1][0], color=color_dict[label], alpha=0.5)
                self.ax.add_patch(rect)
                art3d.pathpatch_2d_to_3d(rect, z=i, zdir="z")
        if traj_show:
            for t in self.ltl_tasks.transitions:
                X_from = self.ltl_tasks.state_dict[t[0]]
                X_to = self.ltl_tasks.state_dict[t[1]]
                x_from = int(X_from[1:-3])
                x_to = int(X_to[1:-3])
                p_from = np.array(self.ltl_tasks.sampling.get_pos(x_from))
                p_to = np.array(self.ltl_tasks.sampling.get_pos(x_to))
                z_from = int(X_from[-1])
                z_to = int(X_to[-1])
                segment = np.vstack((np.array([p_from[0], p_from[1], z_from]), np.array([p_to[0], p_to[1], z_to])))
                plt.plot(segment[:, 0], segment[:, 1], segment[:, 2] + 0.1, '-', color='grey')
        plt.grid()
        plt.pause(0.1)
        # plt.show()

    def update(self, X_from, X_to):
        if (self.x_from != X_from) or (self.x_to != X_to):
            self.x_from = X_from
            self.x_to = X_to
            if self.cur_quiver is not None:
                self.cur_quiver.remove()
            x_from = int(X_from[1:-3])
            x_to = int(X_to[1:-3])
            p_from = np.array(self.ltl_tasks.sampling.get_pos(x_from))
            p_to = np.array(self.ltl_tasks.sampling.get_pos(x_to))
            z_from = int(X_from[-1])
            z_to = int(X_to[-1])
            l = p_to - p_from
            self.cur_quiver = self.ax.quiver(p_from[0], p_from[1], z_from + 0.1, l[0], l[1], z_to - z_from,
                                             color='r', arrow_length_ratio=0.4)
            plt.plot(p_from[0], p_from[1], z_from + 0.1, 'o', color='r')
            segment = np.vstack((np.array([p_from[0], p_from[1], z_from]), np.array([p_to[0], p_to[1], z_to])))
            plt.plot(segment[:, 0], segment[:, 1], segment[:, 2] + 0.1, '-', color=(0.1, 0.1, 0.8, 1.0))
            plt.pause(0.005)


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
    sim = GridDisplay(space, unit, area_dict, form_dict)
    sim.plot_TLgraph3d(True)
    sim.update('X0_f0', 'X1_f2')
    sim.update('X0_f0', 'X2_f2')

    # ltl_tasks.get_stateDict()
    # print(ltl_tasks.ctrl)

    # n = 0
    # cur_index = 'Sinit'
    # while n < 30:
    #     next_index, next_state = ltl_tasks.run(cur_index, change_signal)
    #     cur_index = next_index
    #     print(str(n) + "Next_state: ", next_state)
    #     n += 1