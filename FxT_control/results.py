import numpy as np
import matplotlib.pyplot as plt

dt = 0.01
agent_num = 3
agent_dim = 2
form_topo = {'horizon': [[0.7, 0], [0.7, 0], [-1.4, 0]],
             'vertical': [[0, 0.7], [0, 0.7], [0, -1.4]],
             'triangle': [[1, 0], [-0.5, 0.8], [-0.5, -0.8]]}

traj = np.loadtxt("traj.txt", delimiter=' ')
plt.figure()
step = 5
fix_t = 4
topo = form_topo['triangle']
goal_point = (2.5, 2.5)
traj_len = int((fix_t + 0.5)/dt)
t = np.array(range(0, traj_len, step)) * dt
goal_radius = 0.2
g_max = np.ones(len(t)) * goal_radius
g_min = -np.ones(len(t)) * goal_radius
plt.fill_between(t, g_max, g_min, where=g_max >= g_min, facecolor='#87CEEB', interpolate=True, alpha=0.4)

form_radius = 0.1
f_max = np.ones(len(t)) * form_radius
f_min = -np.ones(len(t)) * form_radius
plt.fill_between(t, f_max, f_min, where=f_max >= f_min, facecolor='#00FF7F', interpolate=True, alpha=0.7)

x_c = [sum(traj[i, 0::2]) / 3 - goal_point[0] for i in range(0, traj_len, step)]
y_c = [sum(traj[i, 1::2]) / 3 - goal_point[1] for i in range(0, traj_len, step)]
x_01 = [traj[i, 2] - traj[i, 0] - topo[0][0] for i in range(0, traj_len, step)]
y_01 = [traj[i, 3] - traj[i, 1] - topo[0][1] for i in range(0, traj_len, step)]
x_12 = [traj[i, 4] - traj[i, 2] - topo[1][0] for i in range(0, traj_len, step)]
y_12 = [traj[i, 5] - traj[i, 3] - topo[1][1] for i in range(0, traj_len, step)]
x_20 = [traj[i, 0] - traj[i, 4] - topo[2][0] for i in range(0, traj_len, step)]
y_20 = [traj[i, 1] - traj[i, 5] - topo[2][1] for i in range(0, traj_len, step)]


plt.plot(t, x_c, color="#FF4500", linestyle='solid', label="x_center")
plt.plot(t, y_c, color="#4169E1", linestyle='solid', label="y_center")
plt.plot(t, x_01, color="#FF4500", linestyle='dotted', label="x_01")
plt.plot(t, y_01, color="#4169E1", linestyle='dotted', label="y_01")
plt.plot(t, x_12, color="#FF4500", linestyle='dashed', label="x_12")
plt.plot(t, y_12, color="#4169E1", linestyle='dashed', label="y_12")
plt.plot(t, x_20, color="#FF4500", linestyle='dashdot', label="x_20")
plt.plot(t, y_20, color="#4169E1", linestyle='dashdot', label="y_20")
plt.axvline(fix_t, color='grey', linestyle='dashed')

plt.xlim((0, fix_t+0.5))
plt.legend(loc='lower right')
plt.show()
