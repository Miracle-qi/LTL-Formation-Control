import numpy as np
from scipy import sparse as sp
from tulip import transys, spec, synth
from tulip.interfaces.omega import synthesize_enumerated_streett

from SymbolicReductions.SymbolicMealy import *
from tulip.interfaces import omega as omega_int
try:
    import omega
    from omega.logic import bitvector as bv
    from omega.games import gr1
    from omega.symbolic import symbolic as sym
    from omega.games import enumeration as enum
except ImportError:
    omega = None


# This class will build FST based on uniform sampling (grid) considering formation
class UniSample:
    def __init__(self, space, unit_size, area_labels, form_labels):
        self.space = space
        self.unit_size = unit_size
        self.form_num = len(form_labels)
        self.area_labels = area_labels
        self.form_labels = form_labels

        self.sys = transys.FTS()
        self.x_num = space[0] // unit_size[0]
        self.y_num = space[1] // unit_size[1]
        self.states = np.array(['X' + str(i) + '_f' + str(j) for i in range(self.x_num * self.y_num)
                                for j in range(self.form_num)])
        self.states_mat = self.states.reshape((self.x_num * self.y_num, self.form_num))

    def setup(self):
        self.sys.states.add_from(self.states)
        self.sys.states.initial.add(self.states[0])
        self.sys.atomic_propositions.add_from(self.get_labels_list())
        labels_set = self.labelling()

        # Add states and decorate TS with state labels (aka atomic propositions)
        for state, label in zip(self.states, labels_set):
            self.sys.states.add(state, ap=label)

        # Add Transitions
        transmat = np.identity(len(self.states))
        for n in range(len(self.states)):
            for m in range(len(self.states)):
                c_m = self.get_coordinate(m)
                p_m = self.get_pos(m//self.form_num)
                f_m = self.get_formation(m)
                fm_avail = self.get_avail_formation(p_m)

                c_n = self.get_coordinate(n)
                p_n = self.get_pos(n//self.form_num)
                f_n = self.get_formation(n)
                fn_avail = self.get_avail_formation(p_n)
                if ((abs(c_m[0] - c_n[0]) + abs(c_m[1] - c_n[1])) <= 1)  \
                        and (f_m in fm_avail) and (f_n in fn_avail):
                    transmat[n][m] = 1
        transmat = sp.lil_matrix(transmat)
        self.sys.transitions.add_adj(transmat, self.states)

    def labelling(self):
        labels_set = [set() for _ in range(self.form_num * self.x_num * self.y_num)]
        for n in range(self.x_num * self.y_num):
            for m in range(self.form_num):
                for scope in self.area_labels:
                    pos = self.get_pos(n)
                    area = self.area_labels[scope]
                    if (scope[0][0] <= pos[0] <= scope[0][1]) and (scope[1][0] <= pos[1] <= scope[1][1]):
                        labels_set[self.form_num * n + m] = {area + "_" + list(self.form_labels.keys())[m]}
                        break
                    else:
                        labels_set[self.form_num * n + m] = {"free" + "_" + list(self.form_labels.keys())[m]}
        return labels_set

    def get_labels_list(self):
        area_labels_list = list(set(self.area_labels.values()))
        area_labels_list.append("free")
        form_labels_list = list(set(self.form_labels.keys()))
        labels_list = []
        for area in area_labels_list:
            for form in form_labels_list:
                labels_list.append(area + "_" + form)
        return labels_list

    def get_avail_formation(self, pos):
        dx = self.unit_size[0] / 2
        dy = self.unit_size[1] / 2
        search_radius = 2
        xp_max = 0
        xn_max = 0
        for n in range(1, int(search_radius // dx)):
            check_xp = [pos[0] + n * dx, pos[1]]
            if (self.is_in_space(check_xp)) and (not self.is_in_obstacle(check_xp)):
                xp_max = n * dx
            else:
                break
        for n in range(1, int(search_radius // dx)):
            check_xn = [pos[0] - n * dx, pos[1]]
            if (self.is_in_space(check_xn)) and (not self.is_in_obstacle(check_xn)):
                xn_max += n * dx
            else:
                break

        yp_max = 0
        yn_max = 0
        for m in range(0, int(search_radius // dy)):
            check_yp = [pos[0], pos[1] + m * dy]
            if (self.is_in_space(check_yp)) and not self.is_in_obstacle(check_yp):
                yp_max = m * dy
            else:
                break
        for m in range(0, int(search_radius // dy)):
            check_yn = [pos[0], pos[1] - m * dy]
            if (self.is_in_space(check_yn)) and not self.is_in_obstacle(check_yn):
                yn_max = m * dy
            else:
                break

        dp_max = 0 # diagonal
        dn_max = 0
        for k in range(0, int(search_radius // dy)):
            check_dp = [pos[0] + k * dy, pos[1] + k * dy]
            if (self.is_in_space(check_dp)) and not self.is_in_obstacle(check_dp):
                dp_max = k * dy
            else:
                break
        for k in range(0, int(search_radius // dy)):
            check_dn = [pos[0] - k * dy, pos[1] - k * dy]
            if (self.is_in_space(check_dn)) and not self.is_in_obstacle(check_dn):
                dn_max = k * dy
            else:
                break

        avail_form = []
        for form, limits in zip(self.form_labels.keys(), self.form_labels.values()):
            if (xp_max + xn_max >= limits[0]) and (yp_max + yn_max >= limits[1]) and (dn_max+dp_max >= limits[2]):
                avail_form.append(form)
        return avail_form

    def is_in_space(self, check_point):
        if (0 <= check_point[0] <= self.space[0]) and (0 <= check_point[1] <= self.space[1]):
            return True
        else:
            return False

    def is_in_obstacle(self, check_point):
        for scope, label in zip(self.area_labels.keys(), self.area_labels.values()):
            if (label == 'obstacle') and (scope[0][0] < check_point[0] < scope[0][1]) \
                    and (scope[1][0] < check_point[1] < scope[1][1]):
                return True
        return False

    def get_pos(self, index):
        row = index % self.y_num + 0.5
        column = index // self.x_num + 0.5
        pos = [row * self.unit_size[0], column * self.unit_size[1]]
        return pos

    def get_coordinate(self, index):
        loc = index // self.form_num
        row = loc % self.x_num
        column = loc // self.x_num
        return (row, column)

    def get_formation(self, index):
        _index = index % self.form_num
        return list(self.form_labels.keys())[_index]

    def get_index(self, coordinate):
        return coordinate[0] * self.x_num + coordinate[1]

    def test_avail_formation(self):
        for m in range(self.space[0]):
            for n in range(self.space[1]):
                avail_form = self.get_avail_formation([m+0.5, n+0.5])
                print("Position: ", [m+0.5, n+0.5], "Available formation:", avail_form)


if __name__ == "__main__":
    space = (5, 5) # The origin is (0,0) by default
    unit = (1, 1)
    area_dict = {((0, 1), (0, 1)): 'home',
                 ((3, 4), (3, 4)): 'bench',
                 ((1, 2), (2, 4)): 'obstacle',
                 ((1, 4), (1, 2)): 'obstacle'}
    form_dict = {'horizon':  (2, 1, 1),
                 'vertical': (1, 2, 1),
                 'triangle': (2, 2, 2)}
    sampling = UniSample(space, unit, area_dict, form_dict)
    sampling.setup()
    sampling.test_avail_formation()

    # My Specification:
    env_vars = {'work'}
    env_init = set()  # empty set
    env_prog = {'!work'}
    env_safe = set()  # empty set
    # Augment the system description to make it GR(1)
    sys_vars = {'X0reach'}
    sys_init = {'home_horizon'}
    sys_prog = {'home_horizon'}
    sys_safe = {'(X (X0reach) <-> bench_triangle) || (X0reach && !work)'}
    sys_safe |= {'!obstacle_horizon && !obstacle_vertical && !obstacle_triangle'}
    sys_prog |= {'X0reach'}

    # Create the specification
    psi = spec.GRSpec(env_vars, sys_vars, env_init, sys_init,
                        env_safe, sys_safe, env_prog, sys_prog)
    psi.qinit = '\A \E'
    psi.moore = False
    ctrl = synth.synthesize(psi, sys=sampling.sys)
    print(ctrl)
