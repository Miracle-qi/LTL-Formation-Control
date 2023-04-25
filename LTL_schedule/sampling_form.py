import numpy as np
from scipy import sparse as sp
from tulip import transys, spec, synth
from tulip.transys import machines


def uniform_sampling(space, unit_size, label_dic):
    sys = transys.FTS()
    x_num = space[0] // unit_size[0]
    y_num = space[0] // unit_size[0]

    def get_pos(index):
        row = index % x_num
        column = index // x_num
        pos = [row * unit_size[0], column * unit_size[1]]
        return pos

    def get_coordinate(index):
        row = index % x_num
        column = index // x_num
        return (row, column)

    def labelling():
        labels_set = [set() for _ in range(x_num * y_num)]
        for label in label_dic:
            for n in range(len(states)):
                pos = get_pos(n)
                area = label_dic[label]
                if area[0][0] <= pos[0] <= area[0][1] and area[1][0] <= pos[1] <= area[1][1]:
                    labels_set[n] = {label}
        return labels_set

    states = ['X' + str(i) for i in range(x_num * y_num)]
    sys.states.add_from(states)
    sys.states.initial.add(states[0])
    labels = list(set(label_set.keys()))
    sys.atomic_propositions.add_from(labels)
    labels_set = labelling()

    # Add states and decorate TS with state labels (aka atomic propositions)
    for state, label in zip(states, labels_set):
        sys.states.add(state, ap=label)

    # Add Transitions
    transmat = np.identity(len(states))
    for n in range(len(states)):
        for m in range(len(states)):
            if (abs(get_coordinate(m)[0] - get_coordinate(n)[0]) <= 1)\
                    and (abs(get_coordinate(m)[1] - get_coordinate(n)[1]) <= 1):
                transmat[n][m] = 1

    transmat = sp.lil_matrix(transmat)
    sys.transitions.add_adj(transmat, states)

    return sys


if __name__ == "__main__":
    space = (4, 4)
    unit = (1, 1)
    label_set = {'home': [[0, 1], [0, 1]],
                 'bench': [[3, 4], [3, 4]],
                 'obstacle': [[0, 2], [1, 2]]}
    sys = uniform_sampling(space, unit, label_set)

    # My Specification:
    env_vars = {'work'}
    env_init = set()  # empty set
    env_prog = '!work'
    env_safe = set()  # empty set
    # Augment the system description to make it GR(1)
    sys_vars = {'X0reach'}
    sys_init = {'home'}
    sys_prog = {'home'}
    sys_safe = {'(X (X0reach) <-> bench) || (X0reach && !work)'}
    sys_safe |= {'!obstacle'}
    sys_prog |= {'X0reach'}

    # Create the specification
    specs = spec.GRSpec(env_vars, sys_vars, env_init, sys_init,
                        env_safe, sys_safe, env_prog, sys_prog)

    specs.moore = True
    specs.qinit = r'\E \A'
    ctrl = synth.synthesize(specs, sys=sys)
    machines.random_run(ctrl, N=10)
    print(ctrl)
