from tulip import transys, spec, synth
try:
    from LTL_schedule.sampling import UniSample
except ImportError:
    from sampling import UniSample


class TaskSchedule:

    def __init__(self, space, unit, label_set, form_dict):
        self.ctrl = None
        self.sampling = None
        self.transitions = None
        self.state_dict = None

        self.space = space
        self.unit = unit
        self.label_set = label_set
        self.form_dict = form_dict

    def setup(self):
        # Create a finite transition system
        self.sampling = UniSample(self.space, self.unit, self.label_set, self.form_dict)
        self.sampling.setup()

        # My Specification:
        env_vars = {'work_cmd', 'low_battery'}
        env_init = set()  # empty set
        # env_prog = {'work_cmd && !low_battery'}
        env_prog = {'!work_cmd'}
        env_prog |= {'!low_battery'}
        env_safe = set()  # empty set
        # Augment the system description to make it GR(1)
        sys_vars = {'X0reach'}
        sys_init = {'home_horizon'}
        sys_prog = {'home_horizon'}
        sys_safe = {'(X (X0reach) <-> bench_triangle) || (X0reach && !(work_cmd && !low_battery))'}
        # sys_safe &= {'(X (X0reach) <-> home_horizon) || (X0reach && !low_battery)'}
        sys_safe |= {'!obstacle_horizon && !obstacle_vertical && !obstacle_triangle '}
        sys_prog |= {'X0reach'}

        # Create the specification
        specs = spec.GRSpec(env_vars, sys_vars, env_init, sys_init,
                            env_safe, sys_safe, env_prog, sys_prog)
        # Moore machines
        specs.qinit = '\A \E'
        specs.moore = False
        self.ctrl = synth.synthesize(specs, sys=self.sampling.sys)
        self.transitions = list(set([(x, y) for (x, y, lab) in self.ctrl.transitions(data=True)]))
        self.get_stateDict()

    def get_stateDict(self):
        states_dict = {}
        state_list = list(self.ctrl.states)
        for state in state_list:
            u, v, edges = list(self.ctrl.edges(state, data=True))[0]
            loc_index = int(edges['loc'][1:-3])
            states_dict[state] = edges['loc']
        self.state_dict = states_dict

    def run(self, state, work_signal, battery_warn):
        # trans = self.ctrl.transitions.find([state])
        # outputs = project_dict(attr_dict, mealy.outputs)
        # machines.random_run(self.ctrl, N=10)
        next_state, dum = self.ctrl.reaction(state, {'work_cmd': work_signal, 'low_battery': battery_warn})
        u, v, edges = list(self.ctrl.edges(next_state, data=True))[0]
        return next_state, edges['loc']


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
    ltl_tasks = TaskSchedule(space, unit, area_dict, form_dict)
    ltl_tasks.setup()
    change_signal = True
    battery_warn = False
    ltl_tasks.get_stateDict()
    print(ltl_tasks.ctrl)

    n = 0
    cur_index = 'Sinit'
    while n < 50:
        next_index, next_state = ltl_tasks.run(cur_index, change_signal, battery_warn)
        if n >= 40:
            battery_warn = True
        cur_index = next_index
        print(str(n) + "Next_state: ", next_state)
        n += 1