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
        env_vars = {"Battery"}
        env_init = set()  # empty set
        env_prog = set()
        env_safe = {'(!Battery && !home) -> X !Battery'}  # empty set
        env_safe |= {'(!Battery && home) -> X (Battery)'}  # empty set

        # System allowed behavior
        sys_vars = set()
        sys_init = {'home && horizon'}
        sys_safe = {'!obstacle'}
        sys_prog = {'bench && triangle'}
        sys_prog |= {'Battery'}

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
            states_dict[state] = edges['loc']
        self.state_dict = states_dict

    def run(self, state, battery):
        next_state, dum = self.ctrl.reaction(state, {'Battery': battery})
        print("currect state:", dum['loc'])
        return next_state, dum


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
    battery = True
    ltl_tasks.get_stateDict()
    print(ltl_tasks.ctrl)

    n = 0
    cur_index = 'Sinit'
    while n < 50:
        next_index, dum = ltl_tasks.run(cur_index, battery)
        cur_index = next_index
        if (dum['bench'] == True) and (dum['triangle'] == True):
            print("Achieved Task!")
        if n == 20:
            battery = False
            print("Low Battery!")
        if (dum['home'] == True) and (dum['horizon'] == True):
            battery = True
            print("Fully Charged!")
        n += 1