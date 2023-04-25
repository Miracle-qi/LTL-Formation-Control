from tulip import transys, spec, synth
from sampling import UniSample
from tulip.transys import machines

class TaskSchedule:

    def __init__(self):
        self.ctrl = None
        self.sampling = None
        self.transitions = None
        self.state_dict = None

        self.space = (4, 4) # The origin is (0,0) by default
        self.unit = (1, 1)
        self.label_set = {((0, 1), (0, 1)): 'home',
                          ((3, 4), (3, 4)): 'bench',
                          ((0, 1), (1, 2)): 'obstacle'}
        self.form_dict = {'horizon':  (1, 1),
                          'vertical': (1, 1),
                          'triangle': (1, 1)}

    def setup(self):
        # Create a finite transition system
        self.sampling = UniSample(self.space, self.unit, self.label_set, self.form_dict)
        self.sampling.setup()

        # My Specification:
        env_vars = {'work'}
        env_init = set()  # empty set
        env_prog = '!work'
        env_safe = set()  # empty set
        # Augment the system description to make it GR(1)
        sys_vars = {'X0reach'}
        sys_init = {'home_horizon'}
        sys_prog = {'home_horizon'}
        sys_safe = {'(X (X0reach) <-> bench_triangle) || (X0reach && !work)'}
        sys_safe |= {'!obstacle_horizon && !obstacle_vertical && !obstacle_triangle '}
        sys_prog |= {'X0reach'}

        # Create the specification
        specs = spec.GRSpec(env_vars, sys_vars, env_init, sys_init,
                            env_safe, sys_safe, env_prog, sys_prog)
        # Moore machines
        specs.moore = False
        specs.qinit = r'\E \A'
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

    def run(self, state, work_signal):
        # trans = self.ctrl.transitions.find([state])
        # outputs = project_dict(attr_dict, mealy.outputs)
        # machines.random_run(self.ctrl, N=10)
        next_state, dum = self.ctrl.reaction(state, {'work': work_signal})
        u, v, edges = list(self.ctrl.edges(next_state, data=True))[0]
        print("Next goal: ", edges['loc'])
        return next_state, edges['loc']


if __name__ == "__main__":
    ltl_tasks = TaskSchedule()
    ltl_tasks.setup()

    change_signal = True
    ltl_tasks.setup()
    ltl_tasks.get_stateDict()
    # ltl_tasks.run(0, change_signal)