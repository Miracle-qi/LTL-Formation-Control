#!/usr/bin/env python

from __future__ import print_function
from tulip import transys
from tulip import spec, synth
from tulip.abstract.plot import plot_partition
from tulip.transys import machines
from tulip.abstract import find_controller


# We label the states using the following picture
#
#     +----+        +----+
#     | X0 |        | X4 |
#     +----+--------+----+
#     | X1 |   X2   | X3 |
#     +----+--------+----+

class TaskSchedule:

    def __init__(self):
        self.cont_partition = None
        self.sys = None
        self.ctrl = None

    def setup(self):
        # Create a finite transition system
        self.sys = transys.FTS()
        # Define the states of the system
        self.sys.states.add_from(['X0', 'X1', 'X2', 'X3', 'X4'])
        self.sys.states.initial.add('X0')
        # Define the allowable transitions
        self.sys.transitions.add_comb({'X0'}, {'X0', 'X1'})
        self.sys.transitions.add_comb({'X1'}, {'X1', 'X0', 'X2'})
        self.sys.transitions.add_comb({'X2'}, {'X1', 'X2', 'X3'})
        self.sys.transitions.add_comb({'X3'}, {'X2', 'X3', 'X4'})
        self.sys.transitions.add_comb({'X4'}, {'X3', 'X4'})
        # Add atomic propositions to the states
        self.sys.atomic_propositions.add_from({'home', 'bench', 'corridor', 'corner1', 'corner2'})
        self.sys.states.add('X0', ap={'home'})
        self.sys.states.add('X1', ap={'corner1'})
        self.sys.states.add('X2', ap={'corridor'})
        self.sys.states.add('X3', ap={'corner2'})
        self.sys.states.add('X4', ap={'bench'})
        '''
         Specification must transfer into GR(1) form:
          env_init && []env_safe && []<>env_prog_1 && ... && []<>env_prog_m ->
              sys_init && []sys_safe && []<>sys_prog_1 && ... && []<>sys_prog_n
        '''
        # My Specification:
        env_vars = {'work'}
        env_init = set()  # empty set
        env_prog = '!work'  # How to control the time of signal?
        env_safe = set()  # empty set
        # Augment the system description to make it GR(1)
        sys_vars = set()
        sys_init = {'home'}
        sys_prog = set()
        sys_safe = {'((home && work)->X(corner1)) &&'
                    '((corner1 && work)->X(corridor)) &&'
                    '((corridor && work)->X(corner2)) &&'
                    '((corner2 && work)->X(bench)) &&'
                    '((bench && work)->X(bench)) &&'
                    '((bench && !work)->X(corner2)) &&'
                    '((corner2 && !work)->X(corridor)) &&'
                    '((corridor && !work)->X(corner1)) &&'
                    '((corner1 && !work)->X(home)) &&'
                    '((home && !work)->X(home))'}

        # Create the specification
        specs = spec.GRSpec(env_vars, sys_vars, env_init, sys_init,
                            env_safe, sys_safe, env_prog, sys_prog)
        # Moore machines
        specs.moore = True
        # synthesizer should find initial system values that satisfy
        # `env_init /\ sys_init` and work, for every environment variable
        # initial values that satisfy `env_init`.
        specs.qinit = r'\E \A'
        self.ctrl = synth.synthesize(specs, sys=self.sys)
        assert self.ctrl is not None, 'unrealizable'

    def run(self, state, work_signal):
        # trans = self.ctrl.transitions.find([state])
        # outputs = project_dict(attr_dict, mealy.outputs)
        # machines.random_run(self.ctrl, N=10)
        next_state, dum = self.ctrl.reaction(state, {'work': work_signal})
        u, v, edges = list(self.ctrl.edges(next_state, data=True))[0]
        print("Next goal: ", edges['loc'])
        return edges['loc']

    def display(self):
        plot_partition(self.sys)

    def record(self):
        print(self.ctrl)
        self.ctrl.save('./Figures/tasks_graph.png')


if __name__ == "__main__":
    ltl_tasks = TaskSchedule()
    ltl_tasks.setup()
    ltl_tasks.record()

    change_signal = [True, False, True]
    ltl_tasks.run(0, change_signal)






