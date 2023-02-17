#!/usr/bin/env python

from __future__ import print_function
from tulip import transys
import logging

import numpy as np
from tulip import spec, synth, hybrid
from polytope import box2poly
from tulip.abstract import prop2part, discretize
from tulip.abstract.plot import plot_partition

# We label the states using the following picture
#
#     +----+        +----+
#     | X0 |        | X4 |
#     +----+--------+----+
#     | X1 |   X2   | X3 |
#     +----+--------+----+

show = False

# Create a finite transition system
sys = transys.FTS()

# Define the states of the system
sys.states.add_from(['X0', 'X1', 'X2', 'X3', 'X4'])
sys.states.initial.add('X0')

# Define the allowable transitions
sys.transitions.add_comb({'X0'}, {'X0', 'X1'})
sys.transitions.add_comb({'X1'}, {'X1', 'X0', 'X2'})
sys.transitions.add_comb({'X2'}, {'X1', 'X2', 'X3'})
sys.transitions.add_comb({'X3'}, {'X2', 'X3', 'X4'})
sys.transitions.add_comb({'X4'}, {'X3', 'X4'})

# Add atomic propositions to the states
sys.atomic_propositions.add_from({'home', 'bench', 'corridor', 'corner1', 'corner2'})
sys.states.add('X0', ap={'home'})
sys.states.add('X1', ap={'corner1'})
sys.states.add('X2', ap={'corridor'})
sys.states.add('X3', ap={'corner2'})
sys.states.add('X4', ap={'bench'})

'''
We must convert this specification into GR(1) form:
  env_init && []env_safe && []<>env_prog_1 && ... && []<>env_prog_m ->
      sys_init && []sys_safe && []<>sys_prog_1 && ... && []<>sys_prog_n
'''
# My Specification:
env_vars = {'work'}
env_init = set()                # empty set
env_prog = '!work'             # How to control the time of signal?
env_safe = set()                # empty set

# Augment the system description to make it GR(1)
sys_vars = set()
sys_init = {'home'}
sys_prog = set()
sys_safe = {'((home && work)->X(corner1)) &&'
            '((corner1 && work)->X(corridor)) &&'
            '((corridor && work)->X(corner2)) &&'
            '((corner2 && work)->X(bench)) &&'
            '((bench && !work)->X(corner2)) &&'
            '((corner2 && !work)->X(corridor)) &&'
            '((corridor && !work)->X(corner1)) &&'
            '((corner1 && !work)->X(home))'}

# Create the specification
specs = spec.GRSpec(env_vars, sys_vars, env_init, sys_init,
                    env_safe, sys_safe, env_prog, sys_prog)

# Moore machines
# controller reads `env_vars, sys_vars`, but not next `env_vars` values
specs.moore = True
# synthesizer should find initial system values that satisfy
# `env_init /\ sys_init` and work, for every environment variable
# initial values that satisfy `env_init`.
specs.qinit = r'\E \A'
ctrl = synth.synthesize(specs, sys=sys)
assert ctrl is not None, 'unrealizable'
ctrl.save('./Figures/form_2_switch.png')
print(ctrl)
