#!/usr/bin/env python

from __future__ import print_function

import logging
from tulip import spec, synth, hybrid
from polytope import box2poly
from tulip.abstract import prop2part, discretize
from tulip.abstract.plot import plot_partition

logging.basicConfig(level=logging.WARNING)
show = True

# Problem parameters
input_bound = 1.0
uncertainty = 0.01

# Continuous state space
cont_state_space = box2poly([[-1., 1.], [-1., 1.]])

cont_props = {}
cont_props['X0'] = box2poly([[0., 1.], [0., 1.]])
cont_props['X1'] = box2poly([[-1., 0.], [0., 1.]])
cont_props['X2'] = box2poly([[-1., 0.], [-1., 0.]])
cont_props['X3'] = box2poly([[0., 1.], [-1., 0.]])

# Compute the proposition preserving partition of the continuous state space
cont_partition = prop2part(cont_state_space, cont_props)
plot_partition(cont_partition) if show else None

'''
GR(1) form:
  env_init && []env_safe && []<>env_prog_1 && ... && []<>env_prog_m ->
      sys_init && []sys_safe && []<>sys_prog_1 && ... && []<>sys_prog_n
'''
env_vars = {'Change'}
env_init = set()                # empty set
env_prog = '！Change'             # How to control the time of signal?
env_safe = set()                # empty set

# Augment the system description to make it GR(1)
sys_vars = set()
sys_init = {'X0'}
sys_prog = set()

sys_safe = {'((X0 && Change) -> X(vertical)) && ((vertical && Change) -> X(horizon)) &&'
            '((horizon && !Change) -> X(horizon)) && ((vertical && !Change) -> X(vertical))'}

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
ctrl = synth.synthesize(specs)
assert ctrl is not None, 'unrealizable'
ctrl.save('./Figures/form_2_switch.png')
print(ctrl)