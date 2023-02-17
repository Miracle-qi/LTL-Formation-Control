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

# Problem parameters
input_bound = 1.0
uncertainty = 0.0

# Continuous state space
cont_state_space = box2poly([[0, 4], [0, 2]])

# Continuous dynamics
# (continuous-state, discrete-time)
A = np.array([[1.0, 0.], [0., 1.0]])
B = np.array([[0.1, 0.], [0., 0.1]])
E = np.array([[1, 0], [0, 1]])

# Available control, possible disturbances
U = input_bound * np.array([[-1., 1.], [-1., 1.]])
W = uncertainty * np.array([[-1., 1.], [-1., 1.]])

# Convert to polyhedral representation
U = box2poly(U)
W = box2poly(W)

# Construct the LTI system describing the dynamics
sys_dyn = hybrid.LtiSysDyn(A, B, E, None, U, W, cont_state_space)

# Define atomic propositions for relevant regions of state space
cont_props = {}
cont_props['home'] = box2poly([[0, 1], [0, 2]])
cont_props['bench'] = box2poly([[3, 4], [0, 2]])
cont_props['corridor'] = box2poly([[1, 3], [0, 1]])
cont_props['wall'] = box2poly([[1, 3], [1, 2]])

# Compute the proposition preserving partition of the continuous state space
cont_partition = prop2part(cont_state_space, cont_props)
plot_partition(cont_partition)

# Given dynamics & proposition-preserving partition, find feasible transitions
disc_dynamics = discretize(
    cont_partition, sys_dyn, closed_loop=True,
    N=8, min_cell_volume=.1, plotit=show
)  # Todo: check the parameters

# Visualize transitions in continuous domain (optional)
plot_partition(disc_dynamics.ppp, disc_dynamics.ts,
               disc_dynamics.ppp2ts)


# Environment variables and assumptions
env_vars = {'work'}
env_init = set()                # empty set
env_prog = 'work'
env_safe = set()                # empty set

# System variables and requirements
sys_vars = set()
sys_init = {'home'}
sys_prog = set()
sys_safe = {'((home && work)->X(corridor)) &&'
            '((corridor && work)->X(bench)) &&'
            '((bench && !work)->X(corridor)) &&'
            '((corridor && !work)->X(home))'}



# Create the specification
specs = spec.GRSpec(env_vars, sys_vars, env_init, sys_init,
                    env_safe, sys_safe, env_prog, sys_prog)

specs.qinit = r'\E \A'
ctrl = synth.synthesize(specs, sys=disc_dynamics.ts, ignore_sys_init=True)
assert ctrl is not None, 'unrealizable'
ctrl.save('./Figures/two_agents.png')
print(ctrl)