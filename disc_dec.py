import logging

from tulip import transys, spec, synth

# Create a finite transition system
sys = transys.FTS()

# Define the states of the system
sys.states.add_from(['X0', 'X1', 'X2'])
sys.states.initial.add('X0')    # start in "horizon" shape

# Define the allowable transitions
sys.transitions.add_comb({'X0'}, {'X0', 'X1'})
sys.transitions.add_comb({'X1'}, {'X1', 'X2'})
sys.transitions.add_comb({'X2'}, {'X2', 'X0'})

#Add atomic propositions to the states
sys.atomic_propositions.add_from({'horizon', 'triangle', 'vertical'})
sys.states.add('X0', ap={'horizon'})
sys.states.add('X1', ap={'triangle'})
sys.states.add('X2', ap={'vertical'})

'''
We must convert this specification into GR(1) form:
  env_init && []env_safe && []<>env_prog_1 && ... && []<>env_prog_m ->
      sys_init && []sys_safe && []<>sys_prog_1 && ... && []<>sys_prog_n
'''

# My Specification: []<>Change -> []<>horizon || []<>vertical || []<>triangle
env_vars = {'Change'}
env_init = set()                # empty set
env_prog = '!Change'             # How to control the time of signal?
# env_prog |= '！Change'
env_safe = set()                # empty set

# Augment the system description to make it GR(1)
sys_vars = set()
sys_init = {'horizon && !triangle && !vertical'}
sys_prog = set()
sys_safe = {'Change && (horizon && !triangle && !vertical) -> (!X(horizon) && X(triangle) && !X(vertical))'}
sys_safe &= {'!Change && (horizon && !triangle && !vertical) -> (X(horizon) && !X(triangle) && !X(vertical))'}
sys_safe &= {'Change && (!horizon && triangle && !vertical) -> (!X(horizon) && !X(triangle) && X(vertical))'}
sys_safe &= {'!Change && (!horizon && triangle && !vertical) -> (!X(horizon) && X(triangle) && !X(vertical))'}
sys_safe &= {'Change && (!horizon && !triangle && vertical) -> (X(horizon) && !X(triangle) && !X(vertical))'}
sys_safe &= {'!Change && (!horizon && !triangle && vertical) -> (!X(horizon) && !X(triangle) && X(vertical))'}



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
print(ctrl)