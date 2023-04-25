
[![](https://res.cloudinary.com/marcomontalbano/image/upload/v1682428720/video_to_markdown/images/youtube--r1aecBOeDq0-c05b58ac6eb4c4700831b2b3070cd403.jpg)](https://www.youtube.com/watch?v=r1aecBOeDq0 "")


## Introduction 
All the `.py` files can be executed directly:
* FxT_control
  * `FxT_QP.py`: Fixed-time CBF-CLF QP controller for the single agent
  * `FxT_QP_Swarm.py`: Fixed-time CBF-CLF QP controller for a robot swarm
* LTL_Schedule
  * `sampling.py`: provide a abstract system with uniform griding 
  * `ltl_ctrl.py`: synthesis a feasible strategy with tulip toolbox
  * `grid_sim.py`: display the high-level strategy in 3D figure

## Dependency 
1. [Casadi](https://web.casadi.org/)
2. [Tulip Toolbox](https://github.com/tulip-control/tulip-control)
3. [Omega](https://github.com/tulip-control/omega)