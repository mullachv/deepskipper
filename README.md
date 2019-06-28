# Deep Skipper
Deep Skipper discovers minimal fuel cost trajectories from Origin to Destination. In this repository we showcase minimal fuel cost path from Panama to Osaka. The technique used for exploration is based on Deep Reinforcement Learning with Policy Gradient based learning.

In order to facilitate this experimentation, we created an OpenAI Gym environment called, Shipping. We also created a common library called nautlabs. We encourage you to write your own agent code for this environment. Our simple policy gradient based implementation is ``` simple_pg.py ```. An evaluation code that corresponds to that trained model is in ``` eval_simple.py ```.


# Model
![alt Deep Reinforcement Learning with Policy Gradient](https://github.com/mullachv/deepskipper/blob/master/rl-agent-model.png "Deep RL with Policy Gradient")

# Data
The data that is required for this setup is the Global Marine weather data from [National Centers for Environmental Information](https://www.ncdc.noaa.gov/cdo-web/datasets):
- wind.json (also known as GFS)
- wave-height.json (also known as WAVE-HEIGHT)
- oscar.json (ocean current information)

# Dependencies

# Installation
```
pip install -e gym_shipping
 
```
# Execution

# Results

# Relevant Files here
## OpenAI Gym environment, Shipping
The contents here include:
```
.
├── eval_simple.py
├── gym-shipping
│   ├── README.md
│   ├── __init__.py
│   ├── gym_shipping
│   │   ├── __init__.py
│   │   ├── envs
│   │   │   ├── __init__.py
│   │   │   ├── shipping_env.py
│   │   └── test
│   │       ├── __init__.py
│   │       └── test_shipping_env.py
│   ├── gym_shipping.egg-info
│   │   ├── PKG-INFO
│   │   ├── SOURCES.txt
│   │   ├── dependency_links.txt
│   │   ├── requires.txt
│   │   └── top_level.txt
│   └── setup.py
├── index.md
├── map_animate.ipynb
├── nautlabs
│   ├── __init__.py
│   ├── shipperf.py
│   └── tests
│       └── test_shipperf.py
├── nav2.mp4
├── requirements.txt
├── runp.sh
└── simple_pg.py

```

