# Deep Skipper
Deep Skipper discovers minimal fuel cost trajectories from Origin to Destination. In this repository we showcase minimal fuel cost path from Panama to Osaka. The technique used for exploration is based on Deep Reinforcement Learning with Policy Gradient based learning.

In order to facilitate this experimentation, we created an OpenAI Gym environment called, Shipping. We also created a common library called nautlabs. We encourage you to write your own agent code for this environment. Our simple policy gradient based implementation is ``` simple_pg.py ```. An evaluation code that corresponds to that trained model is in ``` eval_simple.py ```.


# Model
![alt Deep Reinforcement Learning with Policy Gradient](https://github.com/mullachv/deepskipper/blob/master/DeepRL.png "Deep RL with Policy Gradient")

# Data
The data that is required for this setup is the Global Marine weather data from [National Centers for Environmental Information](https://www.ncdc.noaa.gov/cdo-web/datasets):
- wind.json (also known as GFS)
- wave-height.json (also known as WAVE-HEIGHT)
- oscar.json (ocean current information)

# Dependencies
Following Python packages are required for this suite of the programs to run:
```
pip==10.0.1
numpy==1.15.4
pandas==0.23.0
matplotlib==3.0.3
jsonschema==2.6.0
basemap==1.2.0
basemap-data-hires==1.2.0
boto==2.48.0
botocore==1.12.169
boto3==1.9.146
awscli==1.16.179
geographiclib==1.49
tensorflow==1.13.1
tensorflow-estimator==1.13.0
tensorflow-serving-api==1.13.0
Keras==2.2.4
gym==0.12.5
ffmpeg==4.0
```
# Installation
```
pip install -e gym_shipping
 
```
# Execution
Execute file ```runp.sh```:

```
.
├── index.md
├── map_animate.ipynb
├── nav2.mp4
├── requirements.txt
├── runp.sh

```
# Results

# Relevant Files here
## OpenAI Gym environment, Shipping
The contents here include:
```
.
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

```
## Nautlabs library
```
.
├── nautlabs
│   ├── __init__.py
│   ├── shipperf.py
│   └── tests
│       └── test_shipperf.py

```

## Deep RL Agent
```
.
├── eval_simple.py
└── simple_pg.py

```
