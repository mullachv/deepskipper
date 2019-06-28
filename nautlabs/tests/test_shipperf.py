import gym
import os, subprocess, time, signal
from gym import error, spaces, utils
from gym.utils import seeding
import pandas as pd
import json
import datetime
import os
import os.path as path
import sys
import math
import numpy as np

from geographiclib.geodesic import Geodesic

import random


# try:
#     import hfo_py
# except ImportError as e:
#     raise error.DependencyNotInstalled("{}. (HINT: you can install HFO dependencies with 'pip install gym[ship].)'".format(e))

import logging
logger = logging.getLogger(__name__)
import unittest

import nautlabs.shipperf as ns
class TestBasicFunctions(unittest.TestCase):
	def test_sog_fuel(self):
		oscar_vec = (0,0)
		gfs_vec = (-7.649285, -2.10531 )
		swh = 1.75
		rpm = 20
		heading = 0
		s, f = ns.sog_and_fuel(oscar_vec, gfs_vec, swh, rpm, heading)
		self.assertAlmostEquals(s, 3.384404574)
		self.assertAlmostEquals(f, 3.798791224)

if __name__ == '__main__':
	unittest.main()
