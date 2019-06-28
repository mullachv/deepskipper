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
#geod = Geodesic.WGS84

import random


# try:
#     import hfo_py
# except ImportError as e:
#     raise error.DependencyNotInstalled("{}. (HINT: you can install HFO dependencies with 'pip install gym[ship].)'".format(e))

import logging
logger = logging.getLogger(__name__)
import unittest


from gym_shipping.envs.shipping_env import ShippingEnv
class TestBasicFunctions(unittest.TestCase):
    # def test_check_perf_model(self):
    #     se = ShippingEnv(start_datetime=(2019,6,2,12))
    #     gfs = (1.5, .814) #wind
    #     oscar = (-0.22, 0.04) #current
    #     swh = 0
    #     # 27.52269554 - 88.51104917
	#
    #     rpm =12.86288738
    #     heading = 106.1794662 #degrees
    #     gfs_dir, gfs_speed = np.arctan(gfs[0]/gfs[1]), np.linalg.norm(gfs)
    #     # gfs_c = se.c(speed=gfs_speed,
    #     #                             direction=gfs_dir,
    #     #                             heading=heading)
    #     # oscar_dir, oscar_speed = np.arctan(oscar[0]/oscar[1]), np.linalg.norm(oscar)
    #     # oscar_c = se._composite_value(speed=oscar_speed,
    #     #                               direction=oscar_dir,
    #     #                               heading=heading)
    #     sog, fuel = se.sog_and_fuel(oscar, gfs, swh, rpm, heading)
    #     # sog, fuel = se.compute_perf_metrics(ship_heading=heading,
    #     #                                  oceanic_current_direction=oscar_dir,
    #     #                                  oceanic_current_speed=oscar_c,
    #     #                                  propeller_rpm=rpm,
    #     #                                  wave_height=swh,
    #     #                                  wind_direction=gfs_dir,
    #     #                                  wind_speed=gfs_c)
    #     #np.testing.assert_approx_equal(sog, 23, 4)
    #     #self.assertAlmostEquals(sog, 23)
    #     assert math.isclose(sog, 2.225, abs_tol=.01)

    def test_guilherme_1(self):
        se = ShippingEnv(start_datetime=(2019,6,2,12))
        rpm = 88.137654
        comp_wind=5.916670
        comp_current=1.160865
        wave_height = 1.529305
        sog, fuel = se.sog_fuel_from_composite(rpm, comp_wind, comp_current, wave_height)
        #self.assertAlmostEquals(sog, 17.59)
        # self.assertAlmostEquals(fuel, 43.825)
        #np.testing.assert_approx_equal(sog, 17.59, 3)
        assert math.isclose(sog, 17.59, abs_tol=.01)
        assert math.isclose(fuel, 44.11, abs_tol=.01)

    def test_adjusted_csot(self):
        se = ShippingEnv(start_datetime=(2019,6,2,12))
        # Panama = (9.279095, -90.97440333)  # waypoint 50
        # Osaka = (34.095695, 144.997235) # real Osaka
        # Osaka = (9.620595,	-92.30147167) #waypoint 55
        # Osaka = (12.76326667, - 104.7082783) #waypoint 100
        # Osaka = (26.182365, -158.63953) #waypoint 300
        # Osaka = (9.420316667, -91.50667667) #waypoint 52

        se.reset()

        next_location = (9.420316667, -91.50667667)  # waypoint 52
        heading = ShippingEnv.geod.Inverse(*ShippingEnv.Panama, *next_location)['azi1']
        logger.info('heading: ' + str(heading))

        #Move the ship
        se.step(np.array([80, heading]))
        se.step(np.array([80, heading]))
        obs, reward, done, info = se.step(np.array([80, heading]))

        self.assertAlmostEquals(se.get_adjusted_cost_if_passing_through_destination(ShippingEnv.Osaka),0)

if __name__ == '__main__':
    unittest.main()
