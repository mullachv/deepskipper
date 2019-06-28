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

import nautlabs.shipperf as ns

# try:
#     import hfo_py
# except ImportError as e:
#     raise error.DependencyNotInstalled("{}. (HINT: you can install HFO dependencies with 'pip install gym[ship].)'".format(e))

import logging
logger = logging.getLogger(__name__)

class ShippingEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    basedatafolder = '/Users/vmullachery/mine/insight/'
    defaultfolder = '2019060200'
    #Panama = (8.517163333, -79.45016167)
    Panama = (9.279095 , -90.97440333) #waypoint 50
    #Osaka = (34.095695, 144.997235) # real Osaka
    #Osaka = (12.76326667, - 104.7082783) #waypoint 100
    Osaka = (26.182365, -158.63953) #waypoint 300
    TimeDelta = 3600
    geod = Geodesic.WGS84
    MAX_TIME_ALLOWED = 505*3600 # 505 hours
    PROXIMITY = 10000 # acceptable proximity to (in meters) to destination

    performance_parameters_of_ship = {
        # Factors affecting fuel consumption
        'fw1': 0.067302, #wind factor 
        'fw2': 0.002554, #wind squared factor
        'fc1': 0.092372, #current factor
        'fc2': 0.001005, #current squared factor
        'fh1': 0.150907, #wave height factor
        'fr1': -0.105935, #propeller rpm factor
        'fr3': 0.000066, #propeller rpm cubed factor
        'flr3': 1.273963, #propeller log10 rpm cubed factor

        #Factors affecting ground speed
        'sw1': -0.0388, #wind factor 
        'sw2': -0.0006, #wind squared factor 
        'sc1': 1.0443, #current factor
        'sc2': -0.1531, #current squared factor
        'sh1': -0.2510, #wave height factor
        'sr1': 0.1954, #propeller rpm factor
        
        
        #Cost adjustment factor
        'cost_adjustment_factor': 1./(24), #timedelta is in seconds, 3600 seconds = 1hr.
                                                #fuel (consumption rate) is in per day
        'knots_to_meters_adj': 0.514444444 #Adjust knots to meters/sec for Geodesic API
    }

    def __init__(self, **kwargs):
        self.origin = ShippingEnv.Panama
        self.destination = ShippingEnv.Osaka
        self.WAVEWATCH={}
        self.GFS={}
        self.OSCAR={}
        self.action_space = spaces.Box(low=np.array([0.0, -180.0]), \
                high = np.array([100.0, 180.0]), \
                dtype = np.float32)
        self.observation_space = spaces.Box(low=np.array([-np.infty]), \
                                            high=np.array([np.infty]))
        # DataFrame of all rpms, headings, corresponding lats/longs, metrics related to the voyage
        self._reset_dataframe()
        self.action_and_observables={}
        self.save_action_observables = {}
        self.__initialize(datetime.datetime(*kwargs['start_datetime']).timestamp())

    # def get_parameters(self):
    #     return ShippingEnv.Panama, ShippingEnv.Osaka, ShippingEnv.TimeDelta

    def __initialize(self, timestamp=datetime.datetime.now().timestamp(), timediscretization=3600):
        self.start_timestamp = timestamp
        self.current_timestamp = timestamp
        self.current_location = self.origin
        ShippingEnv.TimeDelta = timediscretization
        # self._csv_filename =  'p2o_' + \
        #     datetime.datetime.utcfromtimestamp(self.start_timestamp).strftime('%Y%m%d%H') + \
        #         '.csv'

    def get_foldername_from_epoch(self, timestamp, weather_type='GFS'):
        """
		Computes the matching weather data folder. For 'OSCAR'
		attempts the previous 10 days from the supplied timestamp
		to find an existing data folder

		Input:
			timestamp in epoch time, for e.g. 1559488082 = June 2, 2019, 3PM
			weather_type in ['GFS', 'OSCAR', 'WAVEWATCH']
		Output:

		"""
        dttmstamp = datetime.datetime.utcfromtimestamp(timestamp)
        basename = dttmstamp.strftime('%Y%m%d')
        hour = datetime.datetime.utcfromtimestamp(timestamp).strftime('%H')

        check = {
            'GFS': self.check_hour(hour),
            'OSCAR': '00',
            'WAVEWATCH': self.check_hour(hour)
        }
        foldername = basename + check[weather_type]
        if weather_type in ['GFS', 'WAVEWATCH']:
            if not (path.exists(path.join(ShippingEnv.basedatafolder, foldername))):
                foldername = ShippingEnv.defaultfolder
        elif weather_type == 'OSCAR':
            counter = 10
            while not (path.exists(path.join(ShippingEnv.basedatafolder, foldername))) and counter > 0:
                dttmstamp = dttmstamp - datetime.timedelta(1)
                foldername = datetime.datetime.strftime(dttmstamp, '%Y%m%d') + '00'
                counter -= 1
            if counter <= 0:
                # hard code to defaultfolder
                foldername = ShippingEnv.defaultfolder
        return foldername

    def check_hour(self, hour):
        """
        Get nearest recorded hour (for weather)
        :param hour:
        :return:
        """
        if hour >= '18':
            return '18'
        elif hour >= '12':
            return '12'
        elif hour >= '06':
            return '06'
        else:
            return '00'

    def oscar_index_to_location(self, index):
        """
        Translate the OSCAR index to latitude, longitude location
        OSCAR contains the Oceanic Current data
        """
        lat = 80*3 - np.float64(index/(360*3))
        lng = 20*3 + np.float64(index%(360*3))
        return {
            'Lat' : lat / 3.0,
            'Lng' : lng / 3.0,
        }

    def wavewatch_index_to_location(self, index):
        """
        Translate the WAVEWATCH index to latitude, longitude location
        WAVEWATCH contains Significant Wave Height data
        """
        lat = 90*2 - np.float64(index/(360*2))
        lng = np.float64(index%(360*2))
        return {
            'Lat' : lat / 2.0,
            'Lng' : lng / 2.0,
        }

    def gfs_index_to_location(self, index):
        """
        Translate the GFS index to latitude, longitude location
        GFS contains Wind data
        """
        lat = 90*2 - np.float64(index/(360*2))
        lng = np.float64(index%(360*2))
        return {
            'Lat' : lat / 2.0,
            'Lng' : lng / 2.0,
        }

    def location_to_oscar_index(self, location):
        """
        Finds the index in oscar data that closely approximates the lat, lon
        :param location:
        :return: index into oscar data
        """
        a = (80 - location['Lat']) * 3 * 360 * 3
        b = np.floor(3 * (location['Lng'] - 20))
        if a > b:
            return int(np.round(a))
        return int(np.round(b))

    def location_to_gfs_index(self, location):
        """
        Finds the index in GFS (wind) data that approximates the lat, lon
        :param location:
        :return: index into GFS data for the location
        """
        a = (90 - location['Lat']) * 2 * 360 * 2
        b = np.floor(2 * location['Lng'])
        if a > b:
            return int(np.round(a))
        return int(np.round(b))

    def location_to_wavewatch_index(self, location):
        """
        Finds the index in WAVEWATCH that approximates the lat, lon
        :param location:
        :return: index into wavewatch data for the location
        """
        a = (90 - location['Lat']) * 2 * 360 * 2
        b = np.floor(2 * location['Lng'])
        if a > b:
            return int(np.round(a))
        return int(np.round(b))

    # def roll_longitude(self, location):
    #     longitude = location['Lng']
    #     longitude =  (longitude % 360) - 180
    #     return {'Lat': location['Lat'], 'Lng': longitude}

    def get_gfs(self, datetime, location):
        """
        Computes the wind velocity vector at the time and location
        Input:
            datetime in epoch time (numbers)
            location in latitude, longitude: {'Lat': 24.4, 'Lng': -2.5}
        Output:
            ocean wind velocity vector (a tuple): (u, v) in meters/second. u = latitudinal speed
            v = longitudinal speed
        Side effect: loads the GFS dictionary for the YYYYMMDDHH corresponding to
        datetime
        """
        foldername = self.get_foldername_from_epoch(datetime, weather_type='GFS')
        ndx = self.location_to_gfs_index(location)
        if foldername not in self.GFS:
            with open(path.join(foldername, 'wind.json')) as file:
                wind_data = json.load(file)
                wind_u = wind_data[0]['data']
                wind_v = wind_data[1]['data']
                du = pd.DataFrame(wind_u, columns=['windU'])
                dv = pd.DataFrame(wind_v, columns=['windV'])

                # Clean up
                du[du['windU'].isna()] = 0
                dv[dv['windV'].isna()] = 0

                # foldername e.g. 2019060200
                self.GFS[foldername] = [(u, v) for (u, v) in zip(du['windU'].values, dv['windV'].values)]
            #     print(GFS.keys())
            #     print(foldername)
        return self.GFS[foldername][ndx % len(self.GFS[foldername])]

    def get_oscar(self, datetime, location):
        """
        Computes the ocean current vector at the time and location
        Input:
            datetime in epoch time (numbers)
            location in latitude, longitude: {'Lat': 24.4, 'Lng': -2.5}
        Output:
            ocean current vector (a tuple): (u, v) in meters/second. u = latitudinal speed
            v = longitudinal speed
        Side effect: loads the OSCAR dictionary for the YYYYMMDD00 corresponding to
        datetime
        """

        foldername = self.get_foldername_from_epoch(datetime, weather_type='OSCAR')
        ndx = self.location_to_oscar_index(location)
        if foldername not in self.OSCAR:
            with open(path.join(foldername, 'oscar.json')) as file:
                oscar_data = json.load(file)
                oscar_data_u = oscar_data[0]['data']
                oscar_data_v = oscar_data[1]['data']
                du = pd.DataFrame(oscar_data_u, columns=['data'])
                dv = pd.DataFrame(oscar_data_v, columns=['data'])

                # Clean up
                du[du['data'].isna()] = 0
                dv[dv['data'].isna()] = 0

                # foldername e.g. 2019060200
                self.OSCAR[foldername] = [(u, v) for (u, v) in zip(du['data'].values, dv['data'].values)]
        # print(OSCAR)
        return self.OSCAR[foldername][ndx % len(self.OSCAR[foldername])]


    def get_wavewatch(self, datetime, location):
        """
        Computes the wave-height at the time and location
        Input:
            datetime in epoch time (numbers)
            location in latitude, longitude: {'Lat': 24.4, 'Lng': -2.5}
        Output:
            wave height in meters
        Side effect: loads the WAVEWATCH dictionary for the YYYYMMDDHH corresponding to
        datetime
        """
        foldername = self.get_foldername_from_epoch(datetime, weather_type='WAVEWATCH')
        ndx = self.location_to_wavewatch_index(location)
        if foldername not in self.WAVEWATCH:
            with open(path.join(foldername, 'wave-height.json')) as file:
                wave_data = json.load(file)
                swh = wave_data[0]['data']
                dw = pd.DataFrame(swh, columns=['swh'])
                dw[dw['swh'] == 'NaN'] = 0
                # foldername e.g. 2019060212
                self.WAVEWATCH[foldername] = dw['swh'].values
        # print(WAVEWATCH)
        return self.WAVEWATCH[foldername][ndx % len(self.WAVEWATCH[foldername])]


    def __step(self, action):
        """

        Parameters
        ----------
        action :

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        self._take_action(action)
        reward = self._get_reward()
        ob = self._get_state()
        episode_over = \
            ( ( ShippingEnv.geod.Inverse(*self.current_location, *self.destination)['s12'] <= ShippingEnv.PROXIMITY ) or
            (self.current_timestamp - self.start_timestamp >= ShippingEnv.MAX_TIME_ALLOWED) )
        return ob, reward, episode_over, {}

    def _reset_dataframe(self):
        self.wf = pd.DataFrame(columns=['rpm','heading','lat','lon','timestamp',\
            'timedelta','wave-height','gfs','oscar','speed_over_ground','fuel_consumption_rate','cost']) 

    def __reset(self):
        self.current_timestamp = self.start_timestamp
        self.current_location = self.origin
        self.action_and_observables={}
        self._reward = 0
        self._reset_dataframe()
        return self._get_state(), self._reward, False, {}

    def __render(self, mode='human', close=False):
        pass

    def create_location_fragment(self, location):
        """
        Creates a location dictionary object with lat, lon
        :param location:
        :return: a dictionary object {'Lat': lat, 'Lng': lon } for the location
        """
        return {'Lat': location[0], 'Lng': location[1]}

    def _vector_direction(self, vx, vy):
        """
        Compute the vector direction in degrees as measured with North being zero,
        and clockwise positive
        :param vx:
        :param vy:
        :return: in degrees the azimuth (angle)
        """
        if (vy is None) or (vy == 0):
            if vx == 0:
                return 0
            if vx < 0:
                return -90
            return 90
        return math.degrees(np.arctan(vx/vy))


    def _take_action(self, action):
        """
        Takes the action specified by the agent. Steps to the next location, and advances times
        :param action: (rpm, heading)
        :return:
        """
        self.action_and_observables['rpm'], self.action_and_observables['heading'] = action
        self.action_and_observables['lat'], self.action_and_observables['lon'] = self.current_location
        self.action_and_observables['timestamp'] = self.current_timestamp
        self.action_and_observables['timedelta'] = ShippingEnv.TimeDelta
        self.action_and_observables['wave-height'] = \
            self.get_wavewatch(self.current_timestamp, self.create_location_fragment(self.current_location))
        self.action_and_observables['gfs'] = \
            self.get_gfs(self.current_timestamp, self.create_location_fragment(self.current_location))
        self.action_and_observables['oscar'] = \
            self.get_oscar(self.current_timestamp, self.create_location_fragment(self.current_location))

        self.action_and_observables['speed_over_ground'], \
        self.action_and_observables['fuel_consumption_rate'] = \
            self.sog_and_fuel(self.action_and_observables['oscar'],
                              self.action_and_observables['gfs'],
                              self.action_and_observables['wave-height'],
                              self.action_and_observables['rpm'],
                              self.action_and_observables['heading']
                              )
        #compute cost is per ShippingEnv.TimeDelta (per hour)
        self.action_and_observables['cost'] = self.action_and_observables['fuel_consumption_rate'] \
                                              * ShippingEnv.performance_parameters_of_ship['cost_adjustment_factor']

        self.wf = self.wf.append(self.action_and_observables, ignore_index=True)

        next_location = self._compute_next_location()

        # -1 for time/step increment, so incentivizes to reach destination quicker
        #self._reward = -self.action_and_observables['cost'] - 1
        self._reward = -1  # for the time step
        if self._we_passthrough_destination(next_location):
            self.current_location = ShippingEnv.Osaka
            self._reward += -self.get_adjusted_cost_if_passing_through_destination(next_location)
        else:
            # # -ShippingEnv.geod.Inverse(*ShippingEnv.Panama, *ShippingEnv.Osaka)['s12'] * 1. / \
            # #     ShippingEnv.geod.Inverse(*self.current_location, *ShippingEnv.Osaka)['s12']  \
            # - 1
            self._reward += -self.action_and_observables['cost']
            #Now move the ship
            g = ShippingEnv.geod.Direct(self.current_location[0],
                self.current_location[1],
                self.action_and_observables['heading'],
                self.action_and_observables['speed_over_ground']*ShippingEnv.TimeDelta*
                ShippingEnv.performance_parameters_of_ship['knots_to_meters_adj'], )

            #print('geod', g)
            self.current_location = g['lat2'], g['lon2']

        # Advance time
        self.current_timestamp += ShippingEnv.TimeDelta
        self.save_action_observables = self.action_and_observables.copy()
        self.action_and_observables.clear()

    def get_adjusted_cost_if_passing_through_destination(self, next_location):
        """
        Next location will be the destination. Cost will be accrued upto the point of arriving
        at the destination and not for the entire TimeDelta duration
        :param next_location:
        :return: cost for the partial trajectory to destination
        """
        line = ShippingEnv.geod.InverseLine(*self.current_location, *next_location)
        n = int(math.ceil(line.s13 / (ShippingEnv.PROXIMITY)))
        for i in range(n + 1):
            s = min(ShippingEnv.PROXIMITY * i, line.s13)
            g = line.Position(s, Geodesic.STANDARD)
            if (ShippingEnv.geod.Inverse(g['lat2'], g['lon2'], *ShippingEnv.Osaka)['s12'] <= ShippingEnv.PROXIMITY):
                return (i + 1) * self.action_and_observables['cost'] * ShippingEnv.PROXIMITY / \
                        (self.action_and_observables['speed_over_ground'] *
                         ShippingEnv.performance_parameters_of_ship['knots_to_meters_adj'] * ShippingEnv.TimeDelta
                         )

        return self.action_and_observables['cost']

    #
    # Compute the next location, but do not move the ship
    #
    def _compute_next_location(self):
        """
        Compute the next location based on current_location, heading, and speed_over_ground, assuming
         we travel for ShippingEnv.TimeDelta
        :return: next_location
        """
        g = ShippingEnv.geod.Direct(self.current_location[0],
                                    self.current_location[1],
                                    self.action_and_observables['heading'],
                                    self.action_and_observables['speed_over_ground'] * ShippingEnv.TimeDelta *
                                    ShippingEnv.performance_parameters_of_ship['knots_to_meters_adj'], )

        # print('geod', g)
        return g['lat2'], g['lon2']

    #
    #https://geographiclib.sourceforge.io/html/python/examples.html?highlight=example
    #
    def _we_passthrough_destination(self, next_location):
        """
        Checks whether the path from current location to next location passes through a destination
        proximity.
        :param next_location:
        :return: True if the trajectory passes through the destination (proximally)
        """
        line = ShippingEnv.geod.InverseLine(*self.current_location, *next_location)
        n = int(math.ceil(line.s13 / (ShippingEnv.PROXIMITY)))
        for i in range(n+1):
            s = min(ShippingEnv.PROXIMITY*i, line.s13)
            g = line.Position(s, Geodesic.STANDARD)
            if (ShippingEnv.geod.Inverse(g['lat2'], g['lon2'], *ShippingEnv.Osaka)['s12'] <= ShippingEnv.PROXIMITY):
                return True
        return False

    def _get_reward(self):
        """ Reward is given for XY. """
        # if self.status == FOOBAR:
        #     return 1
        # elif self.status == ABC:
        #     return self.somestate ** 2
        # else:
        return self._reward # -1 for the step/time increment 

    def _get_state(self):
        """
        Compute the state tuple. State consists of WAVEWATCH dictionary, GFS dictionary, OSCAR dictionary,
         computed action and observables data record, current location, current timestamp, origin, destination
        :return: state tuple
        """
        gfs = self.get_foldername_from_epoch(self.current_timestamp, weather_type='GFS')
        oscar = self.get_foldername_from_epoch(self.current_timestamp, weather_type='OSCAR')
        ww = self.get_foldername_from_epoch(self.current_timestamp, weather_type='WAVEWATCH')

        if gfs in self.GFS:
            return (self.WAVEWATCH[ww], self.GFS[gfs], self.OSCAR[oscar], \
                    self.save_action_observables, self.current_location, \
                    self.current_timestamp, self.origin, self.destination)
        else:
            return ({}, {}, {},
                    self.action_and_observables, self.current_location,
                    self.current_timestamp, self.origin, self.destination)

    def __seed(self, num):
        """
        Set random number seed for the environment. Note the action space will need to
        be set separately
        :param num:
        :return:
        """
        random.seed(num)
        np.random.seed(num)

    def step(self, action):
        return self.__step(action)

    def reset(self):
        return self.__reset()

    def render(self):
        return self.__render(mode='human')

    # def _write_to_csv(self):
    #     #print(self.wf)
    #     self.wf.to_csv(self._csv_filename)

    def close(self):
        # self._write_to_csv()
        sys.exit("Game Terminated")

    def seed(self, num):
        """
        Set the random number seed
        :param num:
        :return:
        """
        return self.__seed(num)

    def _composite_value(self, speed, direction, heading):
        """
        Computes the effective value (composite value) of speed (wind speed or 
        ocean current speed etc.) in the direction of ship's heading (heading)

        Input:
            speed: wind speed/ocean wave speed
            direction: angular direction with respect to North (as 0) computed
            clockwise (as positive) in Degrees
            heading: ship's heading in Degrees

        Returns:
            composite_value: effective speed (of wind/ocean current) 
            in the direction of the ship's heading

        """
        if heading is None or speed is None:
            return 0
        adjusted_direction = direction - heading
        return math.cos(math.radians(adjusted_direction)) * speed
    
    def compute_perf_metrics(self, wind_speed, wind_direction, 
                             oceanic_current_speed, oceanic_current_direction,
                             wave_height, propeller_rpm, ship_heading):
        """
        Computes speed over ground and fuel consumption of the ship
        Input:
            wind_speed: np.sqrt(windU^2 + windV^2)
            wind_direction: math.degrees(np.arctan(windU/windV))
            oceanic_current_speed: np.sqrt(currentU^2 + currentV^2)
            oceanic_current_direction: math.degrees(np.arctan(currentU/currentV))
            wave_height: in meters
            ship_heading: in degrees
            
        Returns:
            speed_over_ground
            fuel_consumption
            
        """
        #
        #ship_heading: 0 degrees is North. Clockwise is positive (I think)
        #
        composite_wind = self._composite_value(wind_speed, wind_direction, ship_heading)
        composite_current = self._composite_value(oceanic_current_speed, oceanic_current_direction, ship_heading)

        #fuel consumption rate in (metric tons/day)
        fuel_consumption = composite_wind*ShippingEnv.performance_parameters_of_ship['fw1'] \
        + (composite_wind**2) * ShippingEnv.performance_parameters_of_ship['fw2'] \
        + composite_current*ShippingEnv.performance_parameters_of_ship['fc1'] \
        + (composite_current**2) * ShippingEnv.performance_parameters_of_ship['fc2'] \
        + wave_height*ShippingEnv.performance_parameters_of_ship['fh1'] \
        + propeller_rpm * ShippingEnv.performance_parameters_of_ship['fr1'] \
        + (propeller_rpm**3) * ShippingEnv.performance_parameters_of_ship['fr3'] \
        + np.log10(propeller_rpm**3) * ShippingEnv.performance_parameters_of_ship['flr3']
        
        speed_over_ground = composite_wind* ShippingEnv.performance_parameters_of_ship['sw1'] \
        + (composite_wind**2) * ShippingEnv.performance_parameters_of_ship['sw2'] \
        + composite_current* ShippingEnv.performance_parameters_of_ship['sc1'] \
        + (composite_current**2) * ShippingEnv.performance_parameters_of_ship['sc2'] \
        + wave_height* ShippingEnv.performance_parameters_of_ship['sh1'] \
        + propeller_rpm* ShippingEnv.performance_parameters_of_ship['sr1']
        
        return speed_over_ground, fuel_consumption

    def sog_and_fuel(self, oscar_vec, gfs_vec, swh, rpm, heading):
        """
        Compute speed over ground (sog) and fuel from weather vectors, rpm and heading
        :param oscar_vec:
        :param gfs_vec:
        :param swh:
        :param rpm:
        :param heading:
        :return:
        """
        gfs_speed = np.linalg.norm(gfs_vec)
        gfs_dir = self._vector_direction(*gfs_vec)

        oscar_speed = np.linalg.norm(oscar_vec)
        oscar_dir = self._vector_direction(*oscar_vec)

        return self.compute_perf_metrics(gfs_speed, gfs_dir, oscar_speed, oscar_dir, swh, rpm, heading)

    def sog_fuel_from_composite(self, rpm, comp_wind, comp_current, wave_height):
        """
        Compute speed over ground, fuel from rpm, composite wind, composite current, wave height
        :param rpm:
        :param comp_wind:
        :param comp_current:
        :param wave_height:
        :return:
        """
        fuel_consumption = comp_wind * ShippingEnv.performance_parameters_of_ship['fw1'] \
        + (comp_wind ** 2) * ShippingEnv.performance_parameters_of_ship['fw2'] \
        + comp_current * ShippingEnv.performance_parameters_of_ship['fc1'] \
        + (comp_current ** 2) * ShippingEnv.performance_parameters_of_ship['fc2'] \
        + wave_height * ShippingEnv.performance_parameters_of_ship['fh1'] \
        + rpm * ShippingEnv.performance_parameters_of_ship['fr1'] \
        + (rpm ** 3) * ShippingEnv.performance_parameters_of_ship['fr3'] \
        + 3 * np.log10(rpm) * ShippingEnv.performance_parameters_of_ship['flr3']

        speed_over_ground = comp_wind * ShippingEnv.performance_parameters_of_ship['sw1'] \
        + (comp_wind ** 2) * ShippingEnv.performance_parameters_of_ship['sw2'] \
        + comp_current * ShippingEnv.performance_parameters_of_ship['sc1'] \
        + (comp_current ** 2) * ShippingEnv.performance_parameters_of_ship['sc2'] \
        + wave_height * ShippingEnv.performance_parameters_of_ship['sh1'] \
        + rpm * ShippingEnv.performance_parameters_of_ship['sr1']

        return speed_over_ground, fuel_consumption
