"""
June 22, 2019
Author: V. Mullachery

This is the main class for the gym_shipping OpenAI environment.
Origin is Panama, Destination is Osaka.

Action space discretization includes radial discretization and speed
discretization. Radial discretization with respect to azimuth (zero north
clockwise positive), ranges from -180 to 0. Ship propeller discretization
between 20 and 100 rpm. These values can be changed by editing the ShippingEnv
class variables.

Time quantization is 3600 secs (1hr).

Each input dimension is a box 2D, two real variables: rpm and heading.
Output state space is an np.array of length 18 each a real value. The state
space variable includes the local GFS (wind U and V components, in meters/sec),
local OSCAR (ocean current U and V components in meters/sec), local WAVEHEIGHT
(wave height in meters). These weather values are read in from a local/S3 weather
files. The voyage start date has to be supplied when invoking gym_shipping as:

 env = gym.make("gym_shipping:Shipping-v0", start_datetime=(2019,6,2,17))

Each action incurrs a cost (negative valued reward) based on the fuel consumed.
Fuel consumption is time quantization * fuel consumption rate. The fuel consumption
rate and speed of the ship are computed within this class using a Nautilus labs
ship performance model.

To call this Open AI gym environment, perform these steps in the agent code:

    #start_datetime is mandatory
    env = gym.make("gym_shipping:Shipping-v0", start_datetime=(2019,6,2,17))

    #random number seeding for replicatable results
    env.seed(7)
    env.action_space.seed(7)

    input_dim = 18 # state space dimension
    output_dim = (32+1) * 2  #discretization defined by agent
                             # angular+1 * radial
                             # for instance angular: 32, radial: 2


    # rh is [rpm, heading] list, rpm [0-100], heading=[-180,180]
    #       sample rh = [80, -72.4]
    #       heading is in degrees
    #
    # s: state (size 18)
    # r: reward, a real number
    # done: boolean to indicate end of episode
    # info: a dictionary with miscellaneous data
    #       for instance, reason for end of episode
    #       info['reason'] = 'destination reached' or
    #       info['reason'] = 'timelimit exceeded'
    s, r, done, info = env.step(rh)

References (for Geographic API)
	https://geographiclib.sourceforge.io/html/python/examples.html?highlight=example

File Structure:
 gym-shipping/
    gym_shipping
        envs
            __init__.py
            shipping_env.py (this file)
        test
            __init__.py
            test_shipping_env.py
        __init__.py
    __init__.py
    setup.py

Installation:
    $ source activate condaenv
    $ pip install -e gym-shipping

"""
import datetime
import logging
import math
import random
from collections import OrderedDict
from collections.abc import Iterable

import gym
import numpy as np
from geographiclib.geodesic import Geodesic
from gym import spaces
from keras import utils as kr_utils

import nautlabs.shipperf as ns

logger = logging.getLogger(__name__)

LOCAL_VS_S3 = 'local'

class ShippingEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    Panama = ns.EnvConstants.Panama
    Osaka = ns.EnvConstants.Osaka
    multi_classes = []
    one_hot_actions = []
    discretizations_matrix = []
    time_delta = ns.EnvConstants.TimeDelta
    geod = Geodesic.WGS84
    max_time_allowed = ns.EnvConstants.MAX_TIME_ALLOWED
    proximity = ns.EnvConstants.PROXIMITY # acceptable proximity to (in meters) to destination
    reward_for_destination = ns.EnvConstants.REWARD_FOR_DESTINATION

    def __init__(self, **kwargs):
        self.origin = ShippingEnv.Panama
        self.destination = ShippingEnv.Osaka
        create_discretization_mapping()
        self.WAVEWATCH={}
        self.GFS={}
        self.OSCAR={}
        self.action_space = spaces.Discrete(get_action_space_n())
            # spaces.Box(low=np.array([ns.EnvConstants.RPM_BEGIN, ns.EnvConstants.ANGULAR_BEGIN]), \
            #     high = np.array([ns.EnvConstants.RPM_END, ns.EnvConstants.ANGULAR_END]), \
            #     dtype = np.float32)
        self.action_space.n = get_action_space_n()
        self.observation_space = spaces.Box(low=np.array([-np.infty]*18), \
                                            high=np.array([np.infty]*18))
        # DataFrame of all rpms, headings, corresponding lats/longs, metrics related to the voyage
        #self._reset_dataframe()
        self.action_and_observables={}
        self.save_action_observables = {}
        self.__initialize(datetime.datetime(*kwargs['start_datetime']).timestamp())
        self.viewer = None

    def __initialize(self, timestamp=datetime.datetime.now().timestamp(), timediscretization=3600):
        self.start_timestamp = timestamp
        self.current_timestamp = timestamp
        self.current_location = self.origin
        ShippingEnv.time_delta = timediscretization

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
        if LOCAL_VS_S3 == 's3':
            foldername = ns.get_s3foldername_from_epoch(datetime, weather_type='GFS')
            if foldername not in self.GFS:
                self.GFS[foldername] = ns.get_s3_gfs(datetime)
                #     print(GFS.keys())
                #     print(foldername)
        else:
            foldername = ns.get_local_foldername_from_epoch(datetime, weather_type='GFS')
            if foldername not in self.GFS:
                self.GFS[foldername] = ns.get_local_gfs(datetime)
                #     print(GFS.keys())
                #     print(foldername)
        ndx = ns.location_to_gfs_index(location)
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
        if LOCAL_VS_S3 == 's3':
            foldername = ns.get_s3foldername_from_epoch(datetime, weather_type='OSCAR')
            if foldername not in self.OSCAR:
                    # foldername e.g. 2019060200
                self.OSCAR[foldername] = ns.get_s3_oscar(datetime)
            # print(OSCAR)
        else:
            foldername = ns.get_local_foldername_from_epoch(datetime, weather_type='OSCAR')
            if foldername not in self.OSCAR:
                # foldername e.g. 2019060200
                self.OSCAR[foldername] = ns.get_local_oscar(datetime)
        ndx = ns.location_to_oscar_index(location)
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
        if LOCAL_VS_S3 == 's3':
            foldername = ns.get_s3foldername_from_epoch(datetime, weather_type='WAVEWATCH')
            if foldername not in self.WAVEWATCH:
                self.WAVEWATCH[foldername] = ns.get_s3_wavewatch(datetime)
        else:
            foldername = ns.get_local_foldername_from_epoch(datetime, weather_type='WAVEWATCH')
            if foldername not in self.WAVEWATCH:
                self.WAVEWATCH[foldername] = ns.get_local_wavewatch(datetime)

        ndx = ns.location_to_wavewatch_index(location)
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
        rpm_heading = get_rpm_heading(action)

        self._take_action(rpm_heading)
        ob = self._get_state()

        if (self.current_timestamp - self.start_timestamp > ShippingEnv.max_time_allowed):
            self._episode_over = True
            if not self._info:
                self._info = {'reason': 'timelimit exceeded'}
                dist = ShippingEnv.geod.Inverse(*self.current_location, *self.destination)['s12']
                self._reward +=  np.exp(-(dist)**2 / (2 * (ShippingEnv.proximity) ** 2)) \
                          * ShippingEnv.reward_for_destination

        reward = self._reward
        info = self._info
        episode_over = self._episode_over

        return ob, reward, episode_over, info

    def __reset(self):
        self.current_timestamp = self.start_timestamp
        self.current_location = self.origin
        self.action_and_observables={}
        self._reward = 0
        self._episode_over = False
        #self._reset_dataframe()
        self._reset_action_observables()
        return self._get_state() #

    def __expand_observables(self, a):
        lst = []
        for x in a:
            if isinstance(x, Iterable):
                for i in range(len(x)):
                    lst += x[i],
            else:
                lst += x,
        return lst

    def _reset_action_observables(self):
        self.action_and_observables['rpm'], self.action_and_observables['heading'] = 0, 0
        self.action_and_observables['lat'], self.action_and_observables['lon'] = self.current_location
        self.action_and_observables['timestamp'] = self.current_timestamp
        self.action_and_observables['timedelta'] = ShippingEnv.time_delta
        self.action_and_observables['wave-height'] = \
            self.get_wavewatch(self.current_timestamp, ns.create_location_fragment(self.current_location))
        self.action_and_observables['gfs'] = \
            self.get_gfs(self.current_timestamp, ns.create_location_fragment(self.current_location))
        self.action_and_observables['oscar'] = \
            self.get_oscar(self.current_timestamp, ns.create_location_fragment(self.current_location))
        self.action_and_observables['speed_over_ground'], \
        self.action_and_observables['fuel_consumption_rate'] = \
            ns.sog_and_fuel(self.action_and_observables['oscar'],
                            self.action_and_observables['gfs'],
                            self.action_and_observables['wave-height'],
                            self.action_and_observables['rpm'],
                            self.action_and_observables['heading']
                            )
        self.action_and_observables['origin'] = ShippingEnv.Panama
        self.action_and_observables['destination'] = ShippingEnv.Osaka

        # compute cost is per ShippingEnv.TimeDelta (per hour)
        self.action_and_observables['cost'] = self.action_and_observables['fuel_consumption_rate'] \
                                              * ns.performance_parameters_of_ship['cost_adjustment_factor']
        self.save_action_observables = self.action_and_observables.copy()
        self.action_and_observables.clear()

    def _take_action(self, rpm_heading):
        """
        Takes the action specified by the agent. Steps to the next location, and advances times
        :param action: (rpm, heading)
        :return:
        """

        #update fields and move to next location
        self._compute_cost_reward_and_move_to_next_location(rpm_heading)

        # Advance time
        self.current_timestamp += ShippingEnv.time_delta
        self.save_action_observables = self.action_and_observables.copy()
        self.action_and_observables.clear()

    def _compute_cost_reward_and_move_to_next_location(self, rpm_heading):
        """
        Compute the next location based on current_location, heading, and speed_over_ground, assuming
         we travel for ShippingEnv.TimeDelta
        :return: next_location
        """
        self._reward = 0
        self.action_and_observables['rpm'], self.action_and_observables['heading'] = rpm_heading
        self.action_and_observables['lat'], self.action_and_observables['lon'] = self.current_location
        self.action_and_observables['timestamp'] = self.current_timestamp
        self.action_and_observables['timedelta'] = ShippingEnv.time_delta
        self.action_and_observables['wave-height'] = \
            self.get_wavewatch(self.current_timestamp, ns.create_location_fragment(self.current_location))
        self.action_and_observables['gfs'] = \
            self.get_gfs(self.current_timestamp, ns.create_location_fragment(self.current_location))
        self.action_and_observables['oscar'] = \
            self.get_oscar(self.current_timestamp, ns.create_location_fragment(self.current_location))
        self.action_and_observables['speed_over_ground'], \
        self.action_and_observables['fuel_consumption_rate'] = \
            ns.sog_and_fuel(self.action_and_observables['oscar'],
                            self.action_and_observables['gfs'],
                            self.action_and_observables['wave-height'],
                            self.action_and_observables['rpm'],
                            self.action_and_observables['heading']
                            )
        self.action_and_observables['origin'] = ShippingEnv.Panama
        self.action_and_observables['destination'] = ShippingEnv.Osaka

        # compute cost is per ShippingEnv.TimeDelta (per hour)
        self.action_and_observables['cost'] = self.action_and_observables['fuel_consumption_rate'] \
                                              * ns.performance_parameters_of_ship['cost_adjustment_factor']

        g = ShippingEnv.geod.Direct(*self.current_location,
                                    self.action_and_observables['heading'],
                                    self.action_and_observables['speed_over_ground'] * ShippingEnv.time_delta *
                                    ns.performance_parameters_of_ship['knots_to_meters_adj'], )
        next_location = g['lat2'], g['lon2']
        next_loc_as_vec = self.vector_for_location_wrt_currloc(next_location)
        dest_as_vec = self.vector_for_location_wrt_currloc(ShippingEnv.Osaka)
        projected_length_of_destvec_onto_trajectory = \
            np.linalg.norm(self.projection_of_first_onto_second(dest_as_vec, next_loc_as_vec))
        dist_ortho_to_trajectory = self.orthogonal_complement_length(dest_as_vec, next_loc_as_vec)
        if ((np.linalg.norm(next_loc_as_vec) >= projected_length_of_destvec_onto_trajectory)
             and (dist_ortho_to_trajectory <= ShippingEnv.proximity)):
            #Within acceptable proximity of destination
            dist_along_trajectory = projected_length_of_destvec_onto_trajectory
            next_location = ShippingEnv.Osaka
            timedelta = (dist_along_trajectory + dist_ortho_to_trajectory)/ \
                (self.action_and_observables['speed_over_ground'] * ns.performance_parameters_of_ship['knots_to_meters_adj'])
            self.action_and_observables['cost'] = \
                self.action_and_observables['fuel_consumption_rate'] * \
                (timedelta / ShippingEnv.time_delta) * \
                ns.performance_parameters_of_ship['cost_adjustment_factor']

            self._info = {'reason': 'destination reached'}
            self._episode_over = True
            self._reward += -self.action_and_observables['cost']
            self._reward += np.exp(-(dist_ortho_to_trajectory)**2 / (2 * (ShippingEnv.proximity) ** 2)) \
                      * ShippingEnv.reward_for_destination
        else: #not within proximal
            self.action_and_observables['cost'] = \
                self.action_and_observables['fuel_consumption_rate'] *\
                ns.performance_parameters_of_ship['cost_adjustment_factor']
            self._reward = -self.action_and_observables['cost']
            self._episode_over = False
            self._info = {}

        self.save_action_observables = self.action_and_observables.copy()
        self.current_location = next_location

    def vector_for_location_wrt_currloc(self, loc):
        g = ShippingEnv.geod.Inverse(*self.current_location, *loc)
        ln = g['s12']
        az = g['azi1']
        return np.array([ln * np.sin(math.radians(az)), ln*np.cos(math.radians(az))])

    def orthogonal_complement_length(self, v, a):
        p = self.projection_of_first_onto_second(v, a)
        return np.linalg.norm(v - p)

    def projection_of_first_onto_second(self, first, second):
        return np.dot(np.outer(second, second), first) / np.dot(second.T, second)

    def _get_state(self):
        """
        Compute the state tuple. State consists of WAVEWATCH dictionary, GFS dictionary, OSCAR dictionary,
         computed action and observables data record, current location, current timestamp, origin, destination
        :return: state tuple
        """
        self.save_action_observables = OrderedDict(sorted(self.save_action_observables.items()))
        return np.array(self.__expand_observables(self.save_action_observables.values()))

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

    def _height(self, xs):
        return np.sin(3 * xs) * .45 + .55

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


    def seed(self, num):
        """
        Set the random number seed
        :param num:
        :return:
        """
        return self.__seed(num)

def create_discretization_mapping():
    ShippingEnv.multi_classes = list(range(ns.EnvConstants.N_ANGULAR * ns.EnvConstants.N_RPMS))
    ShippingEnv.one_hot_actions = kr_utils.to_categorical(ShippingEnv.multi_classes)
    a = np.array(list(ns.EnvConstants.RANGE_RPMS))
    b = np.array(list(ns.EnvConstants.RANGE_HEADINGS))
    m = np.array(np.meshgrid(a,b))
    ShippingEnv.discretizations_matrix = np.reshape(m, (2, (ns.EnvConstants.N_ANGULAR + 1) * ns.EnvConstants.N_RPMS))

def get_nearest_action_from_rpm_heading(rh):
    _interim = ShippingEnv.discretizations_matrix / \
               np.linalg.norm(ShippingEnv.discretizations_matrix, ord=2, axis=0, keepdims=True)
    outcome = rh @ _interim
    return np.argmax(outcome)

def get_rpm_heading(action):
    return np.array(ShippingEnv.discretizations_matrix[:,action])

def get_action_space_n():
    return (ns.EnvConstants.N_ANGULAR + 1) * ns.EnvConstants.N_RPMS

