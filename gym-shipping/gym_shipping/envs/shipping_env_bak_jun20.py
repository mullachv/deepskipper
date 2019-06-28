import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pandas as pd
import datetime
import sys
import math
import numpy as np

from geographiclib.geodesic import Geodesic
import random
import nautlabs.shipperf as ns

from collections.abc import Iterable
#from gym.envs.classic_control import rendering

import logging
logger = logging.getLogger(__name__)

LOCAL_VS_S3 = 'local'
class ShippingEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    #basedatafolder = '/Users/vmullachery/mine/insight/'
    #defaultfolder = '2019060200'
    #Panama = (8.517163333, -79.45016167)
    Panama = (9.279095 , -90.97440333) #waypoint 50
    #Osaka = (34.095695, 144.997235) # real Osaka
    #Osaka = (9.620595,	-92.30147167) #waypoint 55
    Osaka = (12.76326667, - 104.7082783) #waypoint 100
    #Osaka = (26.182365, -158.63953) #waypoint 300
    #Osaka = (9.420316667, -91.50667667) #waypoint 52

    TimeDelta = 3600
    geod = Geodesic.WGS84
    #MAX_TIME_ALLOWED = 505*3600 # 505 hours
    MAX_TIME_ALLOWED = 70*3600 # 200 hours
    PROXIMITY = 20000 # acceptable proximity to (in meters) to destination
    __DESTINATION_NOT_REACHED_MULTIPLIER = 1e-2
    __REWARD_FOR_DESTINATION = 100

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
        self.action_space.n = 2
        self.observation_space = spaces.Box(low=np.array([-np.infty]*18), \
                                            high=np.array([np.infty]*18))
        # DataFrame of all rpms, headings, corresponding lats/longs, metrics related to the voyage
        self._reset_dataframe()
        self.action_and_observables={}
        self.save_action_observables = {}
        self.__initialize(datetime.datetime(*kwargs['start_datetime']).timestamp())
        self.viewer = None

    def __initialize(self, timestamp=datetime.datetime.now().timestamp(), timediscretization=3600):
        self.start_timestamp = timestamp
        self.current_timestamp = timestamp
        self.current_location = self.origin
        ShippingEnv.TimeDelta = timediscretization

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
        self._take_action(action)
        reward = self._get_reward()
        ob = self._get_state()
        info = {}
        episode_over = False
        # episode_over = \
        #     ( ( ShippingEnv.geod.Inverse(*self.current_location, *self.destination)['s12'] <= ShippingEnv.PROXIMITY ) or
        #     (self.current_timestamp - self.start_timestamp >= ShippingEnv.MAX_TIME_ALLOWED) )

        if ( ShippingEnv.geod.Inverse(*self.current_location, *self.destination)['s12'] <= ShippingEnv.PROXIMITY ):
            info = { 'reason': 'destination reached'}
            episode_over = True

        if (self.current_timestamp - self.start_timestamp >= ShippingEnv.MAX_TIME_ALLOWED):
            info = {'reason': 'timelimit exceeded'}
            episode_over = True

        #penalty for not reaching the detination
        if (self.current_timestamp - self.start_timestamp >= ShippingEnv.MAX_TIME_ALLOWED):
            reward += -ShippingEnv.geod.Inverse(*self.current_location, *self.destination)['s12'] \
                      * ShippingEnv.__DESTINATION_NOT_REACHED_MULTIPLIER

        return ob, reward, episode_over, info

    def _reset_dataframe(self):
        self.wf = pd.DataFrame(columns=['rpm','heading','lat','lon','timestamp',\
            'timedelta','wave-height','gfs','oscar','speed_over_ground','fuel_consumption_rate','cost',
                                        'origin','destination'])

    def __reset(self):
        self.current_timestamp = self.start_timestamp
        self.current_location = self.origin
        self.action_and_observables={}
        self._reward = 0
        self._reset_dataframe()
        self.populate_action_observables((0,0))
        return self._get_state() #
        # return np.array([0]*14) #
        #return self._get_state() #, self._reward, False, {}

    def __expand_observables(self, a):
        lst = []
        for x in a:
            if isinstance(x, Iterable):
                for i in range(len(x)):
                    lst += x[i],
            else:
                lst += x,
        return lst

    # def __render(self, mode='human', close=False):
    #     pass
	#

    def _take_action(self, action):
        """
        Takes the action specified by the agent. Steps to the next location, and advances times
        :param action: (rpm, heading)
        :return:
        """
        self.populate_action_observables(action)

        next_location = self._compute_next_location()

        # -1 for time/step increment, so incentivizes to reach destination quicker
        #self._reward = -self.action_and_observables['cost'] - 1
        self._reward = 0  # nothing for the time step
        if self._do_we_passthrough_destination(next_location):
            self.current_location = ShippingEnv.Osaka
            self._reward += -self.get_adjusted_cost_if_passing_through_destination(next_location)
            self._reward += ShippingEnv.__REWARD_FOR_DESTINATION #positive reward,
        else:
            # # -ShippingEnv.geod.Inverse(*ShippingEnv.Panama, *ShippingEnv.Osaka)['s12'] * 1. / \
            # #     ShippingEnv.geod.Inverse(*self.current_location, *ShippingEnv.Osaka)['s12']  \
            # - 1
            self._reward += -self.action_and_observables['cost'] #fuel cost
            #Now move the ship
            # g = ShippingEnv.geod.Direct(self.current_location[0],
            #     self.current_location[1],
            #     self.action_and_observables['heading'],
            #     self.action_and_observables['speed_over_ground']*ShippingEnv.TimeDelta*
            #     ShippingEnv.performance_parameters_of_ship['knots_to_meters_adj'], )
            #print('geod', g)
            #self.current_location = g['lat2'], g['lon2']
            self.current_location = next_location

        # Advance time
        self.current_timestamp += ShippingEnv.TimeDelta
        self.save_action_observables = self.action_and_observables.copy()
        self.action_and_observables.clear()

    def populate_action_observables(self, action):
        self.action_and_observables['rpm'], self.action_and_observables['heading'] = action
        self.action_and_observables['lat'], self.action_and_observables['lon'] = self.current_location
        self.action_and_observables['timestamp'] = self.current_timestamp
        self.action_and_observables['timedelta'] = ShippingEnv.TimeDelta
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
        # compute cost is per ShippingEnv.TimeDelta (per hour)
        self.action_and_observables['cost'] = self.action_and_observables['fuel_consumption_rate'] \
                                              * ShippingEnv.performance_parameters_of_ship['cost_adjustment_factor']
        self.action_and_observables['origin'] = ShippingEnv.Panama
        self.action_and_observables['destination'] = ShippingEnv.Osaka
        #self.wf = self.wf.append(self.action_and_observables, ignore_index=True)

        self.save_action_observables = self.action_and_observables.copy()

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
    def _do_we_passthrough_destination(self, next_location):
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

    def vector_for_location_wrt_currloc(self, loc):
        g = ShippingEnv.geod.Inverse(*loc, self.current_location)
        ln = g['s12']
        az = g['azi1']
        return ln * np.sin(math.radians(az)), ln*np.cos(math.radians(az))

    def orthogonal_complement_dist(self, v, a):
        p = self.projection_of_v_onto_a(v, a)
        return v - p

    def projection_of_v_onto_a(self, v, a):
        return (np.outer(a, a) @ v ) / (a.T @ a)

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
        if LOCAL_VS_S3 == 's3':
            gfs = ns.get_s3foldername_from_epoch(self.current_timestamp, weather_type='GFS')
            oscar = ns.get_s3foldername_from_epoch(self.current_timestamp, weather_type='OSCAR')
            ww = ns.get_s3foldername_from_epoch(self.current_timestamp, weather_type='WAVEWATCH')
        else:
            gfs = ns.get_local_foldername_from_epoch(self.current_timestamp, weather_type='GFS')
            oscar = ns.get_local_foldername_from_epoch(self.current_timestamp, weather_type='OSCAR')
            ww = ns.get_local_foldername_from_epoch(self.current_timestamp, weather_type='WAVEWATCH')

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

    def render(self, mode='human'):
        # self.min_action = -1.0
        # self.max_action = 1.0
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.45  # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
        self.power = 0.0015

        self.state = np.array([0, -1])

        screen_width = 600
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width / world_width
        carwidth = 40
        carheight = 20

        if self.viewer is None:
            #from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = self._height(xs)
            xys = list(zip((xs - self.min_position) * scale, ys * scale))

            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            clearance = 10

            l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
            car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = rendering.make_circle(carheight / 2.5)
            frontwheel.set_color(.5, .5, .5)
            frontwheel.add_attr(rendering.Transform(translation=(carwidth / 4, clearance)))
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight / 2.5)
            backwheel.add_attr(rendering.Transform(translation=(-carwidth / 4, clearance)))
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(.5, .5, .5)
            self.viewer.add_geom(backwheel)
            flagx = (self.goal_position - self.min_position) * scale
            flagy1 = self._height(self.goal_position) * scale
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon([(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)])
            flag.set_color(.8, .8, 0)
            self.viewer.add_geom(flag)

        pos = self.state[0]
        self.cartrans.set_translation((pos - self.min_position) * scale, self._height(pos) * scale)
        self.cartrans.set_rotation(math.cos(3 * pos))


        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

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
