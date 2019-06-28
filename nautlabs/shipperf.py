"""
June 22, 2019
Author: V. Mullachery

This is a library with various utility functions for fetching the weather (GFS, OSCAR and
WAVE-WATCH), and for computing speed, fuel consumption rate based on performance parameters
of the ship. This library can read the weather files off the S3 bucket or the local filesystem.

Usage:
	import nautlabs.shipperf as ns
	ns.get_s3_oscar(datetime)
	ns.location_to_oscar_index(location)

"""
import datetime
import json
import logging
import math
import os.path as path

import boto3
import botocore
import numpy as np
import pandas as pd
from botocore.exceptions import ClientError
from geographiclib.geodesic import Geodesic

geod = Geodesic.WGS84

__s3_profile_name = 'mullachv-IAM'
boto3.setup_default_session(profile_name = __s3_profile_name)
__defaultbucketname = 'gfs1oscar1waveht'
__basedatafolder = '/Users/vmullachery/mine/insight/'
__defaultfolder = '2019060200'
__DATE_RETRACTCOUNT = 10
# for numerical stability for log10 in fuel, speed computation
__NUM_ADJ_DELTA = 1e-10
performance_parameters_of_ship = {
	# Factors affecting fuel consumption
	'fw1': 0.067302,  # wind factor
	'fw2': 0.002554,  # wind squared factor
	'fc1': 0.092372,  # current factor
	'fc2': 0.001005,  # current squared factor
	'fh1': 0.150907,  # wave height factor
	'fr1': -0.105935,  # propeller rpm factor
	'fr3': 0.000066,  # propeller rpm cubed factor
	'flr3': 1.273963,  # propeller log10 rpm cubed factor

	# Factors affecting ground speed
	'sw1': -0.0388,  # wind factor
	'sw2': -0.0006,  # wind squared factor
	'sc1': 1.0443,  # current factor
	'sc2': -0.1531,  # current squared factor
	'sh1': -0.2510,  # wave height factor
	'sr1': 0.1954,  # propeller rpm factor

	# Cost adjustment factor
	'cost_adjustment_factor': 1. / (24),  # timedelta is in seconds, 3600 seconds = 1hr.
	# fuel (consumption rate) is in per day
	'knots_to_meters_adj': 0.514444444  # Adjust knots to meters/sec for Geodesic API
}


def create_location_fragment(location):
	"""
	Creates a location dictionary object with lat, lon
	:param location:
	:return: a dictionary object {'Lat': lat, 'Lng': lon } for the location
	"""
	return {'Lat': location[0], 'Lng': location[1]}


def __vector_direction(vx, vy):
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
	return math.degrees(np.arctan(vx / vy))

def __composite_value( speed, direction, heading):
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


def compute_perf_metrics( wind_speed, wind_direction,
						 oceanic_current_speed, oceanic_current_direction,
						 wave_height, propeller_rpm, ship_heading):
	"""
	Computes speed over ground and fuel consumption of the ship
	Input:
		wind_speed: in m/s np.sqrt(windU^2 + windV^2)
		wind_direction: in degrees math.degrees(np.arctan(windU/windV))
		oceanic_current_speed: in m/s np.sqrt(currentU^2 + currentV^2)
		oceanic_current_direction: in degrees math.degrees(np.arctan(currentU/currentV))
		wave_height: in meters
		ship_heading: in degrees

	Returns:
		speed_over_ground in knots
		fuel_consumption in metric tonnes per day

	"""
	#
	# ship_heading: 0 degrees is North. Clockwise is positive (I think)
	#
	composite_wind = __composite_value(wind_speed, wind_direction, ship_heading)
	composite_current = __composite_value(oceanic_current_speed, oceanic_current_direction, ship_heading)

	return sog_fuel_from_composite(propeller_rpm, composite_wind, composite_current, wave_height)

def sog_and_fuel( oscar_vec, gfs_vec, swh, rpm, heading):
	"""
	Compute speed over ground (sog) and fuel consumption rate
	:param oscar_vec: (lat, lon)
	:param gfs_vec: (lat, lon)
	:param swh: in meters)
	:param rpm: propeller revolutions per minute
	:param heading: in degrees from the North going clockwise
	:return: speed over ground in knots, fuel consumption rate in metric tonnes per day
	"""
	gfs_speed = np.linalg.norm(gfs_vec)
	gfs_dir = __vector_direction(*gfs_vec)

	oscar_speed = np.linalg.norm(oscar_vec)
	oscar_dir = __vector_direction(*oscar_vec)

	return compute_perf_metrics(gfs_speed, gfs_dir, oscar_speed, oscar_dir, swh, rpm, heading)


def sog_fuel_from_composite( rpm, comp_wind, comp_current, wave_height):
	"""
	Compute speed over ground (in knots), fuel consumption rate (in metric tonnes per day)
	from rpm, composite wind, composite current, wave height
	:param rpm:
	:param comp_wind:
	:param comp_current:
	:param wave_height:
	:return:
	"""
	log10adj = max(0, np.log10(np.abs(rpm)+__NUM_ADJ_DELTA))

	# fuel consumption rate in (metric tons/day)
	fuel_consumption = comp_wind * performance_parameters_of_ship['fw1'] \
					   + (comp_wind ** 2) * performance_parameters_of_ship['fw2'] \
					   + comp_current * performance_parameters_of_ship['fc1'] \
					   + (comp_current ** 2) * performance_parameters_of_ship['fc2'] \
					   + wave_height * performance_parameters_of_ship['fh1'] \
					   + rpm * performance_parameters_of_ship['fr1'] \
					   + (rpm ** 3) * performance_parameters_of_ship['fr3'] \
					   + 3 * log10adj * performance_parameters_of_ship['flr3']
	#in knots
	speed_over_ground = comp_wind * performance_parameters_of_ship['sw1'] \
						+ (comp_wind ** 2) * performance_parameters_of_ship['sw2'] \
						+ comp_current * performance_parameters_of_ship['sc1'] \
						+ (comp_current ** 2) * performance_parameters_of_ship['sc2'] \
						+ wave_height * performance_parameters_of_ship['sh1'] \
						+ rpm * performance_parameters_of_ship['sr1']
	speed_over_ground = max(__NUM_ADJ_DELTA, speed_over_ground)
	return speed_over_ground, fuel_consumption


def get_local_foldername_from_epoch(timestamp, weather_type='GFS'):
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
		'GFS': __check_hour(hour),
		'OSCAR': '00',
		'WAVEWATCH': __check_hour(hour)
	}
	foldername = basename + check[weather_type]
	if weather_type in ['GFS', 'WAVEWATCH']:
		if not (path.exists(path.join(__basedatafolder, foldername))):
			foldername = __defaultfolder
	elif weather_type == 'OSCAR':
		counter = 10
		while not (path.exists(path.join(__basedatafolder, foldername))) and counter > 0:
			dttmstamp = dttmstamp - datetime.timedelta(1)
			foldername = datetime.datetime.strftime(dttmstamp, '%Y%m%d') + '00'
			counter -= 1
		if counter <= 0:
			# hard code to defaultfolder
			foldername = __defaultfolder
	return foldername

def get_s3foldername_from_epoch(timestamp, weather_type='GFS'):
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
		'GFS': __check_hour(hour),
		'OSCAR': '00',
		'WAVEWATCH': __check_hour(hour)
	}
	foldername = basename + check[weather_type]
	if weather_type in ['GFS', 'WAVEWATCH']:
		if not s3_keyexists(foldername):
			foldername = __defaultfolder
	elif weather_type == 'OSCAR':
		counter = __DATE_RETRACTCOUNT
		while not (s3_keyexists(foldername)) and counter > 0:
			dttmstamp = dttmstamp - datetime.timedelta(1)
			foldername = datetime.datetime.strftime(dttmstamp, '%Y%m%d') + '00'
			counter -= 1
		if counter <= 0:
			# hardcode to defaultfolder
			foldername = __defaultfolder
	return foldername

def s3_keyexists(key_name):
	s3 = boto3.resource('s3')
	try:
		s3.Object(__defaultbucketname, key_name).load()
	except botocore.exceptions.ClientError as e:
		if e.response['Error']['Code'] == "404":
			return False
		else:
			# Something else has gone wrong.
			raise e
	else:
		return True


def __check_hour(hour):
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

def oscar_index_to_location(index):
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

def wavewatch_index_to_location(index):
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

def gfs_index_to_location(index):
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

def location_to_oscar_index(location):
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

def location_to_gfs_index(location):
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

def location_to_wavewatch_index( location):
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

def local_read(foldername, filename):
	foldername = path.join(__basedatafolder, foldername)
	with open(path.join(foldername, filename)) as file:
		return json.load(file)

def s3_read(key_filename):
    """
    Read a JSON file from an S3 source, into JSON dictionary

    Parameters
    ----------
    source : str
        Path starting with s3://, e.g. 's3://bucket-name/key/foo.bar'
    profile_name : str, optional
        AWS profile

    Returns
    -------
    content : JSON dictionary of the bytes of foo.bar

    botocore.exceptions.NoCredentialsError
        Botocore is not able to find your credentials. Either specify
        profile_name or add the environment variables AWS_ACCESS_KEY_ID,
        AWS_SECRET_ACCESS_KEY and AWS_SESSION_TOKEN.
        See https://boto3.readthedocs.io/en/latest/guide/configuration.html
    """
    session = boto3.Session(profile_name=__s3_profile_name)
    s3 = session.client('s3')
    try:
        obj = s3.get_object(Bucket=__defaultbucketname, Key=key_filename)
        return json.loads(obj['Body'].read().decode())
    except ClientError as cl:
        if cl.response['Error']['Code'] ==  'NoSuchKey':
            if key_filename[:len(__defaultfolder)] != __defaultfolder:
                logging.info('No key found: ' + key_filename + ', returning default: '+__defaultfolder)
                return (s3_read(__defaultfolder+key_filename[len(__defaultfolder):]))
            raise cl
        else:
            raise cl


def get_local_gfs( datetime):
	"""
	Returns the wind velocity vector at the time and location
	Input:
		datetime in epoch time (numbers)
		location in latitude, longitude: {'Lat': 24.4, 'Lng': -2.5}
	Output:
		ocean wind velocity vector (a tuple): (u, v) in meters/second. u = latitudinal speed
		v = longitudinal speed
	Side effect: loads the GFS dictionary for the YYYYMMDDHH corresponding to
	datetime
	"""
	foldername = get_local_foldername_from_epoch(datetime, weather_type='GFS')
	wind_data = local_read(foldername, 'wind.json')
	wind_u = wind_data[0]['data']
	wind_v = wind_data[1]['data']
	du = pd.DataFrame(wind_u, columns=['windU'])
	dv = pd.DataFrame(wind_v, columns=['windV'])

	# Clean up
	du[du['windU'].isna()] = 0
	dv[dv['windV'].isna()] = 0

	return [(u, v) for (u, v) in zip(du['windU'].values, dv['windV'].values)]

def get_s3_gfs( datetime):
	"""
	Returns the wind velocity vector at the time and location
	Input:
		datetime in epoch time (numbers)
		location in latitude, longitude: {'Lat': 24.4, 'Lng': -2.5}
	Output:
		ocean wind velocity vector (a tuple): (u, v) in meters/second. u = latitudinal speed
		v = longitudinal speed
	Side effect: loads the GFS dictionary for the YYYYMMDDHH corresponding to
	datetime
	"""
	foldername = get_s3foldername_from_epoch(datetime, weather_type='GFS')
	wind_data = s3_read(foldername+'/wind.json')
	wind_u = wind_data[0]['data']
	wind_v = wind_data[1]['data']
	du = pd.DataFrame(wind_u, columns=['windU'])
	dv = pd.DataFrame(wind_v, columns=['windV'])

	# Clean up
	du[du['windU'].isna()] = 0
	dv[dv['windV'].isna()] = 0

	return [(u, v) for (u, v) in zip(du['windU'].values, dv['windV'].values)]


def get_local_oscar(datetime):
	"""
	Returns the ocean current vector at the time and location
	Input:
		datetime in epoch time (numbers)
		location in latitude, longitude: {'Lat': 24.4, 'Lng': -2.5}
	Output:
		ocean current vector (a tuple): (u, v) in meters/second. u = latitudinal speed
		v = longitudinal speed
	Side effect: loads the OSCAR dictionary for the YYYYMMDD00 corresponding to
	datetime
	"""

	foldername = get_local_foldername_from_epoch(datetime, weather_type='OSCAR')
	oscar_data = local_read(foldername, 'oscar.json')
	oscar_data_u = oscar_data[0]['data']
	oscar_data_v = oscar_data[1]['data']
	du = pd.DataFrame(oscar_data_u, columns=['data'])
	dv = pd.DataFrame(oscar_data_v, columns=['data'])

	# Clean up
	du[du['data'].isna()] = 0
	dv[dv['data'].isna()] = 0

	return [(u, v) for (u, v) in zip(du['data'].values, dv['data'].values)]


def get_s3_oscar( datetime):
	"""
	Returns the ocean current vector at the time and location
	Input:
		datetime in epoch time (numbers)
		location in latitude, longitude: {'Lat': 24.4, 'Lng': -2.5}
	Output:
		ocean current vector (a tuple): (u, v) in meters/second. u = latitudinal speed
		v = longitudinal speed
	Side effect: loads the OSCAR dictionary for the YYYYMMDD00 corresponding to
	datetime
	"""
	foldername = get_s3foldername_from_epoch(datetime, weather_type='OSCAR')
	oscar_data = s3_read(foldername+'/oscar.json')
	oscar_data_u = oscar_data[0]['data']
	oscar_data_v = oscar_data[1]['data']
	du = pd.DataFrame(oscar_data_u, columns=['data'])
	dv = pd.DataFrame(oscar_data_v, columns=['data'])

	# Clean up
	du[du['data'].isna()] = 0
	dv[dv['data'].isna()] = 0

	return [(u, v) for (u, v) in zip(du['data'].values, dv['data'].values)]

def get_local_wavewatch( datetime):
	"""
	Returns the wave-height at the time and location
	Input:
		datetime in epoch time (numbers)
		location in latitude, longitude: {'Lat': 24.4, 'Lng': -2.5}
	Output:
		wave height in meters
	Side effect: loads the WAVEWATCH dictionary for the YYYYMMDDHH corresponding to
	datetime
	"""
	foldername = get_local_foldername_from_epoch(datetime, weather_type='WAVEWATCH')
	wave_data = local_read(foldername, 'wave-height.json')
	swh = wave_data[0]['data']
	dw = pd.DataFrame(swh, columns=['swh'])
	dw[dw['swh'] == 'NaN'] = 0
	return dw['swh'].values

def get_s3_wavewatch( datetime):
	"""
	Returns the wave-height at the time and location
	Input:
		datetime in epoch time (numbers)
		location in latitude, longitude: {'Lat': 24.4, 'Lng': -2.5}
	Output:
		wave height in meters
	Side effect: loads the WAVEWATCH dictionary for the YYYYMMDDHH corresponding to
	datetime
	"""
	foldername = get_s3foldername_from_epoch(datetime, weather_type='WAVEWATCH')
	wave_data = s3_read(foldername+'/wave-height.json')
	swh = wave_data[0]['data']
	dw = pd.DataFrame(swh, columns=['swh'])
	dw[dw['swh'] == 'NaN'] = 0
	# foldername e.g. 2019060212
	return dw['swh'].values

def get_random_along_geodesic_from_loc_to_dest(currloc, dest):
	return EnvConstants.RPM_BEGIN + (EnvConstants.RPM_END - EnvConstants.RPM_BEGIN) * np.random.random(), \
		 geod.Inverse(*currloc, *dest)['azi1']

class EnvConstants():
    RPM_BEGIN, RPM_END = 80, 100
    ANGULAR_BEGIN, ANGULAR_END = -180, 90
    RPM_STEP = 20
    N_ANGULAR = 48
    N_RPMS = int((RPM_END - RPM_BEGIN) / RPM_STEP)
    RANGE_RPMS = range(RPM_BEGIN, RPM_END, RPM_STEP)
    RANGE_HEADINGS = np.linspace(ANGULAR_BEGIN, ANGULAR_END, N_ANGULAR + 1)  # into 64 intervals
    # Panama = (8.517163333, -79.45016167) # real Panama
    Panama = (9.279095, -90.97440333)  # waypoint 50
    # Osaka = (34.095695, 144.997235) # real Osaka
    # Osaka = (9.620595,	-92.30147167) #waypoint 55
    # Osaka = (12.76326667, - 104.7082783) #waypoint 102
    # Osaka = (26.182365, -158.63953) #waypoint 300
    # Osaka = (9.420316667, -91.50667667) #waypoint 52
    Osaka = (30.19035667, -180 - 180 + 160.273475)  # waypoint 450 - unrolled longitude
    TimeDelta = 3600
    MAX_TIME_ALLOWED = 420 * 3600  # 275 hours
    PROXIMITY = 20000  # acceptable proximity to (in meters) to destination
    REWARD_FOR_DESTINATION = 10000
    column_names = ['rpm', 'heading', 'lat', 'lon', 'timestamp', 'wave-height', 'gfs',
					'oscar', 'speed_over_ground', 'fuel_consumption_rate',
					'cost', 'origin', 'destination']


def save_states_to_file(states, file='outputs/pol_grad_eval_seeded.csv'):
	#
	# Returned state values from the Engine are in an OrderedDict with alphabetically sorted
	# key values
	#
	# [('cost', 0.01737912478920163),  0
	# ('destination', (26.182365, -158.63953)),  1-2
	# ('fuel_consumption_rate', 0.41709899494083919),  3
	# ('gfs', (-7.6492849999999999, -2.1053099999999998)), 4-5
	# ('heading', 0), 6
	# ('lat', 9.279095), 7
	# ('lon', -90.97440333), 8
	#  ('origin', (9.279095, -90.97440333)),  9-10
	#  ('oscar', (0.0, 0.0)), 11-12
	#  ('rpm', 0), 13
	#  ('speed_over_ground', 1e-10), 14
	#  ('timedelta', 3600), 15
	#  ('timestamp', 1560373200.0), 16
	#  ('wave-height', 1.75)]) 17
    wf = create_dataframe()
    for i, observation in enumerate(states):
        if i == 0:
            continue
        cost = observation[0]
        dest = observation[1:3]
        fuel = observation[3]
        gfs = (observation[4:6])
        heading, lat, lon = observation[6:9]
        origin = observation[9:11]
        oscar = observation[11:13]
        rpm, sog, timedelta, timestamp, wh = observation[13:]
        ser = pd.Series({'rpm': rpm,
                         'heading': heading,
                         'lat': lat,
                         'lon': lon,
                         'timestamp': timestamp,
                         'timedelta': timedelta,
                         'wave-height': wh,
                         'gfs': gfs,
                         'oscar': oscar,
                         'speed_over_ground': sog,
                         'fuel_consumption_rate': fuel,
                         'cost': cost,
                         'origin': origin,
                         'destination': dest
                         })
        wf = wf.append(ser, ignore_index=True)
    wf.to_csv(file, columns=EnvConstants.column_names)


def create_dataframe():
    wf = pd.DataFrame(columns={*EnvConstants.column_names})
    return wf