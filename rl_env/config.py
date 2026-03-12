import numpy as np

MAP_SIZE = 100.0
GRID_SIZE = 1.0

NUM_OBSTACLES = 5
OBSTACLE_RADIUS = 3.0
MIN_GOAL_DIST = 60.0
GOAL_RADIUS = 5.0

VEL_NOISE_STD = 0.02
POS_NOISE_STD = 0.05

DRONE_CONFIGS = {
    'SCOUT':{
        'color':np.array([0.85,0.35,0.35]),
        'mass':0.5,
        'max_sense_range':60.0,
        'max_comm_range':40.0,
        'max_speed':1.0,
        'max_accel': 3.0,
        'fric_coeff': 0.1,
        'fov':90,
        'reward_weight':1.0,
        'max_endurance':500.0,
        'battery_capacity':100.0,
        'z_cost_factor':0.2,
        'role_id':0,
    },
    'RELAY':{
        'color':np.array([0.35,0.85,0.35]),
        'mass':1.0,
        'max_sense_range':30.0,
        'max_comm_range':100.0,
        'max_speed':0.8,
        'max_accel': 1.5,
        'fric_coeff': 0.3,
        'fov':60,
        'reward_weight':0.5,
        'max_endurance':400.0,
        'battery_capacity':80.0,
        'z_cost_factor':0.3,
        'role_id':1,
    },
    'EXECUTOR':{
        'color':np.array([0.35,0.35,0.85]),
        'mass':2.0,
        'max_sense_range':30.0,
        'max_comm_range':40.0,
        'max_speed':2.0,
        'max_accel': 1.0,
        'fric_coeff': 0.6,
        'fov':60,
        'reward_weight':2.0,
        'max_endurance':250.0,
        'battery_capacity':50.0,
        'z_cost_factor':0.8,
        'role_id':2,
    }
}

OBS_CONFIG = {
    "dim_self":10,
    "dim_obs_item":2,
    "dim_neigh_item":3,
}