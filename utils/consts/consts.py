# Constants variables for the GymEnv Class will never change
MODEL_TYPE = 'CnnPolicy'
N_ACTIONS = 3  # Number of actions in the scenario
OBSERVATION_SPACE_SHAPE = (100, 160, 1)  # Target shape for the CNN
SKIP_FRAMES = 4  # Number of frames to skip
SAVE_MODEL_FREQUENCY = 10000  # Frequency to save the model
N_EVAL_EPISODES = 20  # Number of episodes to evaluate the model
VERSBOSE = 2 # Verbose level for the model

# Gym Box Configuration
LOW = 0
HIGH = 255

# Default PPO Model Hyperparameters
LEARNING_RATE = 0.0003
CLIP_RANGE = .2
GAMMA = .99
GAE_LAMBDA = .95
N_STEPS = 2048
N_TIMESTEPS = 100000

# Reward Shaping weights
MOVEMENT_W = -2
HITCOUNT_DELTA_WEIGHT = 250
DAMAGE_TAKEN_DELTA_WEIGHT = 5
DAMAGECOUNT_DELTA_WEIGHT = 200
AMMO_DELTA_WEIGHT = 3

# String constants for Train Class
SCENARIO_PATH = 'SCENARIO_PATH'
IS_REWARD_SHAPED = 'IS_REWARD_SHAPED'

GAME_STAGES = [
    'basic',
    'defend_the_center',
    'deadly_corridor',
    'defend_the_line',
    'predict_position',
    'my_way_home',
    'take_cover',
    'health_gathering',
    'health_gathering_supreme'
]
