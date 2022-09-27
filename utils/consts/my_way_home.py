SCENARIO_PATH = 'utils/scenarii/my_way_home.cfg'  # Game Scenario Path
LOG_DIR = 'logs/log_my_way_home'  # Tensorboard Log Directory
CHECKPOINT_DIR = 'train/train_my_way_home'  # Checkpoint Directory
MODEL_PATH = 'utils/models/my_way_home_2709'  # Model Path
MODEL_NAME = 'BasicGymEnvModel'  # Model Name
N_ACTIONS = 3
IS_REWARD_SHAPED = False  # Is the reward shaped?
IS_CURRICULUM = False  # Is the curriculum learning enabled?

# PPO Model Hyperparameters
LEARNING_RATE = 0.0001
CLIP_RANGE = .2
GAMMA = .99
GAE_LAMBDA = .95
N_STEPS = 2048
N_TIMESTEPS = 100000
