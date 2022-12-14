SCENARIO_PATH = 'utils/scenarii/deadly_corridor_doom_skill_3.cfg'  # Game Scenario Path
LOG_DIR = 'logs/log_deadly_corridor/curriculum_1'  # Tensorboard Log Directory
CHECKPOINT_DIR = 'train/train_deadly_corridor/curriculum_1'  # Checkpoint Directory
MODEL_PATH = 'utils/models/deadly_corridor/curriculum_1/best_model_400000.zip'  # Model Path
# MODEL_PATH = 'utils/models/deadly_corridor/curriculum_4/best_model_400000.zip'  # Model Path
MODEL_NAME = 'RewardingGymEnvModel'  # Model Name
N_ACTIONS = 7
IS_REWARD_SHAPED = True  # Is the reward shaped ?
IS_CURRICULUM = True  # Is the curriculum learning enabled ?
CURRICULUM_PATHS = [
    'utils/scenarii/deadly_corridor_doom_skill_1.cfg',
    'utils/scenarii/deadly_corridor_doom_skill_2.cfg',
    'utils/scenarii/deadly_corridor_doom_skill_3.cfg',
    'utils/scenarii/deadly_corridor_doom_skill_4.cfg',
    'utils/scenarii/deadly_corridor_doom_skill_5.cfg',
]

# PPO Model Hyperparameters
LEARNING_RATE = 0.00001
CLIP_RANGE = .1
GAMMA = .9
GAE_LAMBDA = .99
N_STEPS = 8192
N_TIMESTEPS = 500000

# Reward Shaping weights
DAMAGE_TAKEN_DELTA_WEIGHT = 10
HITCOUNT_DELTA_WEIGHT = 200
DAMAGECOUNT_DELTA_WEIGHT = 200
AMMO_DELTA_WEIGHT = 5

DAMAGE_TAKEN = 0
DAMAGECOUNT = 0
AMMO = 52
