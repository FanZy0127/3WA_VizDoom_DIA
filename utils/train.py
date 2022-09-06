import importlib
from DoomAi.utils.consts.consts import *
from DoomAi.classes.Callback import TrainAndLoggingCallback
from stable_baselines3.common import env_checker
from stable_baselines3 import PPO


class Train:
    def __init__(self, level_name):
        self.level_name = level_name
        self.load_constants()

    def train(self):
        env = self.get_env()
        callback = TrainAndLoggingCallback(check_freq=SAVE_MODEL_FREQUENCY, save_path=self.checkpoint_dir)

        # Check the environment
        env_checker.check_env(env)

        model = PPO(MODEL_TYPE,
                    env,
                    tensorboard_log=self.log_dir,
                    verbose=VERSBOSE,
                    learning_rate=self.learning_rate,
                    clip_range=self.clip_range,
                    gamma=self.gamma,
                    gae_lambda=self.gae_lambda,
                    n_steps=self.n_steps)

        model.learn(total_timesteps=self.n_timesteps, callback=callback)

    def get_env(self):
        module = importlib.import_module(f"classes.{self.model_name}")
        env = module.GymEnv(self.scenario_path,
                            n_actions=self.n_actions,
                            )

    def load_constants(self):
        module = importlib.import_module(f"levels.{self.level_name}")
        self.model_name = module.MODEL_NAME
        self.scenario_path = module.SCENARIO_PATH
        self.checkpoint_dir = module.CHECKPOINT_DIR
        self.log_dir = module.LOG_DIR
        self.learning_rate = module.LEARNING_RATE
        self.clip_range = module.CLIP_RANGE
        self.gamma = module.GAMMA
        self.gae_lambda = module.GAE_LAMBDA
        self.n_steps = module.N_STEPS
        self.n_timesteps = module.N_TIMESTEPS
