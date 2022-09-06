import importlib
from DoomAi.utils.consts.consts import *
from stable_baselines3 import PPO
# Import Evaluation Policy to test our agent
from stable_baselines3.common.evaluation import evaluate_policy


class Run():
    def __init__(self, level_name):
        self.level_name = level_name
        self.load_constants()

    def start(self):
        env = self.get_env()
        model = PPO.load(self.model_path)
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=N_EVAL_EPISODES)

        return mean_reward

    def get_env(self):
        module = importlib.import_module(f"classes.{self.model_name}")
        env = module.GymEnv(self.scenario_path, n_actions=self.n_actions, render=True, hd=True)

        return env

    def load_constants(self):
        module = importlib.import_module(f"utils.consts.{self.level_name}")
        self.model_name = module.MODEL_NAME
        self.scenario_path = module.SCENARIO_PATH
        self.model_path = module.MODEL_PATH
        self.n_actions = module.N_ACTIONS
