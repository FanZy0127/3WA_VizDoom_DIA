# last_model_path = str(os.listdir(models_path)[1])  # 1 => best_model_100000.zip for train_defend
# last_model_path = str(os.listdir(models_path)[34])  # 34 => best_model_400000.zip for curriculum_1 & 4
# last_model_path = str(os.listdir(models_path)[30])  # 30 => best_model_370000.zip for curriculum_2 & 3

import typer
from rich import print
from rich.table import Table

from utils.consts.cli import *
from utils.consts.consts import *
from utils.train import Train
from utils.run import Run


def request_level():
    # Show all levels and ask for the one to train
    levels = '\n'.join([f'{i + 1}. {option}' for i, option in enumerate(GAME_LEVELS)])
    print(TRAIN_SELECTION.format(levels))
    choice = typer.prompt(SELECT_AN_OPTION, default=1)
    print(choice)

    # Check if the choice is valid
    if choice > len(GAME_LEVELS):
        raise ValueError(INVALID_CHOICE)
    else:
        return choice


def main():
    print(WELCOME)
    choice = typer.prompt(SELECT_AN_OPTION, default=1)
    print(choice)

    if choice == PLAY:
        level_name = GAME_LEVELS[request_level() - 1]
        Run(level_name).start()
    elif choice == TRAIN:
        level_name = GAME_LEVELS[request_level() - 1]
        Train(level_name).start()

    else:
        raise ValueError(INVALID_CHOICE)


if __name__ == "__main__":
    typer.run(main)