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


def request_stage():
    # Show all stages and ask for the one to train
    stages = '\n'.join([f'{i + 1}. {option}' for i, option in enumerate(GAME_STAGES)])
    print(TRAIN_SELECTION.format(stages))
    choice = typer.prompt(SELECT_AN_OPTION, default=1)

    print(choice)

    # Check if the choice is valid
    if choice > len(GAME_STAGES):
        raise ValueError(INVALID_CHOICE)
    else:
        return choice


def main():
    print(WELCOME)
    choice = typer.prompt(SELECT_AN_OPTION, default=1)
    print(choice)

    if choice == PLAY:
        stage_name = GAME_STAGES[request_stage() - 1]
        Run(stage_name).start()
    elif choice == TRAIN:
        stage_name = GAME_STAGES[request_stage() - 1]
        Train(stage_name).start()
    else:
        raise ValueError(INVALID_CHOICE)


if __name__ == "__main__":
    typer.run(main)
