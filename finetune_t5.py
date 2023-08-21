import os
import shutil
from datetime import datetime


from src.trainer import Trainer
from src.utils import load_config


EXPERIMENT_PATH = "experiments"
EXPERIMENT_CONFIG_NAME = "t5_config.yml"


def main():
    # load config
    config = load_config(EXPERIMENT_CONFIG_NAME)

    # specify and create experiment path
    timestamp = datetime.now().strftime("_%Y%m%d-%H%M%S")
    current_experiment_path = os.path.join(EXPERIMENT_PATH, EXPERIMENT_PATH + timestamp)
    os.makedirs(current_experiment_path)
    # initialize trainer and train
    trainer = Trainer(config, current_experiment_path)
    trainer.train()


if __name__ == "__main__":
    main()
