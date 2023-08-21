import os
import shutil
from datetime import datetime
from pathlib import Path
import sys
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
sys.path.append(str(ROOT/ 'src')) 
from lora_tuning import Lora_Trainer
from utils import load_config


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
    trainer = Lora_Trainer(config, current_experiment_path)
    trainer.train()
if __name__ == "__main__":
    main()