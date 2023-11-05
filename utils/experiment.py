import os
import sys
import json
import torch
import logging
from datetime import datetime

class ReadOnlyConfig:
    def __init__(self, **kwargs):
        self.kv_store = kwargs

    def __getattr__(self, attr):
        return self.kv_store[attr]

    def as_json(self):
        return json.dumps(self.kv_store, indent=4)

def setup_logging(path):
    if not os.path.isdir(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    handler = logging.FileHandler(path)
    logger.addHandler(handler)
    return logger

LOG_DIR = "logs"
CONFIG_DIR = "configs"
INITIAL_MODELS_DIR = "initial_models"
TRAINED_MODELS_DIR = "trained_models"

class Experiment:
    def __init__(self, config: ReadOnlyConfig, home_dir: str = 'experiments/'):
        self.config = config
        self.experiment_id = config.experiment_id
        self.model = config.model
        self.dataset = config.dataset
        self.seed = config.seed
        self.home_dir = home_dir
        time_now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        
        # Setup directory structure (if required), logging
        logpath = self.get_unique_path(subdir=LOG_DIR, suffix='txt')
        if os.path.exists(logpath):
            print(f"Experiment {self.experiment_id} for {self.model}, {self.dataset}, seed: {self.seed} already exists, change experiment_id")
            exit()

        if not os.path.isdir(os.path.join(home_dir, self.experiment_id)):
            os.mkdir(os.path.join(home_dir, self.experiment_id))
            for subdir in [LOG_DIR, INITIAL_MODELS_DIR, TRAINED_MODELS_DIR, CONFIG_DIR]:
                os.mkdir(os.path.join(home_dir, self.experiment_id, subdir), exist_ok=True)
            
        self.logger = setup_logging(logpath)
        self.logger.info(f"Experiment {self.experiment_id} for model: '{self.model}', dataset: '{self.dataset}', seed: {self.seed} started at time: {time_now}")

        # Dump config
        config_path = self.get_unique_path(subdir=CONFIG_DIR, suffix='json')
        with open(config_path, 'w') as config_file:
            config_file.write(self.config.as_json())
    
    def get_unique_path(self, subdir, suffix):
        file_name = f"{self.model}_{self.dataset}_{self.seed}.{suffix}"
        return os.path.join(self.home_dir, self.experiment_id, subdir, file_name)

    def log(self, log_string):
        self.logger.info(log_string)
    
    def best_model_path(self):
        return self.get_unique_path(subdir=TRAINED_MODELS_DIR, suffix='pth')

    def save_best_model_state_dict(self, model, epoch):
        save_path = self.best_model_path()
        self.logger.info(f"saving best validated model to {save_path}")
        with open(save_path, 'wb') as model_handler:
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch
            }, model_handler)
        
    def load_best_model_state_dict(self):
        load_path = self.best_model_path()
        self.logger.info(f"loading best validated model from {load_path}")
        with open(load_path, 'rb') as model_handler:
            model_state_dict = torch.load(model_handler)['model_state_dict']
        return model_state_dict