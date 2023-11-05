import os
import sys
import json
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

class Experiment:
    def __init__(self, config: ReadOnlyConfig, home_dir: str = 'experiments/'):
        self.config = config
        self.experiment_id = config.experiment_id
        self.model = config.model
        self.dataset = config.dataset
        self.seed = config.seed
        self.home_dir = home_dir
        self.time_id = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        
        # Setup directory structure (if required), logging
        if not os.path.isdir(os.path.join(home_dir, self.experiment_id)):
            os.mkdir(os.path.join(home_dir, self.experiment_id))
            for subfolder in ["logs", "initial_models", "trained_models", "configs"]:
                os.mkdir(os.path.join(home_dir, self.experiment_id, subfolder), exist_ok=True)
        unique_path = self.get_unique_path()
        self.logger = setup_logging(f"{unique_path}.txt")

        # Dump config
        with open(os.path.join(home_dir, self.experiment_id, "configs", f"{unique_path}.json"), 'w') as config_file:
            config_file.write(self.config.as_json())
    
    def get_unique_path(self):
        file_name = f"{self.model}_{self.dataset}_{self.seed}_{self.time_id}"
        return os.path.join(self.home_dir, self.experiment_id, "logs", file_name)

    def log(self, log_string):
        self.logger.info(log_string)