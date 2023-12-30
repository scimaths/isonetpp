import json
import torch

class ReadOnlyConfig:
    def __init__(self, **kwargs):
        self.kv_store = kwargs

    def __getattr__(self, attr):
        return self.kv_store[attr]

    def as_json(self):
        return json.dumps(self.kv_store, indent=4)

def seed_everything(seed: int):
    import random, os
    import numpy as np
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False