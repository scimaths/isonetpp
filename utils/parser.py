import torch
import argparse
from utils.tooling import ReadOnlyConfig
from subgraph_matching.model_handler import get_model_names, get_model

class Parser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        experiment_group = self.parser.add_argument_group("experiment_details")
        experiment_group.add_argument("--experiment_id", type=str, required=True, help="Label for the experiment, must be unique")
        experiment_group.add_argument("--model", type=str, required=True, choices=get_model_names(), help="Model name, should be implemented in models/")
        experiment_group.add_argument("--experiment_dir", type=str, required=True, help="Home directory to save the experiment artifacts")
        experiment_group.add_argument("--use_cuda", type=bool, default=True, help="Use CUDA for this experiment?")
        experiment_group.add_argument("--reproducible", type=bool, default=True, help="Should reproducibility be ensured?")
        experiment_group.add_argument("--seed", type=int, default=0, help="Seed for entire experiment, guarantees reproducibility provided the 'reproducible' flag is set")

        early_stopping_group = self.parser.add_argument_group("early_stopping")
        early_stopping_group.add_argument("--tolerance", type=float, default=0.0001, help="Minimum improvement required in metric to continue")
        early_stopping_group.add_argument("--patience", type=int, default=50, help="Maximum number of epochs with no improvements")

        dataset_group = self.parser.add_argument_group("dataset")
        dataset_group.add_argument("--dataset_name", type=str, required=True, choices=["aids", "mutag", "ptc_fm", "ptc_fr", "ptc_mm", "ptc_mr"], help="Name of dataset for experiment")
        dataset_group.add_argument("--dataset_size", type=str, default="small", choices=["small", "large"], help="Size of dataset for experiment - small v/s large")
        dataset_group.add_argument("--dataset_path", type=str, default=".", help="Relative path where datasets are stored")

        optimization_group = self.parser.add_argument_group("optimization")
        optimization_group.add_argument("--margin", type=float, default=0.5, help="Margin for hinging paired loss")
        optimization_group.add_argument("--max_epochs", type=int, default=1000, help="Maximum number of epochs to stop training at")
        optimization_group.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
        optimization_group.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for training")
        optimization_group.add_argument("--weight_decay", type=float, default=0.0005, help="Weight decay for training")

        model_group = self.parser.add_argument_group("model_group")
        model_group.add_argument("--model_config_path", type=str, required=True)

    def parse_args(self):
        self.args = self.parser.parse_args()
        return self.args

    def get_experiment_config(self):
        return ReadOnlyConfig(
            experiment_id = self.args.experiment_id,
            model = self.args.model,
            home_dir = self.args.experiment_dir,
            dataset = f"{self.args.dataset_name}_{self.args.dataset_size}",
            seed = self.args.seed,
        )

    def get_optimization_config(self):
        return ReadOnlyConfig(
            margin = self.args.margin,
            max_epochs = self.args.max_epochs,
            learning_rate = self.args.learning_rate,
            weight_decay = self.args.weight_decay,
        )

    def get_early_stopping_config(self):
        return ReadOnlyConfig(
            patience = self.args.patience,
            tolerance = self.args.tolerance,
        )

    def get_dataset_config(self):
        return ReadOnlyConfig(
            dataset_name = self.args.dataset_name,
            dataset_size = self.args.dataset_size,
            dataset_base_path = self.args.dataset_path,
            batch_size = self.args.batch_size,
        )