from dataclasses import dataclass

@dataclass
class Hyperparams:
    learning_rate: float = 3e-4
    batch_size: int = 10
    num_epochs: int = 2

@dataclass
class HyperparamsK8:
    learning_rate: float = 3e-4
    batch_size: int = 10
    num_epochs: int = 1


@dataclass
class AttentionLoraTrainingParams:
    selector_learning_rate: float = 3e-4
    lora_clones_learning_rate: float = 3e-4
    batch_size: int = 1
    num_epochs: int = 2
    contrastive_lambda: float = 0.01
