from dataclasses import dataclass

@dataclass
class Hyperparams:
    learning_rate: float = 3e-4
    batch_size: int = 10
    num_epochs: int = 2

