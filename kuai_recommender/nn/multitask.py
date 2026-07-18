from torch import nn


class MultiTaskModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dims: dict[str, int]):
        super().__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.task_layers = nn.ModuleDict(
            {
                task: nn.Linear(hidden_dim, output_dim)
                for task, output_dim in output_dims.items()
            }
        )

    def forward(self, x):
        shared_output = self.shared_layers(x)
        outputs = {
            task: layer(shared_output) for task, layer in self.task_layers.items()
        }
        return outputs
