from torch import nn
import torch


class MultiTaskModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        embedding_dims: list[tuple[int, int]],
        output_dims: dict[str, int],
    ):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings, embedding_dim)
                for num_embeddings, embedding_dim in embedding_dims
            ]
        )
        self.embed_width = sum(d for _, d in embedding_dims)
        input_dim += self.embed_width
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

    def forward(self, x, x_cat):
        embedded_features = [
            embedding(x_cat[:, i]) for i, embedding in enumerate(self.embeddings)
        ]
        x = torch.cat([x, *embedded_features], dim=1)

        shared_output = self.shared_layers(x)
        outputs = {
            task: layer(shared_output) for task, layer in self.task_layers.items()
        }
        return outputs
