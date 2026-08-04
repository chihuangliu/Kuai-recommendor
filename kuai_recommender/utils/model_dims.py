from kuai_recommender.config import (
    MULTI_TASK_MODEL_EMBEDDING_DIM,
    MULTI_TASK_MODEL_HIDDEN_DIM,
)
from kuai_recommender.data.utils import get_bucket_size
from kuai_recommender.features.registry import (
    BINARY_TARGETS,
    BUCKET_COLS,
    CONTINUOUS_TARGETS,
    MODEL_CONTINUOUS,
)


def get_model_dims() -> tuple[int, int, list[tuple[int, int]], dict[str, int]]:
    input_dim = len(MODEL_CONTINUOUS)
    embedding_dims = [
        (get_bucket_size()[id_col.source], MULTI_TASK_MODEL_EMBEDDING_DIM)
        for id_col in BUCKET_COLS
    ]
    output_dims = {
        "binary": len(BINARY_TARGETS),
        "continuous": len(CONTINUOUS_TARGETS),
    }
    return input_dim, MULTI_TASK_MODEL_HIDDEN_DIM, embedding_dims, output_dims
