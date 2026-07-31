from kuai_recommender.config import (
    MULTI_TASK_MODEL_EMBEDDING_DIM,
    MULTI_TASK_MODEL_HIDDEN_DIM,
)
from kuai_recommender.data.data_pure import KuaiPureData
from kuai_recommender.data.utils import get_bucket_size


def get_model_dims() -> tuple[int, int, list[tuple[int, int]], dict[str, int]]:
    input_dim = len(KuaiPureData.CONTINUOUS_FEATURES)
    embedding_dims = [
        (get_bucket_size()[id_col], MULTI_TASK_MODEL_EMBEDDING_DIM)
        for id_col in ["user_id", "author_id"]
    ]
    output_dims = {
        "binary": len(KuaiPureData.BINARY_TARGETS),
        "continuous": len(KuaiPureData.CONTINUOUS_TARGETS),
    }
    return input_dim, MULTI_TASK_MODEL_HIDDEN_DIM, embedding_dims, output_dims
