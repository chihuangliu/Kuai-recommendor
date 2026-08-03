import numpy as np
import torch

from kuai_recommender.data.data_pure import KuaiPureData
from kuai_recommender.data.utils import get_bucket_size
from kuai_recommender.features.compute import hash_to_bucket
from kuai_recommender.features.feature_repo.store import store
from kuai_recommender.features.schema import (
    USER_AUTHOR_FEATURES,
    USER_FEATURES,
    VIDEO_FEATURES,
)
from kuai_recommender.nn.multitask import MultiTaskModel
from kuai_recommender.utils.model_dims import get_model_dims


def predict(
    model: MultiTaskModel,
    user_ids: list[str],
    author_ids: list[str],
    video_ids: list[str],
    features: np.ndarray | None = None,
    from_online: bool = True,
) -> dict[str, torch.Tensor]:
    user_id_buckets = torch.tensor(
        [hash_to_bucket(user_id, get_bucket_size()["user_id"]) for user_id in user_ids],
        dtype=torch.long,
    )
    author_id_buckets = torch.tensor(
        [
            hash_to_bucket(author_id, get_bucket_size()["author_id"])
            for author_id in author_ids
        ],
        dtype=torch.long,
    )

    x_cat = torch.stack([user_id_buckets, author_id_buckets], dim=1)
    if from_online:
        all_features = store.get_online_features(
            features=[f"user_features:{f}" for f in USER_FEATURES]
            + [f"user_author_features:{f}" for f in USER_AUTHOR_FEATURES]
            + [f"video_features:{f}" for f in VIDEO_FEATURES],
            entity_rows=[
                {"user_id": int(u), "author_id": int(a), "video_id": int(v)}
                for u, a, v in zip(user_ids, author_ids, video_ids)
            ],
        ).to_dict()

        all_feature_values = []
        for feat in KuaiPureData.CONTINUOUS_FEATURES:
            values: list = [
                v if v is not None else float("nan") for v in all_features[feat]
            ]
            all_feature_values.append(torch.tensor(values, dtype=torch.float32))
        x = torch.stack(all_feature_values, dim=1).nan_to_num(nan=0.0)
    else:
        assert features is not None, "Features must be provided if from_online is False"
        x = torch.tensor(features, dtype=torch.float32).nan_to_num(nan=0.0)

    model.eval()
    with torch.no_grad():
        return model.forward(x, x_cat)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
    )
    parser.add_argument(
        "--user_ids",
        type=str,
        nargs="+",
    )
    parser.add_argument(
        "--author_ids",
        type=str,
        nargs="+",
    )
    parser.add_argument(
        "--video_ids",
        type=str,
        nargs="+",
    )
    args = parser.parse_args()
    user_ids = args.user_ids
    author_ids = args.author_ids
    video_ids = args.video_ids
    model = MultiTaskModel.from_checkpoint(args.checkpoint, *get_model_dims())
    pred = predict(model, user_ids, author_ids, video_ids)
    print(pred)
