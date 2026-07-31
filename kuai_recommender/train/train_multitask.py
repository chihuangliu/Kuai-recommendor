import json
from datetime import datetime
from zoneinfo import ZoneInfo

import torch
from torch import optim
from torch.utils.data import DataLoader

from kuai_recommender.config import (
    BATCH_SIZE,
    EPOCH,
    LEARNING_RATE,
    MULTI_TASK_MODEL_HIDDEN_DIM,
    NEG_KEEP_FRAC,
    POS_WEIGHT,
)
from kuai_recommender.data.data_pure import (
    KuaiPureData,
    KuaiPureDataset,
    collate_with_masks,
)
from kuai_recommender.data.utils import KuaiPureDatasetSplits
from kuai_recommender.features.common import get_training_file_path
from kuai_recommender.nn.multitask import MultiTaskModel
from kuai_recommender.train.train_helper import (
    BinaryScores,
    ContinuousScores,
    calc_loss,
    get_run_dir,
    get_run_id,
    inference_batch,
    prepare_binary_per_batch,
    prepare_continuous_per_batch,
    tensors_to_device,
)
from kuai_recommender.utils.device import get_device
from kuai_recommender.utils.model_dims import get_model_dims


def main():
    cutoff_time = datetime.strptime("2022-04-20", "%Y-%m-%d").replace(
        tzinfo=ZoneInfo("Asia/Shanghai")
    )
    # setup data
    train_dataset = KuaiPureDataset.from_parquet(
        get_training_file_path(KuaiPureDatasetSplits.TRAIN),
        neg_keep_frac=NEG_KEEP_FRAC,
        end_dt=cutoff_time,
    )
    val_dataset = KuaiPureDataset.from_parquet(
        get_training_file_path(KuaiPureDatasetSplits.TRAIN),
        start_dt=cutoff_time,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_with_masks,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=4,
        collate_fn=collate_with_masks,
    )

    # setup model
    input_dim, hidden_dim, embedding_dims, output_dims = get_model_dims()
    device = get_device()
    model = MultiTaskModel(input_dim, hidden_dim, embedding_dims, output_dims).to(
        device
    )

    # train
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    schedular = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH)
    pos_weights = torch.full(
        (len(KuaiPureData.BINARY_TARGETS),), POS_WEIGHT, device=device
    )

    # setup run id
    run_id = get_run_id()

    best_val_loss = float("inf")
    for epoch in range(EPOCH):
        total_train_loss = 0.0
        binary_scores = BinaryScores(KuaiPureData.BINARY_TARGETS)
        continuous_scores = ContinuousScores(KuaiPureData.CONTINUOUS_TARGETS)
        model.train()
        for (
            x_batch,
            x_cat_batch,
            y_binary_batch,
            y_continuous_batch,
            mask_binary_batch,
            mask_continuous_batch,
        ) in train_loader:
            optimizer.zero_grad()

            (
                x_batch,
                x_cat_batch,
                y_binary_batch,
                y_continuous_batch,
                mask_binary_batch,
                mask_continuous_batch,
            ) = tensors_to_device(
                x_batch,
                x_cat_batch,
                y_binary_batch,
                y_continuous_batch,
                mask_binary_batch,
                mask_continuous_batch,
                device=device,
            )

            outputs = inference_batch(
                model,
                x_batch,
                x_cat_batch,
            )
            loss = calc_loss(
                outputs,
                pos_weights,
                y_binary_batch,
                y_continuous_batch,
                mask_binary_batch,
                mask_continuous_batch,
            )
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()

        schedular.step()

        model.eval()
        with torch.no_grad():
            total_val_loss = 0.0
            for (
                x_batch,
                x_cat_batch,
                y_binary_batch,
                y_continuous_batch,
                mask_binary_batch,
                mask_continuous_batch,
            ) in val_loader:
                (
                    x_batch,
                    x_cat_batch,
                    y_binary_batch,
                    y_continuous_batch,
                    mask_binary_batch,
                    mask_continuous_batch,
                ) = tensors_to_device(
                    x_batch,
                    x_cat_batch,
                    y_binary_batch,
                    y_continuous_batch,
                    mask_binary_batch,
                    mask_continuous_batch,
                    device=device,
                )

                outputs = inference_batch(
                    model,
                    x_batch,
                    x_cat_batch,
                )
                loss = calc_loss(
                    outputs,
                    pos_weights,
                    y_binary_batch,
                    y_continuous_batch,
                    mask_binary_batch,
                    mask_continuous_batch,
                )
                total_val_loss += loss.item()

                y_true, y_score, mask = prepare_binary_per_batch(
                    outputs["binary"], y_binary_batch, mask_binary_batch
                )
                binary_scores.append(y_true, y_score, mask)

                y_cont, y_pred, mask_cont = prepare_continuous_per_batch(
                    outputs["continuous"], y_continuous_batch, mask_continuous_batch
                )
                continuous_scores.append(y_cont, y_pred, mask_cont)

            avg_train_loss = total_train_loss / len(train_loader)
            avg_val_loss = total_val_loss / len(val_loader)
            print(
                f"Epoch {epoch:3d}/{EPOCH}  train={avg_train_loss:.4f}  val={avg_val_loss:.4f}"
            )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                run_dir = get_run_dir(run_id)
                run_dir.mkdir(exist_ok=True)
                states = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "schedular_state_dict": schedular.state_dict(),
                }
                torch.save(states, run_dir / "best.pt")

                binary_metrics = binary_scores.dump_metrics()
                continuous_metrics = continuous_scores.dump_metrics()
                metrics = {
                    "binary": binary_metrics,
                    "continuous": continuous_metrics,
                }

                with open(run_dir / "metrics.json", "w") as f:
                    json.dump(metrics, f)


if __name__ == "__main__":
    main()
