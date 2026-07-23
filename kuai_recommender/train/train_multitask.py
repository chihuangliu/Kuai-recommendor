from kuai_recommender.data.data_pure import (
    KuaiPureData,
    KuaiPureDataset,
    collate_with_masks,
)
from kuai_recommender.data.utils import KuaiPureDatasetSplits, get_bucket_size
from kuai_recommender.nn.multitask import MultiTaskModel
from kuai_recommender.config import (
    LEARNING_RATE,
    MULTI_TASK_MODEL_HIDDEN_DIM,
    MULTI_TASK_MODEL_EMBEDDING_DIM,
    NEG_KEEP_FRAC,
    EPOCH,
    BATCH_SIZE,
    POS_WEIGHT,
)
from kuai_recommender.train.train_helper import inference_and_calc_loss
from kuai_recommender.utils.device import get_device
from torch import optim
from torch.utils.data import DataLoader
import torch
from kuai_recommender.train.train_helper import get_run_dir, get_run_id


def main():
    # setup data
    kauai_pure_data_train = KuaiPureDatasetSplits.TRAIN
    kauai_pure_data_val = KuaiPureDatasetSplits.VAL

    train_data = KuaiPureData(
        kauai_pure_data_train,
    )
    val_data = KuaiPureData(kauai_pure_data_val, history=(kauai_pure_data_train,))

    train_dataset = KuaiPureDataset(
        train_data,
        neg_keep_frac=NEG_KEEP_FRAC,
    )

    val_dataset = KuaiPureDataset(
        val_data,
        neg_keep_frac=1.0,
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
    input_dim = len(KuaiPureData.FEATURE_COLUMNS)
    embedding_dims = [
        (get_bucket_size()[id_col], MULTI_TASK_MODEL_EMBEDDING_DIM)
        for id_col in ["user_id", "author_id"]
    ]
    output_dims = {
        "binary": len(KuaiPureData.BINARY_COLUMNS_PREPROCESSED),
        "continuous": len(KuaiPureData.CONTINUOUS_COLUMNS_PREPROCESSED),
    }
    device = get_device()
    model = MultiTaskModel(
        input_dim, MULTI_TASK_MODEL_HIDDEN_DIM, embedding_dims, output_dims
    ).to(device)

    # train
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    schedular = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH)
    pos_weights = torch.full(
        (len(KuaiPureData.BINARY_COLUMNS_PREPROCESSED),), POS_WEIGHT, device=device
    )

    # setup run id
    run_id = get_run_id()

    best_val_loss = float("inf")
    for epoch in range(EPOCH):
        total_train_loss = 0.0
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

            loss = inference_and_calc_loss(
                model,
                device,
                x_batch,
                x_cat_batch,
                y_binary_batch,
                y_continuous_batch,
                mask_binary_batch,
                mask_continuous_batch,
                pos_weights,
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
                loss = inference_and_calc_loss(
                    model,
                    device,
                    x_batch,
                    x_cat_batch,
                    y_binary_batch,
                    y_continuous_batch,
                    mask_binary_batch,
                    mask_continuous_batch,
                    pos_weights,
                )
                total_val_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            avg_val_loss = total_val_loss / len(val_loader)
            print(
                f"Epoch {epoch:3d}/{EPOCH}  train={avg_train_loss:.4f}  val={avg_val_loss:.4f}"
            )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                run_dir = get_run_dir(run_id)
                run_dir.mkdir(exist_ok=True)
                torch.save(model.state_dict(), run_dir / "best.pt")


if __name__ == "__main__":
    main()
