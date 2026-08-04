import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import NamedTuple, Protocol
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch
from feast import FeatureStore
from tqdm import tqdm

import kuai_recommender
from kuai_recommender.nn.multitask import MultiTaskModel
from kuai_recommender.serving.predictor import predict
from kuai_recommender.train.train_helper import (
    BinaryScores,
    ContinuousScores,
    prepare_binary_per_batch,
    prepare_continuous_per_batch,
)
from kuai_recommender.utils.model_dims import get_model_dims

from ..data.utils import KuaiPureDatasetSplits
from ..features.compute import build_base_frame, set_engagement_targets
from ..features.feature_repo.store import store
from ..features.registry import (
    BINARY_SIGNALS,
    BINARY_TARGETS,
    CONTINUOUS_TARGETS,
    MODEL_CONTINUOUS,
    UA,
    USER,
    VIDEO,
    Kind,
    get_cols,
)
from ..features.streaming import ServedFeatures, WindowFeatureAgg

USER_WINDOW_COLS = [c.name for c in get_cols(USER, Kind.ROLLING)]
USER_AUTHOR_WINDOW_COLS = [c.name for c in get_cols(UA, Kind.ROLLING)]
VIDEO_WINDOW_COLS = [c.name for c in get_cols(VIDEO, Kind.ROLLING)]
VIDEO_CUM_COLS = [c.name for c in get_cols(VIDEO, Kind.CUMULATIVE)]


def warmup() -> WindowFeatureAgg:
    base_df = build_base_frame().sort_values("dt")
    user_ids = base_df["user_id"].to_numpy(dtype=np.int64)
    author_ids = base_df["author_id"].to_numpy(dtype=np.int64)
    video_ids = base_df["video_id"].to_numpy(dtype=np.int64)
    is_clicks = base_df["is_click"].to_numpy(dtype=np.int64)
    dts = list(base_df["dt"])
    signals = base_df[BINARY_SIGNALS].to_numpy(dtype=np.float32)
    wf_agg = WindowFeatureAgg(pd.Timedelta("7D"))

    for i in range(len(base_df)):
        wf_agg.step(
            user_id=user_ids[i],
            author_id=author_ids[i],
            video_id=video_ids[i],
            ts=dts[i],
            is_click=is_clicks[i],
            signal=signals[i],
        )
    return wf_agg


@dataclass
class EvalInput:
    user_id: int
    author_id: int
    video_id: int
    signal: np.ndarray
    binary_targets: np.ndarray
    continuous_targets: np.ndarray

    def __post_init__(self):
        self.binary_mask = ~np.isnan(self.binary_targets)
        self.continuous_mask = ~np.isnan(self.continuous_targets)


@dataclass
class EvalConfig:
    id: str
    model: MultiTaskModel
    eval_batch_size: int
    eval_start_dt: datetime | None = None
    eval_end_dt: datetime | None = None

    def __post_init__(self):
        self.output_dir = (
            Path(kuai_recommender.__file__).resolve().parents[1] / "eval_res" / self.id
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def dump_metadata(self) -> None:
        metadata = {
            "id": self.id,
            "eval_batch_size": self.eval_batch_size,
            "eval_start_dt": self.eval_start_dt.isoformat()
            if self.eval_start_dt
            else None,
            "eval_end_dt": self.eval_end_dt.isoformat() if self.eval_end_dt else None,
        }
        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)


def _feat2dict(signal: np.ndarray, feature_names: list[str]) -> dict[str, float | int]:
    return {name: value for name, value in zip(feature_names, signal)}


class FeatureSource:
    from_online: bool = False

    def get_features(self, event: NamedTuple, served: ServedFeatures) -> np.ndarray:
        raise NotImplementedError


class ServedFeaturesSource(FeatureSource):
    from_online = False

    def get_features(self, event: NamedTuple, served: ServedFeatures) -> np.ndarray:
        named_served = {
            **dict(zip(USER_WINDOW_COLS, served.user)),
            **dict(zip(USER_AUTHOR_WINDOW_COLS, served.user_author)),
            **dict(zip(VIDEO_WINDOW_COLS, served.video)),
            **dict(zip(VIDEO_CUM_COLS, served.video_cum)),
        }

        return np.array(
            [named_served[f] for f in MODEL_CONTINUOUS],
            dtype=np.float32,
        )


class OnlineFeatureSource(FeatureSource):
    pass


class ReplaySink(Protocol):
    def on_event(self, event: NamedTuple, served: ServedFeatures) -> None: ...
    def on_hour_start(self, hour: pd.Timestamp) -> None: ...
    def on_hour_end(self, hour: pd.Timestamp) -> None: ...
    def on_finish(self) -> None: ...


class OnlineStoreSink(ReplaySink):
    def __init__(self, store: FeatureStore):
        self.store = store
        self.latest = {
            "user": {},
            "ua": {},
            "video": {},
        }

    def on_event(self, event: NamedTuple, served: ServedFeatures) -> None:
        self.latest["user"][event.user_id] = (event.dt, served.user)
        self.latest["video"][event.video_id] = (
            event.dt,
            served.video,
            served.video_cum,
        )
        self.latest["ua"][(event.user_id, event.author_id)] = (
            event.dt,
            served.user_author,
        )

    def on_hour_start(self, hour: pd.Timestamp) -> None:
        pass

    def on_hour_end(self, hour: pd.Timestamp) -> None:
        self.store.write_to_online_store(
            "user_features",
            pd.DataFrame(
                [
                    {
                        "user_id": user_id,
                        "dt": dt,
                        **_feat2dict(features, USER_WINDOW_COLS),
                    }
                    for user_id, (dt, features) in self.latest["user"].items()
                ]
            ),
        )
        self.store.write_to_online_store(
            "user_author_features",
            pd.DataFrame(
                [
                    {
                        "user_id": user_id,
                        "author_id": author_id,
                        "dt": dt,
                        **_feat2dict(features, USER_AUTHOR_WINDOW_COLS),
                    }
                    for (user_id, author_id), (dt, features) in self.latest[
                        "ua"
                    ].items()
                ]
            ),
        )
        self.store.write_to_online_store(
            "video_features",
            pd.DataFrame(
                [
                    {
                        "video_id": video_id,
                        "dt": dt,
                        **_feat2dict(features, VIDEO_WINDOW_COLS),
                        **_feat2dict(video_cum, VIDEO_CUM_COLS),
                    }
                    for video_id, (dt, features, video_cum) in self.latest[
                        "video"
                    ].items()
                ]
            ),
        )

    def on_finish(self) -> None:
        pass


class EvalSink(ReplaySink):
    def __init__(
        self,
        model: MultiTaskModel,
        feature_source: FeatureSource,
        eval_config: EvalConfig,
    ) -> None:
        self.model = model
        self.feature_source = feature_source
        self.eval_config = eval_config
        self.binary_scores = BinaryScores(BINARY_TARGETS)
        self.continuous_scores = ContinuousScores(CONTINUOUS_TARGETS)

    def on_event(self, event: NamedTuple, served: ServedFeatures):
        x = (
            self.feature_source.get_features(event, served)
            if not self.feature_source.from_online
            else None
        )
        self.hourly_eval_inputs.append(
            EvalInput(
                user_id=event.user_id,
                author_id=event.author_id,
                video_id=event.video_id,
                signal=x,
                binary_targets=np.array(
                    [getattr(event, f) for f in BINARY_TARGETS],
                    dtype=np.float32,
                ),
                continuous_targets=np.array(
                    [getattr(event, f) for f in CONTINUOUS_TARGETS],
                    dtype=np.float32,
                ),
            )
        )

    def on_hour_start(self, hour: pd.Timestamp) -> None:

        self.hourly_eval_inputs: list[EvalInput] = []

    def on_hour_end(self, hour: pd.Timestamp) -> None:
        for i in range(
            0, len(self.hourly_eval_inputs), self.eval_config.eval_batch_size
        ):
            batch_eval_inputs = self.hourly_eval_inputs[
                i : i + self.eval_config.eval_batch_size
            ]
            binary_masks = np.stack(
                [inp.binary_mask for inp in batch_eval_inputs], axis=0
            )
            continuous_masks = np.stack(
                [inp.continuous_mask for inp in batch_eval_inputs], axis=0
            )
            binary_targets = np.stack(
                [inp.binary_targets for inp in batch_eval_inputs], axis=0
            )
            continuous_targets = np.stack(
                [inp.continuous_targets for inp in batch_eval_inputs], axis=0
            )
            user_ids = [inp.user_id for inp in batch_eval_inputs]
            author_ids = [inp.author_id for inp in batch_eval_inputs]
            video_ids = [inp.video_id for inp in batch_eval_inputs]

            if not self.feature_source.from_online:
                x = np.stack([inp.signal for inp in batch_eval_inputs], axis=0)
            else:
                x = None

            pred = predict(
                self.eval_config.model,
                user_ids,
                author_ids,
                video_ids,
                features=x,
                from_online=self.feature_source.from_online,
            )

            binary_targets, binary_pred, binary_masks = prepare_binary_per_batch(
                pred["binary"],
                torch.from_numpy(binary_targets),
                torch.from_numpy(binary_masks),
            )
            continuous_targets, continuous_pred, continuous_masks = (
                prepare_continuous_per_batch(
                    pred["continuous"],
                    torch.from_numpy(continuous_targets),
                    torch.from_numpy(continuous_masks),
                )
            )
            self.binary_scores.append(binary_targets, binary_pred, binary_masks)
            self.continuous_scores.append(
                continuous_targets, continuous_pred, continuous_masks
            )

    def on_finish(self) -> None:
        binary_metrics = self.binary_scores.dump_metrics()
        continuous_metrics = self.continuous_scores.dump_metrics()
        metrics = {
            "binary": binary_metrics,
            "continuous": continuous_metrics,
        }
        with open(self.eval_config.output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
        self.eval_config.dump_metadata()


def replay(
    wf_agg: WindowFeatureAgg,
    df: pd.DataFrame,
    sinks: list[ReplaySink],
) -> None:
    for hour, batch_df in tqdm(df.groupby("hour_floor"), desc="Replaying"):
        for sink in sinks:
            sink.on_hour_start(hour)
        for event in batch_df.itertuples(index=False):
            served = wf_agg.step(
                user_id=event.user_id,
                author_id=event.author_id,
                video_id=event.video_id,
                ts=event.dt,
                is_click=event.is_click,
                signal=np.array(
                    [getattr(event, f) for f in BINARY_SIGNALS], dtype=np.float32
                ),
            )
            for sink in sinks:
                sink.on_event(event, served)
        for sink in sinks:
            sink.on_hour_end(hour)
    for sink in sinks:
        sink.on_finish()


def prepare_frame(
    split: KuaiPureDatasetSplits,
    start_dt: datetime | None = None,
    end_dt: datetime | None = None,
    with_targets: bool = True,
) -> pd.DataFrame:
    df = build_base_frame(split).sort_values("dt")
    if start_dt is not None:
        df = df[df["dt"] >= start_dt]
    if end_dt is not None:
        df = df[df["dt"] < end_dt]
    if with_targets:
        df = set_engagement_targets(df)
    if df.empty:
        raise ValueError("No data to process.")
    df["hour_floor"] = df["dt"].dt.floor("h")
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval",
        action="store_true",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=[split.name.lower() for split in KuaiPureDatasetSplits],
        required=True,
    )
    parser.add_argument(
        "--eval_start_date",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--eval_end_date",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--eval_model_id",
        type=str,
        required=True,
    )
    parser.add_argument("--eval_from_online", action="store_true")
    parser.add_argument(
        "--no_write_to_online",
        action="store_false",
        dest="write_to_online",
    )
    parser.set_defaults(write_to_online=True)
    args = parser.parse_args()
    split = KuaiPureDatasetSplits[args.split.upper()]
    df = build_base_frame(split).sort_values("dt")
    df["hour_floor"] = df["dt"].dt.floor("h")

    if args.eval:
        checkpoint = (
            Path(kuai_recommender.__file__).resolve().parents[1]
            / "run"
            / args.eval_model_id
            / "best.pt"
        )
        eval_config = EvalConfig(
            eval_batch_size=32,
            eval_start_dt=datetime.strptime(args.eval_start_date, "%Y-%m-%d").replace(
                tzinfo=ZoneInfo("Asia/Shanghai")
            )
            if args.eval_start_date
            else None,
            eval_end_dt=datetime.strptime(args.eval_end_date, "%Y-%m-%d").replace(
                tzinfo=ZoneInfo("Asia/Shanghai")
            )
            if args.eval_end_date
            else None,
            id=args.eval_model_id,
            model=MultiTaskModel.from_checkpoint(checkpoint, *get_model_dims()),
        )
    print("Warming up...")
    wf_agg = warmup()

    print("Replaying...")
    df = prepare_frame(
        split,
        eval_config.eval_start_dt if args.eval else None,
        eval_config.eval_end_dt if args.eval else None,
        with_targets=args.eval,
    )
    sinks: list[ReplaySink] = []
    if args.eval:
        feature_source = (
            OnlineFeatureSource() if args.eval_from_online else ServedFeaturesSource()
        )
        eval_sink = EvalSink(
            model=eval_config.model,
            feature_source=feature_source,
            eval_config=eval_config,
        )
        sinks.append(eval_sink)
    if args.write_to_online:
        online_sink = OnlineStoreSink(store)
        sinks.append(online_sink)

    replay(wf_agg, df, sinks)
