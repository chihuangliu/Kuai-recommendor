from dataclasses import dataclass
from enum import Enum

BINARY_SIGNALS = [
    "is_click",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
    "long_view",
    "is_profile_enter",
]


class Kind(Enum):
    ROLLING = "rolling"
    CUMULATIVE = "cumulative"
    STATIC = "static"
    ARRAY = "array"


class DType(Enum):
    FLOAT = "float"
    INT = "int"


@dataclass(frozen=True)
class View:
    name: str
    join_keys: tuple[str, ...]


@dataclass(frozen=True)
class Col:
    name: str
    view: View
    kind: Kind
    dtype: DType
    signal: str | None = None


@dataclass(frozen=True)
class BucketCol:
    source: str

    @property
    def name(self) -> str:
        return f"{self.source}_bucket"


def _rolling(view, suffix: str, signals=BINARY_SIGNALS) -> list[Col]:
    return [
        Col(f"{s}_rolling_{suffix}", view, Kind.ROLLING, DType.FLOAT, s)
        for s in signals
    ]


# --- FEATURE STORE ---
USER = View("user_features", ("user_id",))
UA = View("user_author_features", ("user_id", "author_id"))
VIDEO = View("video_features", ("video_id",))

STORE_COLS = (
    _rolling(USER, "user_id")
    + _rolling(UA, "user_id_author_id")
    + _rolling(VIDEO, "video_id")
    + [
        Col(
            "is_click_cumulative_video_id",
            VIDEO,
            Kind.CUMULATIVE,
            DType.INT,
            "is_click",
        ),
    ]
)

# --- MODEL ---
MODEL_CONTINUOUS = [
    "is_click_rolling_user_id",
    "long_view_rolling_user_id",
    "is_like_rolling_user_id",
    "is_profile_enter_rolling_user_id",
    "is_click_rolling_user_id_author_id",
    "long_view_rolling_user_id_author_id",
    "is_like_rolling_user_id_author_id",
    "is_click_rolling_video_id",
    "long_view_rolling_video_id",
    "is_like_rolling_video_id",
    "is_click_cumulative_video_id",
]
BUCKET_COLS = [BucketCol("user_id"), BucketCol("author_id")]
MODEL_CATEGORICAL = [c.name for c in BUCKET_COLS]
BINARY_TARGETS = BINARY_SIGNALS + ["is_skip"]
CONTINUOUS_TARGETS = ["dwell_log"]

ENGAGEMENT_INPUT_COLUMNS = ["duration_ms", "play_time_ms"]


def get_cols(view: View, kind: Kind | None = None) -> list[Col]:
    return [
        c for c in STORE_COLS if c.view == view and (kind is None or c.kind == kind)
    ]


def get_rolling_signals(view: View) -> list[str]:
    return [c.signal for c in STORE_COLS if c.view == view and c.kind == Kind.ROLLING]


def get_online_refs():
    by_name = {c.name: c for c in STORE_COLS}
    return [f"{by_name[c].view.name}:{c}" for c in MODEL_CONTINUOUS]


_stored = {c.name: c for c in STORE_COLS}
assert set(MODEL_CONTINUOUS).issubset(_stored.keys()), (
    "Some model continuous features are not stored in the feature store"
)
