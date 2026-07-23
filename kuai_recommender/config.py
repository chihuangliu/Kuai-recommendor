# data
FEATURES = [
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

# model
MULTI_TASK_MODEL_HIDDEN_DIM = 128
MULTI_TASK_MODEL_EMBEDDING_DIM = 128

# training
NEG_KEEP_FRAC = 0.5
LEARNING_RATE = 1e-4
EPOCH = 1
BATCH_SIZE = 32
POS_WEIGHT = 2
