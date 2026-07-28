BINARY_FEATURES = [
    "is_click",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
    "long_view",
    "is_profile_enter",
]

USER_FEATURES = [f"{f}_rolling_user_id" for f in BINARY_FEATURES]
USER_AUTHOR_FEATURES = [f"{f}_rolling_user_id_author_id" for f in BINARY_FEATURES]

VIDEO_FEATURES_FLOAT = [f"{f}_rolling_video_id" for f in BINARY_FEATURES]
VIDEO_FEATURES_INT = ["is_click_cumulative_video_id"]
VIDEO_FEATURES = VIDEO_FEATURES_FLOAT + VIDEO_FEATURES_INT
ENGAGEMENT_INPUT_COLUMNS = ["duration_ms", "play_time_ms"]
