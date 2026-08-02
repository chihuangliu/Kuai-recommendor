from collections import deque
from collections.abc import Hashable, Iterator
from typing import NamedTuple, Protocol, TypeVar

import numpy as np
import pandas as pd

S = TypeVar("S")


class KeyedStateStore(Protocol[S]):
    def get(self, key: Hashable) -> None | S: ...
    def put(self, key: Hashable, value: S) -> None: ...
    def items(self) -> Iterator[tuple[Hashable, S]]: ...


class DictStateStore(KeyedStateStore[S]):
    def __init__(self) -> None:
        self._store: dict[Hashable, S] = {}

    def get(self, key: Hashable) -> None | S:
        return self._store.get(key)

    def put(self, key: Hashable, value: S) -> None:
        self._store[key] = value

    def items(self) -> Iterator[tuple[Hashable, S]]:
        return iter(self._store.items())


class Event(NamedTuple):
    ts: pd.Timestamp
    signal: np.ndarray


class WindowState(NamedTuple):
    events: deque[Event]
    sum_vec: np.ndarray


class WindowAvg:
    def __init__(self, store: KeyedStateStore[WindowState], window: pd.Timedelta):
        self.store = store
        self.window = window

    def step(self, key: Hashable, ts: pd.Timestamp, signal: np.ndarray) -> np.ndarray:
        window_state: WindowState = self.store.get(key)
        if window_state is None:
            window_state = WindowState(deque(), np.zeros_like(signal, dtype=float))
        events = window_state.events.copy()
        sum_vec = window_state.sum_vec
        while events and events[-1].ts < ts - self.window:
            _, signal_old = events.pop()
            sum_vec -= signal_old
        res = (
            sum_vec / len(events)
            if events
            else np.full_like(signal, fill_value=np.nan, dtype=float)
        )

        events.appendleft(Event(ts, signal))
        sum_vec += signal
        self.store.put(key, WindowState(events, sum_vec))
        return res


class RunningCount(NamedTuple):
    count: int


class RunningCountAgg:
    def __init__(self, store: KeyedStateStore[RunningCount]):
        self.store = store

    def step(self, key: Hashable, x: int) -> int:
        running_count = self.store.get(key)
        if running_count is None:
            running_count = RunningCount(0)

        res = running_count.count
        self.store.put(key, RunningCount(running_count.count + x))
        return res


class WindowFeatureAgg:
    def __init__(self, window: pd.Timedelta):
        self.user_avg = WindowAvg(DictStateStore(), window)
        self.ua_avg = WindowAvg(DictStateStore(), window)
        self.video_avg = WindowAvg(DictStateStore(), window)
        self.video_cum = RunningCountAgg(DictStateStore())

    def step(
        self,
        *,
        ts: pd.Timestamp,
        user_id: int,
        author_id: float,
        video_id: int,
        signal: np.ndarray,
        is_click: int,
    ) -> dict[str, np.ndarray]:
        served = {
            "user": self.user_avg.step(user_id, ts, signal),
            "video": self.video_avg.step(video_id, ts, signal),
            "video_cum": np.array(
                [self.video_cum.step(video_id, is_click)], dtype="int32"
            ),
        }
        if not pd.isna(author_id):
            served["user_author"] = self.ua_avg.step(
                (user_id, int(author_id)), ts, signal
            )
        return served
