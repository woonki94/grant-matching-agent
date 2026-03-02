from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Optional, Sequence, TypeVar, cast

TItem = TypeVar("TItem")
TOut = TypeVar("TOut")
TState = TypeVar("TState")


def build_thread_local_getter(factory: Callable[[], TState]) -> Callable[[], TState]:
    local = threading.local()

    def _get() -> TState:
        state = getattr(local, "state", None)
        if state is None:
            state = factory()
            local.state = state
        return cast(TState, state)

    return _get


def resolve_pool_size(*, max_workers: int, task_count: int) -> int:
    if task_count <= 0:
        return 0
    return min(task_count, max(1, int(max_workers)))


def parallel_map(
    items: Sequence[TItem],
    *,
    max_workers: int,
    run_item: Callable[[TItem], TOut],
    on_error: Optional[Callable[[int, TItem, Exception], TOut]] = None,
) -> List[TOut]:
    count = len(items)
    if count == 0:
        return []

    pool_size = resolve_pool_size(max_workers=max_workers, task_count=count)
    results: List[Optional[TOut]] = [None] * count

    def _safe_run(index: int, item: TItem) -> TOut:
        try:
            return run_item(item)
        except Exception as e:
            if on_error is None:
                raise
            return on_error(index, item, e)

    if pool_size <= 1:
        for index, item in enumerate(items):
            results[index] = _safe_run(index, item)
        return cast(List[TOut], results)

    with ThreadPoolExecutor(max_workers=pool_size) as ex:
        futures = {
            ex.submit(_safe_run, index, item): index
            for index, item in enumerate(items)
        }
        for fut in as_completed(futures):
            index = futures[fut]
            results[index] = fut.result()

    return cast(List[TOut], results)
