"""Training session manager: enforces mandatory CL1 rest periods."""

from __future__ import annotations

import asyncio
import logging
import time

import config

logger = logging.getLogger(__name__)


class SessionManager:
    """Tracks active training time and inserts rest periods as required.

    CL1 cells must rest for REST_SECONDS after every MAX_TRAIN_SECONDS of use.
    Call `await rest_if_needed()` at the top of each training step.
    """

    def __init__(self):
        self._segment_start: float = time.monotonic()
        self._total_trained_s: float = 0.0

    @property
    def segment_elapsed_s(self) -> float:
        return time.monotonic() - self._segment_start

    @property
    def total_trained_s(self) -> float:
        return self._total_trained_s + self.segment_elapsed_s

    def needs_rest(self) -> bool:
        return self.segment_elapsed_s >= config.MAX_TRAIN_SECONDS

    async def rest_if_needed(self) -> None:
        if not self.needs_rest():
            return
        elapsed = self.segment_elapsed_s
        self._total_trained_s += elapsed
        logger.info(
            "CL1 rest period: trained %.1f min this segment (%.1f min total). "
            "Resting for %.0f min.",
            elapsed / 60,
            self._total_trained_s / 60,
            config.REST_SECONDS / 60,
        )
        await asyncio.sleep(config.REST_SECONDS)
        self._segment_start = time.monotonic()
        logger.info("Rest complete — resuming training.")
