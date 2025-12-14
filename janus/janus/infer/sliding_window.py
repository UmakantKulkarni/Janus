"""Sliding window buffer for log lines."""

from __future__ import annotations

from collections import deque
import logging
from typing import Deque, Dict, List

logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)


class SlidingWindow:
    """Maintain a sliding window per key."""

    def __init__(self, max_lines: int = 128) -> None:
        self.max_lines = max_lines
        self.buffers: Dict[str, Deque[str]] = {}

    def add(self, key: str, line: str) -> List[str]:
        #logger.debug("Adding line to key %s", key)
        buf = self.buffers.setdefault(key, deque(maxlen=self.max_lines))
        buf.append(line)
        return list(buf)
