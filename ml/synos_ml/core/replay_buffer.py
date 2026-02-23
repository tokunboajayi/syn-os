"""
Experience Replay Buffer for Syn OS Synapse Core.

Stores (state, action, reward, next_state) tuples in a high-performance
circular buffer.  Data is periodically flushed to disk (JSON-Lines) for
offline training and historical analysis.

Design Notes
────────────
• Fixed capacity — oldest entries are silently evicted (ring buffer).
• Thread-safe via a simple lock (the OS collector runs on a background thread).
• `sample(n)` returns a random mini-batch for SGD, following the standard
  RL experience-replay pattern from DeepMind's DQN paper.
"""

import json
import logging
import os
import random
import time
import threading
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Experience:
    """Single experience tuple."""
    state: Dict[str, float] = field(default_factory=dict)
    action: str = ""                    # e.g. "throttle_cpu", "flush_cache"
    reward: float = 0.0                 # derived from Δhealth_score
    next_state: Dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ExperienceReplay:
    """
    Thread-safe circular replay buffer with disk persistence.

    Usage
    ─────
        buf = ExperienceReplay(capacity=10_000)
        buf.push(Experience(state={...}, action="noop", ...))
        batch = buf.sample(32)
    """

    def __init__(
        self,
        capacity: int = 50_000,
        persist_dir: str = "data/replay",
        flush_interval: int = 500,
    ):
        self.capacity = capacity
        self.persist_dir = persist_dir
        self.flush_interval = flush_interval

        self._buffer: deque[Experience] = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._unflushed = 0
        self._total_pushed = 0

        os.makedirs(persist_dir, exist_ok=True)

    # ── Core API ─────────────────────────────────────────────────────────

    def push(self, exp: Experience) -> None:
        """Add an experience to the buffer."""
        with self._lock:
            self._buffer.append(exp)
            self._total_pushed += 1
            self._unflushed += 1

        # Auto-flush to disk periodically
        if self._unflushed >= self.flush_interval:
            self.flush()

    def push_metrics(
        self,
        state: Dict[str, float],
        action: str = "observe",
        reward: float = 0.0,
        next_state: Optional[Dict[str, float]] = None,
    ) -> None:
        """Convenience method to push raw dicts."""
        self.push(Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state or state,
        ))

    def sample(self, batch_size: int = 32) -> List[Experience]:
        """Return a uniformly random mini-batch."""
        with self._lock:
            n = min(batch_size, len(self._buffer))
            return random.sample(list(self._buffer), n) if n > 0 else []

    def sample_states(self, batch_size: int = 32) -> List[Dict[str, float]]:
        """Return just the state dicts — handy for autoencoder training."""
        return [e.state for e in self.sample(batch_size)]

    # ── Persistence ──────────────────────────────────────────────────────

    def flush(self) -> None:
        """Write unflushed experiences to a JSONL file on disk."""
        with self._lock:
            if self._unflushed == 0:
                return
            entries = list(self._buffer)[-self._unflushed:]
            self._unflushed = 0

        ts = int(time.time())
        path = Path(self.persist_dir) / f"replay_{ts}.jsonl"
        try:
            with open(path, "w") as f:
                for exp in entries:
                    f.write(json.dumps(asdict(exp)) + "\n")
            logger.info(f"Flushed {len(entries)} experiences → {path}")
        except Exception as e:
            logger.error(f"Failed to flush replay buffer: {e}")

    def load_from_disk(self, max_files: int = 10) -> int:
        """Re-hydrate buffer from the most recent JSONL files."""
        files = sorted(Path(self.persist_dir).glob("replay_*.jsonl"), reverse=True)
        loaded = 0
        for f in files[:max_files]:
            try:
                with open(f) as fp:
                    for line in fp:
                        data = json.loads(line)
                        self.push(Experience(**data))
                        loaded += 1
            except Exception as e:
                logger.error(f"Error loading {f}: {e}")
        logger.info(f"Loaded {loaded} experiences from disk")
        return loaded

    # ── Stats ────────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return len(self._buffer)

    @property
    def total_pushed(self) -> int:
        return self._total_pushed

    def stats(self) -> Dict[str, Any]:
        return {
            "buffer_size": self.size,
            "capacity": self.capacity,
            "total_pushed": self._total_pushed,
            "unflushed": self._unflushed,
            "utilisation": f"{self.size / self.capacity * 100:.1f}%",
        }
