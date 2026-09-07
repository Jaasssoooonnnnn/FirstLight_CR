"""Sparse, portable training-match archives for later native-render replay."""

from __future__ import annotations

from dataclasses import dataclass, field
import gzip
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, TYPE_CHECKING
import uuid

from ..contracts import EpisodeConfigV1
from ..snapshot import SnapshotOperationV1

if TYPE_CHECKING:
    from ..battle_env import BattleEnvV1


TRAINING_REPLAY_VERSION = "speed-xbow-training-replay.v1"


class TrainingReplayError(RuntimeError):
    """Raised when a retained training replay is invalid or corrupted."""


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class TrainingReplayV1:
    """One completed self-play match plus the exact applied action trace."""

    completed_match_index: int
    ruleset_id: str
    episode_config: EpisodeConfigV1
    episode_id: str
    operations: tuple[SnapshotOperationV1, ...]
    terminal: Mapping[str, Any]
    source: Mapping[str, Any] = field(default_factory=dict)
    created_at_ns: int = field(default_factory=time.time_ns)
    version: str = field(default=TRAINING_REPLAY_VERSION, init=False)

    def __post_init__(self) -> None:
        if self.completed_match_index <= 0:
            raise TrainingReplayError("completed_match_index must be positive")
        if not self.ruleset_id:
            raise TrainingReplayError("ruleset_id must not be empty")
        if self.episode_config.ruleset_id != self.ruleset_id:
            raise TrainingReplayError("episode and replay ruleset IDs do not match")
        if not self.episode_id:
            raise TrainingReplayError("episode_id must not be empty")
        previous_end: int | None = None
        for index, operation in enumerate(self.operations):
            if operation.end_native_tick < operation.start_native_tick:
                raise TrainingReplayError(f"operation {index} has a negative tick interval")
            if previous_end is not None and operation.start_native_tick != previous_end:
                raise TrainingReplayError(f"operation {index} is not contiguous")
            previous_end = operation.end_native_tick
        object.__setattr__(self, "terminal", _plain(self.terminal))
        object.__setattr__(self, "source", _plain(self.source))

    @property
    def start_native_tick(self) -> int:
        return self.operations[0].start_native_tick if self.operations else 0

    @property
    def end_native_tick(self) -> int:
        return self.operations[-1].end_native_tick if self.operations else 0

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "version": self.version,
            "completed_match_index": self.completed_match_index,
            "created_at_ns": self.created_at_ns,
            "ruleset_id": self.ruleset_id,
            "episode_config": self.episode_config.to_dict(),
            "episode_id": self.episode_id,
            "start_native_tick": self.start_native_tick,
            "end_native_tick": self.end_native_tick,
            "operations": [operation.to_dict() for operation in self.operations],
            "terminal": _plain(self.terminal),
            "source": _plain(self.source),
        }
        payload["content_sha256"] = hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()
        return payload

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TrainingReplayV1":
        data = dict(value)
        expected = data.pop("content_sha256", None)
        if data.get("version") != TRAINING_REPLAY_VERSION:
            raise TrainingReplayError(f"unsupported training replay version {data.get('version')!r}")
        if not isinstance(expected, str) or not expected:
            raise TrainingReplayError("training replay has no content checksum")
        actual = hashlib.sha256(_canonical(data).encode("utf-8")).hexdigest()
        if actual != expected:
            raise TrainingReplayError("training replay content checksum mismatch")
        replay = cls(
            completed_match_index=int(data["completed_match_index"]),
            created_at_ns=int(data["created_at_ns"]),
            ruleset_id=str(data["ruleset_id"]),
            episode_config=EpisodeConfigV1.from_mapping(data["episode_config"]),
            episode_id=str(data["episode_id"]),
            operations=tuple(SnapshotOperationV1.from_mapping(item) for item in data.get("operations", ())),
            terminal=dict(data.get("terminal", {})),
            source=dict(data.get("source", {})),
        )
        if int(data.get("start_native_tick", 0)) != (replay.start_native_tick):
            raise TrainingReplayError("training replay start tick does not match its trace")
        if int(data.get("end_native_tick", 0)) != (replay.end_native_tick):
            raise TrainingReplayError("training replay end tick does not match its trace")
        return replay

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "TrainingReplayV1":
        source = Path(path)
        payload = source.read_bytes()
        if payload.startswith(b"\x1f\x8b"):
            try:
                payload = gzip.decompress(payload)
            except (EOFError, gzip.BadGzipFile) as error:
                raise TrainingReplayError(f"invalid gzip training replay: {source}") from error
        try:
            value = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise TrainingReplayError(f"invalid training replay JSON: {source}") from error
        if not isinstance(value, Mapping):
            raise TrainingReplayError("training replay root must be an object")
        return cls.from_mapping(value)

    def save(self, path: str | os.PathLike[str], *, overwrite: bool = False, compresslevel: int = 6) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and not overwrite:
            raise FileExistsError(destination)
        encoded = (_canonical(self.to_dict()) + "\n").encode("utf-8")
        payload = gzip.compress(encoded, compresslevel=compresslevel, mtime=0)
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        temporary.write_bytes(payload)
        if destination.exists() and not overwrite:
            temporary.unlink(missing_ok=True)
            raise FileExistsError(destination)
        os.replace(temporary, destination)
        return destination


def capture_training_replay(
    environment: "BattleEnvV1",
    *,
    completed_match_index: int,
    worker_id: int,
    global_slot: int,
    local_slot: int,
    final_observation: Any | None = None,
) -> TrainingReplayV1:
    """Capture a terminal environment without another native API request."""

    episode = environment.episode_config
    episode_id = environment.episode_id
    if episode is None or episode_id is None:
        raise TrainingReplayError("environment must be reset before replay capture")
    observations = getattr(environment, "_raw", None)
    if not isinstance(observations, Mapping):
        raise TrainingReplayError("environment has no final native observation")
    owner_view = final_observation if final_observation is not None else environment.observe(owner=0)
    trace = environment.trace
    players = sorted(
        ({"owner": int(player.owner), "crowns": int(player.crowns)} for player in owner_view.players),
        key=lambda item: item["owner"],
    )
    terminal = {
        **owner_view.terminal.to_dict(),
        "truncated": bool(owner_view.truncated),
        "native_tick": int(observations["tick"]),
        "trace_end_native_tick": (int(trace[-1].end_native_tick) if trace else int(observations["tick"])),
        "players": players,
    }
    return TrainingReplayV1(
        completed_match_index=completed_match_index,
        ruleset_id=environment.ruleset_id,
        episode_config=episode,
        episode_id=episode_id,
        operations=tuple(SnapshotOperationV1.from_runtime(operation) for operation in trace),
        terminal=terminal,
        source={"worker_id": int(worker_id), "global_slot": int(global_slot), "local_slot": int(local_slot)},
    )


def replay_filename(replay: TrainingReplayV1) -> str:
    episode = (
        "".join(character for character in replay.episode_id.lower() if character in "0123456789abcdef")[:12]
        or "unknown"
    )
    return f"match-{replay.completed_match_index:012d}-seed-{replay.episode_config.seed}-episode-{episode}.crr.json.gz"


def save_training_replay(replay: TrainingReplayV1, directory: str | os.PathLike[str]) -> Path:
    destination = Path(directory) / replay_filename(replay)
    try:
        return replay.save(destination)
    except FileExistsError:
        # A crash can leave an archive newer than the last checkpoint. On
        # resume, the same restored episode may cross the same completion
        # ordinal again. Treat that exact identity as an idempotent success,
        # while still failing closed on an unrelated filename collision.
        existing = TrainingReplayV1.load(destination)
        if (
            existing.completed_match_index != replay.completed_match_index
            or existing.episode_id != replay.episode_id
            or existing.ruleset_id != replay.ruleset_id
        ):
            raise TrainingReplayError(f"training replay filename collision: {destination}")
        return destination


def list_training_replays(directory: str | os.PathLike[str]) -> tuple[Path, ...]:
    root = Path(directory)
    if not root.exists():
        return ()
    return tuple(sorted(root.glob("match-*.crr.json.gz"), key=lambda path: path.name, reverse=True))
