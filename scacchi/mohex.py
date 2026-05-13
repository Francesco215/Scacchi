from __future__ import annotations

import os
import random
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import jax
import jax.numpy as jnp


DEFAULT_MOHEX_BINARY = (
    Path(__file__).resolve().parents[1]
    / "third_party"
    / "benzene-vanilla-cmake"
    / "build"
    / "src"
    / "mohex"
    / "mohex"
)


def mohex_binary() -> Path:
    return Path(os.environ.get("MOHEX_BINARY", DEFAULT_MOHEX_BINARY))


MOHEX_BINARY = mohex_binary()


def _action_to_notation(action: int, board_size: int) -> str:
    row, col = divmod(action, board_size)
    return f"{chr(ord('a') + col)}{row + 1}"


def _state_to_sgf(state, board_size: int) -> str:
    board = jax.device_get(state._x.board).reshape(board_size, board_size)
    to_play = int(jax.device_get(state._x.color))

    # PGX stores stones from the next-to-play color as positive values.
    if to_play == 0:
        black = board > 0
        white = board < 0
    else:
        black = board < 0
        white = board > 0

    moves: list[str] = []
    for row, col in jax.device_get(jnp.argwhere(jnp.asarray(black))):
        moves.append(f"B[{_action_to_notation(int(row) * board_size + int(col), board_size)}]")
    for row, col in jax.device_get(jnp.argwhere(jnp.asarray(white))):
        moves.append(f"W[{_action_to_notation(int(row) * board_size + int(col), board_size)}]")

    return (
        f"(;AP[Scacchi]FF[4]GM[11]SZ[{board_size}]"
        f"{';' if moves else ''}{';'.join(moves)})"
    )


def _mohex_move_to_action(move: str, board_size: int) -> int:
    move = move.strip().lower()
    if move in {"swap-pieces", "swap"}:
        return board_size * board_size
    if len(move) < 2 or not move[0].isalpha() or not move[1:].isdigit():
        raise ValueError(f"MoHex returned invalid move: {move!r}")
    col = ord(move[0]) - ord("a")
    row = int(move[1:]) - 1
    if not (0 <= row < board_size and 0 <= col < board_size):
        raise ValueError(f"MoHex returned out-of-bounds move: {move!r}")
    return row * board_size + col


def _first_legal_action(state) -> int:
    legal = jax.device_get(state.legal_action_mask)
    for action, is_legal in enumerate(legal):
        if bool(is_legal):
            return action
    raise ValueError("No legal actions available for non-terminal state.")


def _random_legal_action(state) -> int:
    legal = jax.device_get(state.legal_action_mask)
    candidates = [i for i, is_legal in enumerate(legal) if bool(is_legal)]
    if not candidates:
        raise ValueError("No legal actions available for non-terminal state.")
    return random.choice(candidates)


def _state_at(batch_state, index: int):
    return jax.tree_util.tree_map(lambda x: x[index], batch_state)


class MoHexProcess:
    def __init__(
        self,
        binary: str,
        max_memory: int | None = None,
        max_time: float | None = None,
        max_games: int | None = None,
        max_nodes: int | None = None,
        num_threads: int | None = None,
        dfpn_threads: int | None = 4,
        parallel_solver: bool = False,
        solver: bool = True,
    ):
        config = self._write_config(
            max_memory=max_memory,
            max_time=max_time,
            max_games=max_games,
            max_nodes=max_nodes,
            num_threads=num_threads,
            dfpn_threads=dfpn_threads,
            parallel_solver=parallel_solver,
            solver=solver,
        )
        self._config_path = config.name
        config.close()
        self._proc = subprocess.Popen(
            [binary, "--use-logfile=0", f"--config={self._config_path}"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    @staticmethod
    def _write_config(
        max_memory: int | None,
        max_time: float | None,
        max_games: int | None,
        max_nodes: int | None,
        num_threads: int | None,
        dfpn_threads: int | None,
        parallel_solver: bool,
        solver: bool,
    ):
        lines: list[str] = ["param_game allow_swap 0"]
        if max_games is not None:
            lines.append(f"param_mohex max_games {max_games}")
            if max_games < 11:
                lines.append(f"param_mohex expand_threshold {max_games - 1}")
        if solver:
            lines.extend(
                [
                    "param_mohex knowledge_threshold 0",
                    f"param_mohex use_parallel_solver {int(parallel_solver)}",
                ]
            )
            if dfpn_threads is not None:
                lines.append(f"param_dfpn threads {dfpn_threads}")
        if num_threads is not None:
            lines.append(f"param_mohex num_threads {num_threads}")
        if max_memory is not None:
            lines.append(f"param_mohex max_memory {max_memory}")
        if max_time is not None:
            lines.append("param_mohex use_time_management 1")
            lines.append(f"param_game game_time {max_time / 2}")
        if max_nodes is not None:
            lines.append(f"param_mohex max_nodes {max_nodes}")

        config = tempfile.NamedTemporaryFile("w", delete=False, prefix="mohex-config-")
        config.write("\n".join(lines))
        return config

    def _read_answer(self) -> str:
        assert self._proc.stdout is not None
        lines: list[str] = []
        while True:
            line = self._proc.stdout.readline()
            if line == "":
                stderr = ""
                if self._proc.stderr is not None:
                    stderr = self._proc.stderr.read()
                raise RuntimeError(f"MoHex exited unexpectedly. stderr:\n{stderr}")
            if line == "\n":
                break
            lines.append(line)

        answer = "".join(lines)
        if not answer.startswith("="):
            raise ValueError(answer[2:].strip() if len(answer) > 2 else answer)
        return answer[1:].strip() if len(lines) == 1 else answer[2:]

    def query(self, command: str) -> str:
        assert self._proc.stdin is not None
        self._proc.stdin.write(f"{command}\n")
        self._proc.stdin.flush()
        return self._read_answer()

    def genmove(self, state, board_size: int) -> int:
        sgf = _state_to_sgf(state, board_size)
        with tempfile.NamedTemporaryFile("w") as handle:
            handle.write(sgf)
            handle.flush()
            self.query(f"boardsize {board_size}")
            self.query("clear_board")
            self.query(f"loadsgf {handle.name}")
        color = "black" if int(jax.device_get(state._x.color)) == 0 else "white"
        move = self.query(f"reg_genmove {color}")
        if move.strip().lower() in {"invalid", "resign"}:
            return _first_legal_action(state)
        action = _mohex_move_to_action(move, board_size)
        if not bool(jax.device_get(state.legal_action_mask[action])):
            raise ValueError(f"MoHex returned illegal move {move!r} (action {action})")
        return action

    def close(self) -> None:
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
        try:
            os.unlink(self._config_path)
        except FileNotFoundError:
            pass


def make_mohex_action_selector(env, config, my_player: int = 0):
    del env
    board_size = 11 if config.board_size is None else int(config.board_size)
    binary = mohex_binary()
    if not binary.exists():
        raise FileNotFoundError(f"MoHex binary not found at {binary}")
    max_memory = getattr(config, "mohex_max_memory", None)
    max_time = getattr(config, "mohex_max_time", None)
    max_games = getattr(config, "mohex_max_games", None)
    max_nodes = getattr(config, "mohex_max_nodes", None)
    num_processes = max(1, int(getattr(config, "mohex_num_processes", 1)))
    num_threads = getattr(config, "mohex_num_threads", None)
    dfpn_threads = getattr(config, "mohex_dfpn_threads", None)
    parallel_solver = bool(getattr(config, "mohex_parallel_solver", False))

    def new_opponent() -> MoHexProcess:
        return MoHexProcess(
            str(binary),
            max_memory=max_memory,
            max_time=max_time,
            max_games=max_games,
            max_nodes=max_nodes,
            num_threads=num_threads,
            dfpn_threads=dfpn_threads,
            parallel_solver=parallel_solver,
        )

    opponents = [new_opponent() for _ in range(num_processes)]
    executor = ThreadPoolExecutor(max_workers=num_processes)

    def restart(slot: int) -> None:
        opponents[slot].close()
        opponents[slot] = new_opponent()

    def solve_chunk(slot: int, states):
        out = []
        for index, single_state in states:
            try:
                action = opponents[slot].genmove(single_state, board_size)
            except (RuntimeError, ValueError):
                restart(slot)
                try:
                    action = opponents[slot].genmove(single_state, board_size)
                except (RuntimeError, ValueError):
                    restart(slot)
                    action = _random_legal_action(single_state)
            out.append((index, action))
        return out

    def choose_mohex_actions(env_state) -> jax.Array:
        batch_size = int(env_state.terminated.shape[0])
        actions = jnp.zeros((batch_size,), dtype=jnp.int32)
        work = [[] for _ in range(num_processes)]
        next_slot = 0
        for i in range(batch_size):
            if bool(jax.device_get(env_state.terminated[i])):
                continue
            if int(jax.device_get(env_state.current_player[i])) == my_player:
                continue
            single_state = _state_at(env_state, i)
            work[next_slot].append((i, single_state))
            next_slot = (next_slot + 1) % num_processes

        futures = [
            executor.submit(solve_chunk, slot, states)
            for slot, states in enumerate(work)
            if states
        ]
        for future in as_completed(futures):
            for i, action in future.result():
                actions = actions.at[i].set(action)
        return actions

    def close() -> None:
        executor.shutdown(wait=True)
        for opponent in opponents:
            opponent.close()

    return choose_mohex_actions, close
