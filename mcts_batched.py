# mcts_batched.py
# Efficient single-tree MCTS with vectorized nodes + central batched evaluator.
# Python 3.8+ compatible.

import math
import time
import hashlib
import threading
from threading import RLock, Lock, Event
from concurrent.futures import ThreadPoolExecutor
from collections import deque, OrderedDict, defaultdict
from typing import Optional, List, Dict, Deque, Any

import numpy as np
import chess
import torch

import encoder

# ---- Optional enhanced encoder (position history) ---------------------------------------------
try:
    from encoder_enhanced import PositionHistory  # type: ignore
    HISTORY_AVAILABLE = True
except Exception:
    HISTORY_AVAILABLE = False
    PositionHistory = None  # type: ignore

# ============================== Global defaults (device-tuned at runtime) =======================

# Exploration constant
C_PUCT = 1.5

# Virtual loss per traversing thread
VIRTUAL_LOSS = 1.0

# Defaults; actual values for the evaluator are derived from the model device in Root.__init__
DEFAULT_MAX_BATCH = 256
DEFAULT_TIMEOUT_MS = 2

# Thread pool workers (0 => auto = min(32, os.cpu_count() or 8))
MAX_WORKERS_HINT = 0

# Transposition table cap (LRU). 0 disables (safe default).
TT_MAX_NODES = 0

# Legal-move cache cap (FEN->list(moves))
LEGAL_CACHE_MAX = 50_000

# ================================================================================================

def _fen_hash(board: chess.Board) -> str:
    return hashlib.md5(board.fen().encode('utf-8')).hexdigest()

def _copy_board(b: chess.Board) -> chess.Board:
    try:
        return b.copy(stack=False)
    except TypeError:
        return b.copy()

# --------- LRU Transposition Table --------------------------------------------------------------

class _NodeStoreLRU:
    def __init__(self, capacity: int):
        self.capacity = max(0, int(capacity))
        self._d: "OrderedDict[str, Node]" = OrderedDict()
        self._lock = RLock()

    def get(self, key: str):
        if self.capacity <= 0:
            return None
        with self._lock:
            n = self._d.get(key)
            if n is not None:
                self._d.move_to_end(key, last=True)
            return n

    def put(self, key: str, node: "Node"):
        if self.capacity <= 0:
            return
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key, last=True)
                self._d[key] = node
                return
            self._d[key] = node
            while len(self._d) > self.capacity:
                self._d.popitem(last=False)

# --------- Legal moves cache (per FEN) ----------------------------------------------------------

class _LegalCacheLRU:
    def __init__(self, capacity: int):
        self.capacity = max(0, int(capacity))
        self._d: "OrderedDict[str, List[chess.Move]]" = OrderedDict()
        self._lock = RLock()

    def get_or_set(self, board: chess.Board) -> List[chess.Move]:
        key = _fen_hash(board)
        with self._lock:
            lm = self._d.get(key)
            if lm is not None:
                self._d.move_to_end(key, last=True)
                return lm
        lm_new = list(board.legal_moves)
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key, last=True)
                return self._d[key]
            self._d[key] = lm_new
            while len(self._d) > self.capacity:
                self._d.popitem(last=False)
        return lm_new

_TT = _NodeStoreLRU(TT_MAX_NODES)
_LEGAL_CACHE = _LegalCacheLRU(LEGAL_CACHE_MAX)

# --------- Utility ------------------------------------------------------------------------------

def _safe_probs(priors: np.ndarray) -> np.ndarray:
    """Sanitize priors: replace NaNs/negatives, renormalize; ensure dtype float32."""
    p = np.asarray(priors, dtype=np.float32).flatten()
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    p[p < 0.0] = 0.0
    s = float(p.sum())
    if s <= 0.0:
        if p.size == 0:
            return p
        p.fill(1.0 / p.size)
    else:
        p /= s
    return p

def _terminal_value_from_board(board: chess.Board) -> float:
    """Return leaf value in [-1, 1] from the perspective of the side to move at `board`."""
    res = board.result(claim_draw=True)
    outcome = encoder.parseResult(res)  # {-1, 0, +1} as White POV
    if not board.turn:
        outcome *= -1
    return float(outcome)

def _policy_for_board(board: chess.Board, raw_policy: Any, legal_moves: Optional[List[chess.Move]] = None) -> np.ndarray:
    """
    Ensure policy is aligned to THIS board's legal moves.
    If vector already matches len(legal_moves), sanitize.
    If full head (4608/4672), decode via encoder.decodePolicyOutput.
    Otherwise, uniform.
    """
    if legal_moves is None:
        legal_moves = list(board.legal_moves)
    n_legal = len(legal_moves)
    if n_legal == 0:
        return np.zeros(0, dtype=np.float32)

    p = np.asarray(raw_policy, dtype=np.float32).flatten()
    if p.size == n_legal:
        return _safe_probs(p)

    if hasattr(encoder, "decodePolicyOutput") and p.size in (4608, 4672, 72*64, 73*64):
        try:
            decoded = encoder.decodePolicyOutput(board, p)
            return _safe_probs(np.asarray(decoded, dtype=np.float32))
        except Exception:
            pass

    # Fallback: uniform keeps search alive
    return np.full(n_legal, 1.0 / n_legal, dtype=np.float32)

# ============================== Vectorized Node =================================================

class Node:
    """
    Lightweight, vectorized node:
      - moves: List[chess.Move]
      - P: prior probabilities per move        (float32[n])
      - N: visit counts per move               (float32[n])
      - W: accumulated values per move ([-1,1]) (float32[n])
      - V: virtual losses per move             (float32[n])
      - children: List[Optional[Node]]
    """
    __slots__ = ("lock", "moves", "P", "N", "W", "V", "children", "is_root")

    def __init__(self, moves: List[chess.Move], priors: np.ndarray, is_root: bool = False):
        self.lock = RLock()
        self.moves: List[chess.Move] = moves
        n = len(moves)
        self.P = _safe_probs(np.asarray(priors, dtype=np.float32).reshape(n)) if n > 0 else np.zeros(0, np.float32)
        self.N = np.zeros(n, dtype=np.float32)
        self.W = np.zeros(n, dtype=np.float32)
        self.V = np.zeros(n, dtype=np.float32)
        self.children: List[Optional["Node"]] = [None] * n
        self.is_root = is_root

    # ------------ Selection ---------------------------------------------------------------------

    def select_action(self, c_puct: float) -> int:
        """
        Vectorized PUCT argmax_a [ Q(s,a) + U(s,a) ].
        Q(s,a) = W / (N + V) (0 if N+V == 0); U(s,a) = c_puct * P * sqrt(sum(N+V)) / (1 + N + V)
        """
        Nv = self.N + self.V
        total = float(Nv.sum())
        if self.P.size == 0:
            return 0
        Q = np.zeros_like(Nv)
        mask = Nv > 0.0
        Q[mask] = self.W[mask] / Nv[mask]
        U = (c_puct * self.P * math.sqrt(1.0 + total)) / (1.0 + Nv)
        return int(np.argmax(Q + U))

    def add_virtual_loss(self, a: int, vloss: float):
        with self.lock:
            self.V[a] += vloss

    def clear_virtual_loss(self, a: int, vloss: float):
        with self.lock:
            self.V[a] = max(0.0, self.V[a] - vloss)

    def is_expanded(self, a: int) -> bool:
        return self.children[a] is not None

    def expand_child(self, a: int, child: "Node"):
        with self.lock:
            if self.children[a] is None:
                self.children[a] = child

    def total_visits(self) -> float:
        return float(self.N.sum())

    def edge_Q(self, a: int) -> float:
        n = self.N[a] + self.V[a]
        if n > 0.0:
            return float(self.W[a] / n)  # [-1,1]
        return 0.0

    def stats_string(self) -> str:
        N = self.N.copy()
        order = np.argsort(-N)
        header = '|{: ^10}|{: ^10}|{: ^10}|{: ^10}|{: ^10}|\n'.format('move', 'P', 'N', 'Q', 'UCT')
        lines = [header]
        Nv = self.N + self.V
        total = float(Nv.sum())
        Q = np.zeros_like(self.N)
        mask = Nv > 0.0
        Q[mask] = self.W[mask] / Nv[mask]               # [-1,1]
        Q01 = 0.5 * (Q + 1.0)                            # [0,1] for display
        U = C_PUCT * self.P * math.sqrt(1.0 + total) / (1.0 + Nv)
        UCT = Q + U
        for idx in order[:min(50, len(order))]:
            mv = self.moves[idx]
            lines.append('|{: ^10}|{:10.4f}|{:10.0f}|{:10.4f}|{:10.4f}|\n'.format(
                str(mv), float(self.P[idx]), float(self.N[idx]), float(Q01[idx]), float(UCT[idx])
            ))
        return ''.join(lines)

# ============================== Batched Evaluator ==============================================

class _EvalRequest:
    __slots__ = ("board", "history", "event", "value", "probs", "err", "key")

    def __init__(self, board: chess.Board, history: Any):
        self.board = _copy_board(board)
        self.history = history
        self.event = Event()
        self.value: Optional[float] = None
        self.probs: Optional[np.ndarray] = None
        self.err: Optional[Exception] = None
        self.key = _fen_hash(board)

class _Evaluator:
    """
    Dedicated thread that batches leaf evaluations across worker threads:
      - short timeout micro-batching
      - dedup identical positions
      - uses encoder.callNeuralNetworkBatched(...)
      - robust fallbacks to keep search alive on API differences
    """
    def __init__(self, model, max_batch: int, timeout_ms: int):
        self.model = model
        self.max_batch = int(max_batch)
        self.timeout = float(timeout_ms) / 1000.0
        self._cv = threading.Condition()
        self._queue: Deque[_EvalRequest] = deque()
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="MCTS-Evaluator", daemon=True)
        self._thread.start()

    def stop(self):
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        self._thread.join(timeout=2.0)

    def submit(self, board: chess.Board, history: Any) -> _EvalRequest:
        req = _EvalRequest(board, history)
        with self._cv:
            self._queue.append(req)
            self._cv.notify()
        return req

    def _run(self):
        while True:
            with self._cv:
                while not self._queue and not self._stop:
                    self._cv.wait(timeout=self.timeout)
                if self._stop and not self._queue:
                    return
                batch: List[_EvalRequest] = []
                while self._queue and len(batch) < self.max_batch:
                    batch.append(self._queue.popleft())

            # Deduplicate by FEN hash
            by_key: Dict[str, List[_EvalRequest]] = defaultdict(list)
            unique_boards: List[chess.Board] = []
            unique_histories: List[Any] = []
            order: List[str] = []
            for r in batch:
                if r.key not in by_key:
                    by_key[r.key] = [r]
                    unique_boards.append(r.board)
                    unique_histories.append(r.history)
                    order.append(r.key)
                else:
                    by_key[r.key].append(r)

            pass_histories = any(h is not None for h in unique_histories)
            histories_arg = unique_histories if pass_histories else None

            try:
                # Prefer batched calls; handle both signatures
                if histories_arg is not None:
                    values, probs = encoder.callNeuralNetworkBatched(unique_boards, self.model, histories_arg)
                else:
                    values, probs = encoder.callNeuralNetworkBatched(unique_boards, self.model)
            except TypeError:
                values, probs = encoder.callNeuralNetworkBatched(unique_boards, self.model)
            except Exception:
                # last resort: per-item single calls; keep search alive
                values, probs = [], []
                for i, b in enumerate(unique_boards):
                    h = unique_histories[i] if pass_histories else None
                    try:
                        v, p = encoder.callNeuralNetwork(b, self.model, h)
                    except TypeError:
                        v, p = encoder.callNeuralNetwork(b, self.model)
                    values.append(v)
                    probs.append(p)

            for i, key in enumerate(order):
                vs = float(values[i])  # [-1,1]
                ps = np.asarray(probs[i])
                for req in by_key[key]:
                    req.value = vs
                    req.probs = ps
                    req.event.set()

# ============================== Root Controller ================================================

class Root:
    """
    Single-tree MCTS with central batched evaluator.
    """

    def __init__(self,
                 board: chess.Board,
                 neuralNetwork,
                 position_history: Any = None,
                 use_enhanced_encoder: bool = False,
                 epsilon_dirichlet: float = 0.0,
                 alpha_dirichlet: float = 0.3):
        self.board_root = _copy_board(board)
        self.model = neuralNetwork
        self.use_enhanced_encoder = bool(use_enhanced_encoder and HISTORY_AVAILABLE)
        self.position_history_root = position_history if self.use_enhanced_encoder else None
        self.epsilon_dirichlet = float(epsilon_dirichlet)
        self.alpha_dirichlet = float(alpha_dirichlet)

        # Device-tuned evaluator parameters
        dev = next(self.model.parameters()).device if hasattr(self.model, "parameters") else torch.device("cpu")
        if dev.type == "mps":
            max_batch = 512
            timeout_ms = 6
        elif dev.type == "cuda":
            max_batch = 256
            timeout_ms = 2
        else:  # cpu or unknown
            max_batch = 64
            timeout_ms = 3

        # Root priors (align to legal moves robustly)
        val0, priors0_raw = encoder.callNeuralNetwork(self.board_root, self.model, self.position_history_root)
        moves0 = _LEGAL_CACHE.get_or_set(self.board_root)
        priors0 = _policy_for_board(self.board_root, priors0_raw, moves0)

        # Optional Dirichlet noise (for self-play training)
        if self.epsilon_dirichlet > 0.0 and len(priors0) > 0:
            noise = np.random.dirichlet([self.alpha_dirichlet] * len(priors0)).astype(np.float32)
            priors0 = _safe_probs((1.0 - self.epsilon_dirichlet) * priors0 + self.epsilon_dirichlet * noise)

        self.root = Node(moves0, priors0, is_root=True)

        # Batched evaluator thread (device tuned)
        self._evaluator = _Evaluator(self.model, max_batch, timeout_ms)

        # Per-root executor (so we can honor target concurrency from playchess.py)
        self._executor = None
        self._executor_lock = Lock()

        # Stats compatibility
        self.same_paths = 0

    # ---------------- Public Search Entrypoints --------------------------------------------------

    def parallelRollouts(self, board, neuralNetwork, num_parallel_rollouts: int):
        """Perform `num_parallel_rollouts` TOTAL rollouts; keep about that many tasks in flight."""
        self._run_total_rollouts(int(num_parallel_rollouts), target_concurrency=int(num_parallel_rollouts))

    def parallelRolloutsTotal(self, board, neuralNetwork, total_rollouts: int, num_parallel_rollouts: int):
        """Compatibility wrapper used by playchess.py."""
        self._run_total_rollouts(int(total_rollouts), target_concurrency=int(num_parallel_rollouts))

    # ---------------- Query methods --------------------------------------------------------------

    def getVisitCounts(self, board: chess.Board) -> np.ndarray:
        """4608-dim visit counts vector for training."""
        visit_counts = np.zeros(4608, dtype=np.float32)
        for i, mv in enumerate(self.root.moves):
            n = self.root.N[i]
            if n <= 0.0:
                continue
            if not board.turn:
                from encoder import mirrorMove
                mv_enc = mirrorMove(mv)
                plane, r, f = encoder.moveToIdx(mv_enc)
            else:
                plane, r, f = encoder.moveToIdx(mv)
            idx = plane * 64 + r * 8 + f
            if 0 <= idx < visit_counts.shape[0]:
                visit_counts[idx] = n
        return visit_counts

    def maxNSelect(self):
        """Return wrapper exposing move/N/Q (Q mapped to [0,1]) from the root."""
        if len(self.root.moves) == 0:
            return None
        total_N = float(self.root.N.sum())
        if total_N <= 0.0:
            idx = int(np.argmax(self.root.P)) if self.root.P.size > 0 else 0
        else:
            idx = int(np.argmax(self.root.N))
        mv = self.root.moves[idx]
        visits = float(self.root.N[idx])
        Q = self.root.edge_Q(idx)  # [-1,1]

        class _EdgeWrapper:
            def __init__(self, move, N, Q):
                self._move = move
                self._N = N
                self._Q = Q
            def getMove(self):
                return self._move
            def getN(self):
                return self._N
            def getQ(self):
                return 0.5 * (self._Q + 1.0)  # [0,1]

        return _EdgeWrapper(mv, visits, Q)

    def getN(self) -> float:
        return self.root.total_visits()

    def getQ(self) -> float:
        """Weighted average Q over root, mapped to [0,1]."""
        Nv = self.root.N + self.root.V
        mask = Nv > 0.0
        if not np.any(mask):
            return 0.5
        q_edges = np.zeros_like(Nv)
        q_edges[mask] = self.root.W[mask] / Nv[mask]  # [-1,1]
        weights = self.root.N
        total = float(weights.sum())
        if total <= 0.0:
            return 0.5
        q_avg = float((q_edges * weights).sum() / total)
        return 0.5 * (q_avg + 1.0)

    def getStatisticsString(self) -> str:
        return self.root.stats_string()

    def cleanup(self):
        if self._evaluator:
            self._evaluator.stop()
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None

    # ---------------- Internals ------------------------------------------------------------------

    def _ensure_executor(self, target_concurrency: int):
        if self._executor is None:
            with self._executor_lock:
                if self._executor is None:
                    # Size executor to requested concurrency, capped to keep the system responsive.
                    max_workers = max(1, min(256, target_concurrency))
                    self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="MCTS-Worker")

    def _run_total_rollouts(self, total: int, target_concurrency: int):
        if total <= 0:
            return
        self._ensure_executor(target_concurrency)

        outstanding = 0
        done = 0
        done_lock = Lock()
        finished = threading.Event()

        def _one():
            nonlocal done
            try:
                self._rollout_once()
            finally:
                with done_lock:
                    done += 1
                    if done >= total:
                        finished.set()

        # Keep ~target_concurrency tasks in flight to fill the evaluator’s batches
        burst = min(target_concurrency, total)
        for _ in range(burst):
            self._executor.submit(_one)
            outstanding += 1

        while not finished.is_set():
            time.sleep(0.0005)
            with done_lock:
                window = max(0, target_concurrency - (outstanding - done))
            to_submit = min(window, total - outstanding)
            for _ in range(to_submit):
                self._executor.submit(_one)
                outstanding += 1

        finished.wait()

    def _rollout_once(self):
        board = _copy_board(self.board_root)
        hist = None
        if self.use_enhanced_encoder and self.position_history_root is not None:
            hist = PositionHistory(self.position_history_root.history_length)  # type: ignore
            hist.history = list(self.position_history_root.history)  # type: ignore

        path_nodes: List[Node] = []
        path_actions: List[int] = []

        node = self.root
        while True:
            path_nodes.append(node)

            # Terminal at current node?
            legal_here = _LEGAL_CACHE.get_or_set(board)
            if not legal_here:
                v_leaf = _terminal_value_from_board(board)  # [-1,1]
                self._backup_path(path_nodes, path_actions, v_leaf)
                return

            # Selection
            a = node.select_action(C_PUCT)
            mv = node.moves[a]

            # Virtual loss on chosen edge
            node.add_virtual_loss(a, VIRTUAL_LOSS)
            path_actions.append(a)

            # Advance
            board.push(mv)
            if hist is not None:
                hist.add_position(board)  # type: ignore

            # If already expanded, descend
            if node.is_expanded(a):
                node = node.children[a]  # type: ignore
                continue

            # Short-circuit if leaf position is terminal
            child_legal = _LEGAL_CACHE.get_or_set(board)
            if not child_legal:
                v_leaf = _terminal_value_from_board(board)
                self._backup_path(path_nodes, path_actions, v_leaf)
                return

            # Evaluate leaf via batched evaluator
            req = self._evaluator.submit(board, hist)
            req.event.wait()
            if req.err is not None:
                # Fail closed: clear vloss and abort this rollout
                node.clear_virtual_loss(a, VIRTUAL_LOSS)
                return

            value = float(req.value)  # [-1,1] from leaf side
            priors_raw = req.probs
            priors = _policy_for_board(board, priors_raw, child_legal)

            # Create/attach child node (optionally reuse from TT)
            child_key = req.key
            child = _TT.get(child_key)
            if child is None:
                child = Node(child_legal, priors, is_root=False)
                _TT.put(child_key, child)
            node.expand_child(a, child)

            # Backup and return
            self._backup_path(path_nodes, path_actions, value)
            return

    def _backup_path(self, nodes: List[Node], actions: List[int], leaf_value: float):
        """
        Backpropagate from leaf to root:
          - leaf_value is from the perspective of the side to move at the leaf ([-1,1]).
          - **Flip first**, then add at each step toward the root (so parent gets -leaf_value).
          - Clear the virtual loss on each traversed edge.
        """
        v = leaf_value
        for i in range(len(actions) - 1, -1, -1):
            n = nodes[i]
            a = actions[i]
            v = -v  # FIX: flip BEFORE applying to this parent edge
            with n.lock:
                n.N[a] += 1.0
                n.W[a] += v
                n.V[a] = max(0.0, n.V[a] - VIRTUAL_LOSS)

# ============================== Compatibility helpers ==========================================

def clear_caches():
    """Clear global caches (legal moves + transposition table)."""
    global _TT, _LEGAL_CACHE
    _TT = _NodeStoreLRU(TT_MAX_NODES)
    _LEGAL_CACHE = _LegalCacheLRU(LEGAL_CACHE_MAX)

def callNeuralNetworkOptimized(board, neuralNetwork, history=None):
    return encoder.callNeuralNetwork(board, neuralNetwork, history)

def callNeuralNetworkBatchedOptimized(boards, neuralNetwork, histories=None):
    try:
        return encoder.callNeuralNetworkBatched(boards, neuralNetwork, histories)
    except TypeError:
        return encoder.callNeuralNetworkBatched(boards, neuralNetwork)
