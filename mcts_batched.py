# mcts_batched.py
# Single-tree MCTS with:
#   - Correct PUCT math under parallelism (Q uses N only; virtual loss only reduces U)
#   - Asynchronous producer/consumer pipeline for leaves (keeps GPU busy with few Python threads)
#   - Integration with your existing parallel inference server (v2 if available, else v1)
#   - Windows-safe: server process is NOT daemonic (so it may spawn children/Manager)
# Python 3.8+ compatible (no PEP 604/585 types).

import math
import time
import hashlib
import threading
from threading import RLock, Lock, Event
from collections import deque, OrderedDict, defaultdict
from typing import Optional, List, Dict, Deque, Any
import queue
import uuid
import os
import atexit

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

# ---- Inference server (existing in your repo) -------------------------------------------------
import multiprocessing as mp
from multiprocessing import Process, Queue as MPQueue, Event as MPEvent, Manager
from inference_server import InferenceRequest, InferenceResult, start_inference_server_from_state

USE_V2 = False
try:
    from inference_server_parallel_v2 import (
        start_inference_server_from_state as start_parallel_inference_servers_from_state_v2
    )
    USE_V2 = True
except Exception:
    try:
        from inference_server_parallel import (
            start_parallel_inference_servers_from_state as start_parallel_inference_servers_from_state_v1
        )
        USE_V2 = False
    except Exception:
        # We will fall back to start_inference_server_from_state
        start_parallel_inference_servers_from_state_v1 = None  # type: ignore

# ============================== Tunables ========================================================

# Exploration constant (tune 1.25–2.5 for your net).
C_PUCT = 1.5

# Virtual loss per traversing thread (discourages path collisions).
VIRTUAL_LOSS = 1.0

# Evaluator micro-batching (device tuned later).
DEFAULT_MAX_BATCH = 512
DEFAULT_TIMEOUT_MS = 2     # CUDA ~2ms; MPS ~6ms (adjusted below)

# Transposition table cap (LRU). 0 disables (safe default).
TT_MAX_NODES = 0

# Legal-move cache cap.
LEGAL_CACHE_MAX = 50_000

# Selector threads: min(os.cpu_count(), num_threads, cap).
SELECTOR_THREADS_CAP = 32

# Cap waiting leaves; capacity ≈ CAP_MULT * max_batch
LEAF_QUEUE_CAP_MULT = 6

# Number of inference server workers (only for parallel server variants).
NUM_INFERENCE_SERVERS = 4

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

    if hasattr(encoder, "decodePolicyOutput") and p.size in (4608, 4672, 72 * 64, 73 * 64):
        try:
            decoded = encoder.decodePolicyOutput(board, p)
            return _safe_probs(np.asarray(decoded, dtype=np.float32))
        except Exception:
            pass

    return np.full(n_legal, 1.0 / n_legal, dtype=np.float32)

# ============================== Vectorized Node =================================================

class Node:
    """
    Lightweight, vectorized node:
      - moves: List[chess.Move]
      - P: prior probabilities per move        (float32[n])
      - N: visit counts per move               (float32[n])
      - W: accumulated values per move ([-1,1]) (float32[n])  <-- parent POV
      - V: virtual losses per move             (float32[n])   <-- discourages collisions
      - children: List[Optional[Node]]
    """
    __slots__ = ("lock", "moves", "P", "N", "W", "V", "children", "is_root")

    def __init__(self, moves: List[chess.Move], priors: np.ndarray, is_root: bool = False):
        self.lock = RLock()
        self.moves: List[chess.Move] = moves
        n = len(moves)
        self.P = _safe_probs(np.asarray(priors, dtype=np.float32).reshape(n)) if n > 0 else np.zeros(0, np.float32)
        self.N = np.zeros(n, dtype=np.float32)
        self.W = np.zeros(n, dtype=np.float32)   # sum of [-1, 1] values in this node's POV for each edge
        self.V = np.zeros(n, dtype=np.float32)   # virtual losses (NOT counted in Q)
        self.children: List[Optional["Node"]] = [None] * n
        self.is_root = is_root

    # ------------ Selection ---------------------------------------------------------------------

    def select_action(self, c_puct: float) -> int:
        """
        PUCT argmax_a [ Q(s,a) + U(s,a) ] where:
          Q(s,a) = W / max(N, eps)    (no virtual losses here)
          U(s,a) = c_puct * P * sqrt(sum(N) + 1) / (1 + N + V)
        """
        if self.P.size == 0:
            return 0

        # Q part
        eps = 1e-8
        Q = np.zeros_like(self.N)
        mask = self.N > 0.0
        Q[mask] = self.W[mask] / (self.N[mask] + eps)

        # U part (virtual loss reduces attractiveness of heavily contested edges)
        totalN = float(self.N.sum())
        U = (c_puct * self.P * math.sqrt(1.0 + totalN)) / (1.0 + self.N + self.V)

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
        if self.N[a] > 0.0:
            return float(self.W[a] / self.N[a])  # [-1,1]
        return 0.0

    def stats_string(self) -> str:
        N = self.N.copy()
        order = np.argsort(-N)
        header = '|{: ^10}|{: ^10}|{: ^10}|{: ^10}|{: ^10}|\n'.format('move', 'P', 'N', 'Q', 'UCT')
        lines = [header]
        Q = np.zeros_like(self.N)
        mask = self.N > 0.0
        Q[mask] = self.W[mask] / self.N[mask]               # [-1,1]
        Q01 = 0.5 * (Q + 1.0)                                # [0,1]
        totalN = float(self.N.sum())
        U = C_PUCT * self.P * math.sqrt(1.0 + totalN) / (1.0 + self.N + self.V)
        UCT = Q + U
        for idx in order[:min(50, len(order))]:
            mv = self.moves[idx]
            lines.append('|{: ^10}|{:10.4f}|{:10.0f}|{:10.4f}|{:10.4f}|\n'.format(
                str(mv), float(self.P[idx]), float(self.N[idx]), float(Q01[idx]), float(UCT[idx])
            ))
        return ''.join(lines)

# ============================== Inference Server Evaluator =====================================

class _ServerHandle:
    """
    Owns the parallel inference server subprocess(es) and provides a main request queue.
    Singleton across Roots (started once, reused across moves).
    """
    def __init__(self, model, max_batch: int, timeout_ms: int):
        self.model = model
        self.main_queue: MPQueue = mp.Queue()
        self.stop_event: MPEvent = mp.Event()
        self.process: Optional[Process] = None
        self.manager = Manager()  # For per-request result queues (proxy objects picklable across processes)

        # Extract model state on CPU (avoid CUDA pickling issues)
        original_device = next(self.model.parameters()).device
        model_state = self.model.cpu().state_dict()

        # Detect config (channels/blocks)
        try:
            conv1_out = self.model.convBlock1.conv1.out_channels
            num_blocks = len(self.model.residualBlocks)
            model_config = {'num_blocks': num_blocks, 'num_channels': conv1_out}
        except Exception:
            model_config = {'num_blocks': 20, 'num_channels': 256}

        # Device type for the server side
        if original_device.type == 'cuda':
            device_type = 'cuda'
        elif original_device.type == 'mps':
            device_type = 'mps'
        else:
            device_type = 'cpu'

        # Pick an entrypoint
        if USE_V2:
            target_fn = start_parallel_inference_servers_from_state_v2
            args = (model_state, model_config, device_type,
                    self.main_queue, self.stop_event,
                    NUM_INFERENCE_SERVERS, max_batch, timeout_ms)
        elif start_parallel_inference_servers_from_state_v1 is not None:
            target_fn = start_parallel_inference_servers_from_state_v1
            args = (model_state, model_config, device_type,
                    self.main_queue, self.stop_event,
                    NUM_INFERENCE_SERVERS, max_batch, timeout_ms)
        else:
            target_fn = start_inference_server_from_state
            args = (model_state, model_config, device_type,
                    self.main_queue, self.stop_event,
                    NUM_INFERENCE_SERVERS, max_batch, timeout_ms)

        # Start server process (IMPORTANT: NOT daemonic on Windows)
        self.process = Process(target=target_fn, args=args)
        # Do NOT set self.process.daemon = True  <-- this caused the crash on Windows
        self.process.start()

        # Return model to its original device in this main process
        self.model.to(original_device)

    def make_result_queue(self):
        return self.manager.Queue()

    def close(self):
        try:
            self.stop_event.set()
        except Exception:
            pass
        try:
            if self.process and self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=2.0)
        except Exception:
            pass
        try:
            self.manager.shutdown()
        except Exception:
            pass

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

class _ServerEvaluator:
    """
    Batching front-end that submits requests to your parallel inference server.
    - Collects up to `max_batch` or waits `timeout_ms`
    - Deduplicates positions
    - Sends a list of (InferenceRequest, result_queue) to the server
    - Waits for results for that batch, signals waiting callers
    """
    def __init__(self, model, max_batch: int, timeout_ms: int):
        self.handle = _ServerHandle(model, max_batch, timeout_ms)
        self.max_batch = int(max_batch)
        self.timeout = float(timeout_ms) / 1000.0
        self._cv = threading.Condition()
        self._queue: Deque[_EvalRequest] = deque()
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="MCTS-Evaluator-Server", daemon=True)
        self._thread.start()

    def stop(self):
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        self._thread.join(timeout=2.0)
        self.handle.close()

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

            # Deduplicate by FEN
            by_key: Dict[str, List[_EvalRequest]] = defaultdict(list)
            unique_reqs: List[_EvalRequest] = []
            for r in batch:
                if r.key not in by_key:
                    by_key[r.key] = [r]
                    unique_reqs.append(r)
                else:
                    by_key[r.key].append(r)

            if not unique_reqs:
                continue

            # Build server request list
            req_list = []
            for r in unique_reqs:
                res_q = self.handle.make_result_queue()
                inf_req = InferenceRequest(f"mcts_{uuid.uuid4().hex[:10]}", r.board.fen(), 0)
                req_list.append((inf_req, res_q))

            # Send to server in one shot
            self.handle.main_queue.put(req_list)

            # Collect results and signal waiters
            for i, r in enumerate(unique_reqs):
                _, res_q = req_list[i]
                try:
                    res = res_q.get()
                    r.value = float(res.value)  # [-1,1] leaf POV
                    r.probs = np.asarray(res.move_probabilities, dtype=np.float32)  # already decoded to legal order
                    for rr in by_key[r.key]:
                        rr.value = r.value
                        rr.probs = r.probs
                        rr.event.set()
                except Exception as e:
                    for rr in by_key[r.key]:
                        rr.err = e
                        rr.event.set()

# ============================== Root Controller (async pipeline) ===============================

class _LeafContext:
    __slots__ = ("nodes", "actions", "board", "history", "parent", "parent_a", "key", "child_legal")

    def __init__(self,
                 nodes: List[Node],
                 actions: List[int],
                 board: chess.Board,
                 history: Any,
                 parent: Node,
                 parent_a: int,
                 key: str,
                 child_legal: List[chess.Move]):
        self.nodes = nodes
        self.actions = actions
        self.board = board
        self.history = history
        self.parent = parent
        self.parent_a = parent_a
        self.key = key
        self.child_legal = child_legal

class Root:
    """
    Single-tree MCTS with:
      - Selector threads (producers) that perform selection+virtual loss and enqueue leaf contexts.
      - Evaluator thread backed by your inference server (shared across Roots).
      - Main thread expands/backprops results as they arrive.
    """

    # Shared evaluator (server) across all Root instances
    _shared_eval: Optional[_ServerEvaluator] = None
    _shared_lock = Lock()
    _shared_params = None  # (max_batch, timeout_ms)

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

        # Device-tuned evaluator params (only used on first creation)
        dev = next(self.model.parameters()).device if hasattr(self.model, "parameters") else torch.device("cpu")
        if dev.type == "mps":
            max_batch = 512
            timeout_ms = 6
        elif dev.type == "cuda":
            max_batch = DEFAULT_MAX_BATCH
            timeout_ms = DEFAULT_TIMEOUT_MS
        else:
            max_batch = 64
            timeout_ms = 3
        self._max_batch = max_batch
        self._timeout_ms = timeout_ms

        # Root priors (first eval is local in-process; server used thereafter)
        val0, priors0_raw = encoder.callNeuralNetwork(self.board_root, self.model, self.position_history_root)
        moves0 = _LEGAL_CACHE.get_or_set(self.board_root)
        priors0 = _policy_for_board(self.board_root, priors0_raw, moves0)

        # Optional Dirichlet noise (for self-play training)
        if self.epsilon_dirichlet > 0.0 and len(priors0) > 0:
            noise = np.random.dirichlet([self.alpha_dirichlet] * len(priors0)).astype(np.float32)
            priors0 = _safe_probs((1.0 - self.epsilon_dirichlet) * priors0 + self.epsilon_dirichlet * noise)

        self.root = Node(moves0, priors0, is_root=True)

        # Ensure shared evaluator is up
        with Root._shared_lock:
            if Root._shared_eval is None:
                Root._shared_eval = _ServerEvaluator(self.model, max_batch, timeout_ms)
                Root._shared_params = (max_batch, timeout_ms)

        # Stats compatibility
        self.same_paths = 0

    # ---------------- Public Search Entrypoints --------------------------------------------------

    def parallelRollouts(self, board, neuralNetwork, num_parallel_rollouts: int):
        """Perform `num_parallel_rollouts` TOTAL rollouts (kept for compatibility)."""
        self._search(total=int(num_parallel_rollouts), target_concurrency=int(num_parallel_rollouts))

    def parallelRolloutsTotal(self, board, neuralNetwork, total_rollouts: int, num_parallel_rollouts: int):
        """Compatibility wrapper used by playchess.py."""
        self._search(total=int(total_rollouts), target_concurrency=int(num_parallel_rollouts))

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
        mask = self.root.N > 0.0
        if not np.any(mask):
            return 0.5
        q_edges = np.zeros_like(self.root.N)
        q_edges[mask] = self.root.W[mask] / self.root.N[mask]  # [-1,1]
        weights = self.root.N
        total = float(weights.sum())
        if total <= 0.0:
            return 0.5
        q_avg = float((q_edges * weights).sum() / total)
        return 0.5 * (q_avg + 1.0)

    def getStatisticsString(self) -> str:
        return self.root.stats_string()

    def cleanup(self):
        # Shared evaluator persists across Roots; cleaned up at exit.
        pass

    # ---------------- Async pipeline internals ---------------------------------------------------

    def _search(self, total: int, target_concurrency: int):
        if total <= 0:
            return

        # Runtime parameters
        n_selectors = max(1, min(SELECTOR_THREADS_CAP, os.cpu_count() or 8, target_concurrency))
        # If shared evaluator was created with different params, use those for queue sizing
        if Root._shared_params is not None:
            max_batch, _ = Root._shared_params
        else:
            max_batch = self._max_batch
        max_inflight = max(int(max_batch) * LEAF_QUEUE_CAP_MULT, target_concurrency)

        # Thread-safe counters
        produced = 0
        completed = 0
        produced_lock = Lock()
        completed_lock = Lock()
        stop_event = Event()

        leaf_q = queue.Queue(maxsize=max_inflight)  # type: ignore

        def selector_worker():
            nonlocal produced, completed
            base_board = _copy_board(self.board_root)
            base_hist = None
            if self.use_enhanced_encoder and self.position_history_root is not None:
                base_hist = PositionHistory(self.position_history_root.history_length)  # type: ignore
                base_hist.history = list(self.position_history_root.history)  # type: ignore
            while not stop_event.is_set():
                with produced_lock:
                    if produced >= total:
                        break
                ctx = self._select_to_leaf(base_board, base_hist)
                if ctx is None:
                    with completed_lock:
                        completed += 1
                        if completed >= total:
                            stop_event.set()
                    continue
                try:
                    leaf_q.put(ctx, timeout=0.01)
                except queue.Full:
                    time.sleep(0.001)
                    continue
                with produced_lock:
                    produced += 1
                    if produced >= total:
                        pass

        selectors = [threading.Thread(target=selector_worker, name=f"MCTS-Selector-{i+1}", daemon=True)
                     for i in range(n_selectors)]
        for t in selectors:
            t.start()

        pending: List[Any] = []  # list of (EvalRequest, LeafContext)

        try:
            while True:
                with completed_lock:
                    if completed >= total:
                        break

                # Fill pending up to ~target_concurrency
                while len(pending) < target_concurrency:
                    try:
                        ctx = leaf_q.get_nowait()
                    except queue.Empty:
                        break
                    req = Root._shared_eval.submit(ctx.board, ctx.history)  # type: ignore
                    pending.append((req, ctx))

                if not pending:
                    try:
                        ctx = leaf_q.get(timeout=0.002)
                        req = Root._shared_eval.submit(ctx.board, ctx.history)  # type: ignore
                        pending.append((req, ctx))
                    except queue.Empty:
                        time.sleep(0.0005)
                        continue

                # Consume any ready results
                i = 0
                advanced = False
                while i < len(pending):
                    req, ctx = pending[i]
                    if req.event.is_set():
                        self._expand_and_backup(ctx, req.value, req.probs)
                        pending.pop(i)
                        with completed_lock:
                            completed += 1
                            if completed >= total:
                                stop_event.set()
                        advanced = True
                    else:
                        i += 1

                if not advanced:
                    time.sleep(0.0005)

        finally:
            stop_event.set()
            for t in selectors:
                t.join(timeout=0.5)

    def _select_to_leaf(self, base_board: chess.Board, base_hist: Any) -> Optional["_LeafContext"]:
        board = _copy_board(base_board)
        hist = None
        if self.use_enhanced_encoder and base_hist is not None:
            hist = PositionHistory(base_hist.history_length)  # type: ignore
            hist.history = list(base_hist.history)  # type: ignore

        path_nodes: List[Node] = []
        path_actions: List[int] = []
        node = self.root

        while True:
            path_nodes.append(node)

            legal_here = _LEGAL_CACHE.get_or_set(board)
            if not legal_here:
                v_leaf = _terminal_value_from_board(board)  # [-1,1]
                self._backup_path(path_nodes, path_actions, v_leaf)
                return None

            a = node.select_action(C_PUCT)
            mv = node.moves[a]

            node.add_virtual_loss(a, VIRTUAL_LOSS)
            path_actions.append(a)

            board.push(mv)
            if hist is not None:
                hist.add_position(board)  # type: ignore

            if node.is_expanded(a):
                node = node.children[a]  # type: ignore
                continue

            child_legal = _LEGAL_CACHE.get_or_set(board)
            if not child_legal:
                v_leaf = _terminal_value_from_board(board)
                self._backup_path(path_nodes, path_actions, v_leaf)
                return None

            parent = node
            parent_a = a
            key = _fen_hash(board)
            return _LeafContext(path_nodes, path_actions, _copy_board(board), hist, parent, parent_a, key, child_legal)

    def _expand_and_backup(self, ctx: "_LeafContext", value_any: Any, probs_any: Any):
        # Server returns probs already decoded to legal order; still sanitize defensively
        value = float(value_any)  # [-1,1] leaf POV
        priors = _policy_for_board(ctx.board, probs_any, ctx.child_legal)

        child = _TT.get(ctx.key)
        if child is None:
            child = Node(ctx.child_legal, priors, is_root=False)
            _TT.put(ctx.key, child)
        ctx.parent.expand_child(ctx.parent_a, child)

        self._backup_path(ctx.nodes, ctx.actions, value)

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
            v = -v  # parent's POV
            with n.lock:
                n.N[a] += 1.0
                n.W[a] += v
                n.V[a] = max(0.0, n.V[a] - VIRTUAL_LOSS)

# ============================== Compatibility helpers & atexit =================================

def clear_caches():
    """Clear global caches (legal moves + transposition table)."""
    global _TT, _LEGAL_CACHE
    _TT = _NodeStoreLRU(TT_MAX_NODES)
    _LEGAL_CACHE = _LegalCacheLRU(LEGAL_CACHE_MAX)

def callNeuralNetworkOptimized(board, neuralNetwork, history=None):
    return encoder.callNeuralNetwork(board, neuralNetwork, history)

def callNeuralNetworkBatchedOptimized(boards, neuralNetwork, histories=None):
    # Not used by this file (we route via the parallel inference server), but keep API parity.
    try:
        return encoder.callNeuralNetworkBatched(boards, neuralNetwork, histories)
    except TypeError:
        return encoder.callNeuralNetworkBatched(boards, neuralNetwork)

@atexit.register
def _shutdown_shared_evaluator():
    # Stop the shared evaluator/server when Python exits
    try:
        if Root._shared_eval is not None:
            Root._shared_eval.stop()
            Root._shared_eval = None
    except Exception:
        pass
