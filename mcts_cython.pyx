# distutils: language = c++
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

import cython
import threading
import queue
import uuid
import math
from collections import namedtuple
import chess
import numpy as np

# --- FIX: Import from the new, independent config file ---
from mcts_config import Config, InferenceRequest, ThreadSafeCounter

# --- Cython C-level Declarations ---
cdef class CyNode:
    # C-level attributes for speed
    cdef public CyNode parent
    cdef public object children
    cdef public object move
    cdef public double visit_count
    cdef public double total_action_value
    cdef public double prior_probability
    cdef public object lock # A Python threading.Lock

    def __cinit__(self, CyNode parent, object move, double prior_p):
        self.parent = parent
        self.move = move
        self.prior_probability = prior_p
        self.visit_count = 0
        self.total_action_value = 0.0
        self.lock = threading.Lock()
        self.children = {}

    cpdef double get_mean_action_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.total_action_value / self.visit_count

    cpdef tuple select_best_child(self):
        cdef double best_score = -1e9
        cdef CyNode best_child = None
        cdef object best_move = None
        cdef double parent_visits_sqrt
        cdef double q_value, ucb_score, score

        if self.visit_count <= 0:
            return None, None

        parent_visits_sqrt = math.sqrt(self.visit_count)

        for move, child in self.children.items():
            with child.lock:
                q_value = -child.get_mean_action_value() if child.visit_count > 0 else 0.0
                ucb_score = Config.CPUCT * child.prior_probability * parent_visits_sqrt / (1 + child.visit_count)
                score = q_value + ucb_score

            if score > best_score:
                best_score = score
                best_child = child
                best_move = move
        return best_move, best_child

    cpdef expand(self, policy_dict):
        for move, prior in policy_dict.items():
            if move not in self.children:
                self.children[move] = CyNode(self, move, prior)

    cpdef backpropagate(self, double value):
        cdef CyNode node = self
        while node is not None:
            with node.lock:
                node.visit_count -= (Config.VIRTUAL_LOSS - 1)
                node.total_action_value += value
            value = -value
            node = node.parent

# --- Cython Worker Logic ---
cdef class CyMCTSWorkerLogic:
    cdef public int worker_id
    cdef public CyNode root_node
    cdef public object root_board
    cdef public object nn_manager
    cdef public object simulations_done
    cdef public object board
    cdef public object encoder

    def __init__(self, int worker_id, CyNode root_node, object root_board, object nn_manager, object simulations_done, object encoder_module):
        self.worker_id = worker_id
        self.root_node = root_node
        self.root_board = root_board
        self.nn_manager = nn_manager
        self.simulations_done = simulations_done
        self.board = self.root_board.copy()
        self.encoder = encoder_module

    cpdef run_one_simulation(self):
        while len(self.board.move_stack) > len(self.root_board.move_stack):
            self.board.pop()

        cdef list path = []
        cdef CyNode node = self.root_node
        cdef bint is_expanded
        cdef object move, outcome
        cdef CyNode child_node
        cdef double value

        while True:
            with node.lock:
                is_expanded = bool(node.children)
                node.visit_count += Config.VIRTUAL_LOSS
                if is_expanded:
                    move, child_node = node.select_best_child()
                    if child_node is None:
                        node.visit_count -= Config.VIRTUAL_LOSS
                        return
                else:
                    break
            path.append(node)
            self.board.push(move)
            node = child_node

        outcome = self.board.outcome(claim_draw=True)
        if outcome is not None:
            value = 0.0
            if outcome.winner is True: value = 1.0
            elif outcome.winner is False: value = -1.0
        else:
            encoded_state, mask = self.encoder.encodePositionForInference(self.board)
            request_id = uuid.uuid4().hex
            completion_event = threading.Event()
            self.nn_manager.inference_queue.put(
                InferenceRequest(request_id, encoded_state, mask, completion_event, self.board.fen())
            )
            completion_event.wait()
            policy_array, value = self.nn_manager.results_dict.pop(request_id)

            legal_moves = list(self.board.legal_moves)
            policy_dict = {move: prob for move, prob in zip(legal_moves, policy_array)}

            with node.lock:
                node.expand(policy_dict)

        node.backpropagate(value)
