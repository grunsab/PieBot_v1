"""
Shared memory tree implementation for multi-process MCTS.

This module provides a tree structure that can be shared across multiple processes
using multiprocessing shared memory primitives.
"""

import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
import struct
import chess
from threading import Lock
import ctypes

# Constants for tree structure
MAX_NODES = 2000000  # Maximum nodes in tree
MAX_EDGES_PER_NODE = 256  # Maximum legal moves in chess position
EDGE_SIZE = 32  # Bytes per edge (move, P, child_idx, virtual_loss)
NODE_SIZE = 64  # Bytes per node (N, sum_Q, edges_start, edges_count, lock)

class SharedEdge:
    """Edge representation in shared memory."""
    
    def __init__(self, shared_mem, offset):
        self.shared_mem = shared_mem
        self.offset = offset
        self._lock = Lock()
        
    def get_move(self):
        """Get the chess move (stored as from_square, to_square, promotion)."""
        data = self.shared_mem.buf[self.offset:self.offset + 4]
        from_square, to_square, promotion, _ = struct.unpack('BBBB', data)
        if promotion == 255:  # No promotion
            return chess.Move(from_square, to_square)
        else:
            return chess.Move(from_square, to_square, promotion=promotion)
    
    def set_move(self, move):
        """Set the chess move."""
        from_square = move.from_square
        to_square = move.to_square
        promotion = move.promotion if move.promotion else 255
        data = struct.pack('BBBB', from_square, to_square, promotion, 0)
        self.shared_mem.buf[self.offset:self.offset + 4] = data
        
    def get_P(self):
        """Get prior probability."""
        data = self.shared_mem.buf[self.offset + 4:self.offset + 8]
        return struct.unpack('f', data)[0]
    
    def set_P(self, P):
        """Set prior probability."""
        data = struct.pack('f', P)
        self.shared_mem.buf[self.offset + 4:self.offset + 8] = data
        
    def get_child_idx(self):
        """Get child node index (-1 if no child)."""
        data = self.shared_mem.buf[self.offset + 8:self.offset + 12]
        return struct.unpack('i', data)[0]
    
    def set_child_idx(self, idx):
        """Set child node index."""
        data = struct.pack('i', idx)
        self.shared_mem.buf[self.offset + 8:self.offset + 12] = data
        
    def get_virtual_loss(self):
        """Get virtual loss (using atomic operations)."""
        # Use ctypes for atomic read
        ptr = ctypes.c_float.from_buffer(self.shared_mem.buf, self.offset + 12)
        return ptr.value
    
    def add_virtual_loss(self, amount):
        """Add virtual loss atomically."""
        with self._lock:
            current = self.get_virtual_loss()
            self.set_virtual_loss(current + amount)
    
    def set_virtual_loss(self, value):
        """Set virtual loss."""
        data = struct.pack('f', value)
        self.shared_mem.buf[self.offset + 12:self.offset + 16] = data
        
    def clear_virtual_loss(self):
        """Clear virtual loss."""
        self.set_virtual_loss(0.0)


class SharedNode:
    """Node representation in shared memory."""
    
    def __init__(self, shared_mem, offset, node_idx):
        self.shared_mem = shared_mem
        self.offset = offset
        self.node_idx = node_idx
        self._lock = mp.Lock()
        
    def get_N(self):
        """Get visit count."""
        data = self.shared_mem.buf[self.offset:self.offset + 8]
        return struct.unpack('d', data)[0]
    
    def get_sum_Q(self):
        """Get sum of Q values."""
        data = self.shared_mem.buf[self.offset + 8:self.offset + 16]
        return struct.unpack('d', data)[0]
    
    def get_Q(self):
        """Get average Q value."""
        N = self.get_N()
        if N == 0:
            return 0.0
        return self.get_sum_Q() / N
    
    def update_stats(self, value, from_child_perspective):
        """Thread-safe update of node statistics."""
        with self._lock:
            # Read current values
            data = self.shared_mem.buf[self.offset:self.offset + 16]
            N, sum_Q = struct.unpack('dd', data)
            
            # Update
            N += 1
            if from_child_perspective:
                sum_Q += 1.0 - value
            else:
                sum_Q += value
                
            # Write back
            data = struct.pack('dd', N, sum_Q)
            self.shared_mem.buf[self.offset:self.offset + 16] = data
    
    def get_edges_info(self):
        """Get edges start index and count."""
        data = self.shared_mem.buf[self.offset + 16:self.offset + 24]
        edges_start, edges_count = struct.unpack('ii', data)
        return edges_start, edges_count
    
    def set_edges_info(self, edges_start, edges_count):
        """Set edges start index and count."""
        data = struct.pack('ii', edges_start, edges_count)
        self.shared_mem.buf[self.offset + 16:self.offset + 24] = data
        
    def get_edge(self, edge_idx, edges_shared_mem):
        """Get edge by index."""
        edges_start, edges_count = self.get_edges_info()
        if edge_idx >= edges_count:
            return None
        edge_offset = (edges_start + edge_idx) * EDGE_SIZE
        return SharedEdge(edges_shared_mem, edge_offset)
    
    def get_edges(self, edges_shared_mem):
        """Get all edges."""
        edges_start, edges_count = self.get_edges_info()
        edges = []
        for i in range(edges_count):
            edge_offset = (edges_start + i) * EDGE_SIZE
            edges.append(SharedEdge(edges_shared_mem, edge_offset))
        return edges


class SharedTree:
    """MCTS tree in shared memory accessible by multiple processes."""
    
    def __init__(self, name_prefix="mcts_tree"):
        """Initialize shared memory tree."""
        self.name_prefix = name_prefix
        
        # Create shared memory for nodes
        self.nodes_shm_name = f"{name_prefix}_nodes"
        self.nodes_shm = shared_memory.SharedMemory(
            create=True, 
            size=MAX_NODES * NODE_SIZE,
            name=self.nodes_shm_name
        )
        
        # Create shared memory for edges  
        self.edges_shm_name = f"{name_prefix}_edges"
        self.edges_shm = shared_memory.SharedMemory(
            create=True,
            size=MAX_NODES * MAX_EDGES_PER_NODE * EDGE_SIZE,
            name=self.edges_shm_name
        )
        
        # Node and edge allocation counters (in shared memory)
        self.counters_shm_name = f"{name_prefix}_counters"
        self.counters_shm = shared_memory.SharedMemory(
            create=True,
            size=16,  # 8 bytes for node counter, 8 bytes for edge counter
            name=self.counters_shm_name
        )
        
        # Initialize counters
        struct.pack_into('qq', self.counters_shm.buf, 0, 0, 0)
        
        # Lock for allocation
        self.alloc_lock = mp.Lock()
        
    def allocate_node(self):
        """Allocate a new node, returns node index."""
        with self.alloc_lock:
            node_count, edge_count = struct.unpack('qq', self.counters_shm.buf[:16])
            if node_count >= MAX_NODES:
                raise RuntimeError("Maximum nodes reached")
            
            node_idx = node_count
            node_count += 1
            struct.pack_into('qq', self.counters_shm.buf, 0, node_count, edge_count)
            
            # Initialize node data
            offset = node_idx * NODE_SIZE
            # N=0, sum_Q=0, edges_start=-1, edges_count=0
            data = struct.pack('ddii', 0.0, 0.0, -1, 0)
            self.nodes_shm.buf[offset:offset + 24] = data
            
            return node_idx
    
    def allocate_edges(self, count):
        """Allocate edges, returns start index."""
        with self.alloc_lock:
            node_count, edge_count = struct.unpack('qq', self.counters_shm.buf[:16])
            if edge_count + count > MAX_NODES * MAX_EDGES_PER_NODE:
                raise RuntimeError("Maximum edges reached")
                
            edge_start = edge_count
            edge_count += count
            struct.pack_into('qq', self.counters_shm.buf, 0, node_count, edge_count)
            
            return edge_start
    
    def get_node(self, node_idx):
        """Get node by index."""
        if node_idx < 0:
            return None
        offset = node_idx * NODE_SIZE
        return SharedNode(self.nodes_shm, offset, node_idx)
    
    def create_root(self, board, value, move_probabilities):
        """Create root node."""
        node_idx = self.allocate_node()
        node = self.get_node(node_idx)
        
        # Initialize root with initial visit
        node.update_stats(value, False)
        
        # Add edges for legal moves
        legal_moves = list(board.legal_moves)
        if legal_moves:
            edges_start = self.allocate_edges(len(legal_moves))
            node.set_edges_info(edges_start, len(legal_moves))
            
            # Initialize edges
            for i, move in enumerate(legal_moves):
                edge_offset = (edges_start + i) * EDGE_SIZE
                edge = SharedEdge(self.edges_shm, edge_offset)
                edge.set_move(move)
                edge.set_P(move_probabilities[i])
                edge.set_child_idx(-1)
                edge.set_virtual_loss(0.0)
                
        return node_idx
    
    def expand_node(self, parent_edge, board, value, move_probabilities):
        """Expand a node (create child)."""
        # Check if already expanded
        if parent_edge.get_child_idx() >= 0:
            return False
            
        # Allocate new node
        node_idx = self.allocate_node()
        node = self.get_node(node_idx)
        
        # Initialize with first visit
        node.update_stats(value, False)
        
        # Add edges for legal moves
        legal_moves = list(board.legal_moves)
        if legal_moves:
            edges_start = self.allocate_edges(len(legal_moves))
            node.set_edges_info(edges_start, len(legal_moves))
            
            # Initialize edges
            for i, move in enumerate(legal_moves):
                edge_offset = (edges_start + i) * EDGE_SIZE
                edge = SharedEdge(self.edges_shm, edge_offset)
                edge.set_move(move)
                edge.set_P(move_probabilities[i])
                edge.set_child_idx(-1)
                edge.set_virtual_loss(0.0)
        
        # Link parent to child
        parent_edge.set_child_idx(node_idx)
        
        return True
    
    def cleanup(self):
        """Clean up shared memory."""
        self.nodes_shm.close()
        self.nodes_shm.unlink()
        self.edges_shm.close() 
        self.edges_shm.unlink()
        self.counters_shm.close()
        self.counters_shm.unlink()


class SharedTreeClient:
    """Client for accessing shared tree from worker processes."""
    
    def __init__(self, name_prefix="mcts_tree"):
        """Connect to existing shared tree."""
        self.name_prefix = name_prefix
        
        # Connect to shared memory
        self.nodes_shm = shared_memory.SharedMemory(name=f"{name_prefix}_nodes")
        self.edges_shm = shared_memory.SharedMemory(name=f"{name_prefix}_edges")
        self.counters_shm = shared_memory.SharedMemory(name=f"{name_prefix}_counters")
        
    def get_node(self, node_idx):
        """Get node by index."""
        if node_idx < 0:
            return None
        offset = node_idx * NODE_SIZE
        return SharedNode(self.nodes_shm, offset, node_idx)
    
    def cleanup(self):
        """Clean up client connections."""
        self.nodes_shm.close()
        self.edges_shm.close()
        self.counters_shm.close()