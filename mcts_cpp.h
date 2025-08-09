#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/pytypes.h>
#include <vector>
#include <string>
#include <mutex>
#include <thread>
#include <future>
#include <optional>
#include <atomic>

namespace py = pybind11;

class Node;

struct NNResult {
    py::dict policy;
    double value;
};

// NNResult and Node structures are defined here

class Node {
public:
    Node* parent;
    py::object move;
    double prior;
    py::dict children;
    std::mutex node_mtx;
    double visit_count = 0;
    double total_action_value = 0;

    Node(Node* parent, py::object move, double prior);
    double get_mean_action_value() const;
    Node* select_best_child(double cpuct);
    void expand(const py::dict& policy);
    void backpropagate(double value, int virtual_loss);
};

class MCTS_CPP {
public:
    MCTS_CPP(py::object inference_queue, int num_workers, int batch_size, double cpuct, int virtual_loss, double dirichlet_alpha, double dirichlet_epsilon);
    ~MCTS_CPP();

    py::object search(py::object board, int num_simulations);
    // --- FIX: Renamed the function to match the bindings.cpp file ---
    void stop_nn_manager();

private:
    py::object inference_queue;

    std::unique_ptr<Node> root;
    std::vector<std::thread> workers;
    std::atomic<bool> stop_threads{false};
    std::atomic<int> simulations_done{0};

    int NUM_WORKERS;
    int BATCH_SIZE;
    double CPUCT;
    int VIRTUAL_LOSS;
    double DIRICHLET_ALPHA;
    double DIRICHLET_EPSILON;

    void worker_loop(py::object root_board, int simulation_limit);
    NNResult evaluate_position(py::object board);
};
