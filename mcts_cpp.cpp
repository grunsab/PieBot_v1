#include "mcts_cpp.h"
#include <iostream>
#include <random>
#include <cmath>
#include <chrono>
#include <stdexcept>
#include <thread>

Node::Node(Node* p, py::object m, double pr) : parent(p), move(m), prior(pr) {
    // GIL should already be held by caller
    children = py::dict();
}

double Node::get_mean_action_value() const {
    if (visit_count == 0) return 0.0;
    return total_action_value / visit_count;
}

Node* Node::select_best_child(double cpuct) {
    Node* best_child = nullptr;
    double best_score = -1e9;
    
    // GIL should already be held by caller
    
    std::lock_guard<std::mutex> lock(node_mtx);
    if (visit_count <= 0) return nullptr;
    double parent_visits_sqrt = std::sqrt(visit_count);

    for (auto item : children) {
        py::object move = py::reinterpret_borrow<py::object>(item.first);
        Node* child = static_cast<Node*>(item.second.cast<py::capsule>().get_pointer());

        std::lock_guard<std::mutex> child_lock(child->node_mtx);
        double q_value = (child->visit_count > 0) ? -child->get_mean_action_value() : 0.0;
        double ucb_score = cpuct * child->prior * parent_visits_sqrt / (1.0 + child->visit_count);
        double score = q_value + ucb_score;

        if (score > best_score) {
            best_score = score;
            best_child = child;
        }
    }
    return best_child;
}

void Node::expand(const py::dict& policy) {
    // GIL should already be held by caller
    std::lock_guard<std::mutex> lock(node_mtx);
    for (auto item : policy) {
        py::object move = py::reinterpret_borrow<py::object>(item.first);
        double prior = item.second.cast<double>();
        if (!children.contains(move)) {
            children[move] = py::capsule(new Node(this, move, prior), [](void *p) { delete static_cast<Node*>(p); });
        }
    }
}

void Node::backpropagate(double value, int virtual_loss) {
    Node* current = this;
    while (current != nullptr) {
        {
            std::lock_guard<std::mutex> lock(current->node_mtx);
            // Net effect is +1 visit, removing virtual loss and adding the real visit
            current->visit_count -= (virtual_loss - 1);
            current->total_action_value += value;
        }
        value = -value;
        current = current->parent;
    }
}

// --- MCTS_CPP Implementation ---

MCTS_CPP::MCTS_CPP(py::object queue_obj, int num_workers, int batch_size, double cpuct, int virtual_loss, double dirichlet_alpha, double dirichlet_epsilon)
    : inference_queue(queue_obj), NUM_WORKERS(num_workers), BATCH_SIZE(batch_size), CPUCT(cpuct), VIRTUAL_LOSS(virtual_loss), DIRICHLET_ALPHA(dirichlet_alpha), DIRICHLET_EPSILON(dirichlet_epsilon) {
    // The constructor is now very simple, just storing the passed-in values.
    // No Python modules are imported here.
}

MCTS_CPP::~MCTS_CPP() {
    // Ensure threads are stopped cleanly when the object is destroyed
    stop_threads.exchange(true);
    for (auto& worker : workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

NNResult MCTS_CPP::evaluate_position(py::object board) {
    py::gil_scoped_acquire acquire;
    
    // Import the InferenceRequest class from Python
    py::module_ inference_module = py::module_::import("inference_request");
    py::object InferenceRequest = inference_module.attr("InferenceRequest");
    
    // Create a new InferenceRequest with the board
    py::object req = InferenceRequest(board);
    
    // Put the request in the queue
    inference_queue.attr("put")(req);
    
    // Wait for the result
    py::object result_dict = req.attr("wait_for_result")(10.0);
    
    // Convert Python result to C++ NNResult
    NNResult result;
    try {
        result.policy = result_dict["policy"].cast<py::dict>();
        result.value = result_dict["value"].cast<double>();
    } catch (const py::error_already_set& e) {
        std::cerr << "Error converting result: " << e.what() << std::endl;
        throw;
    }
    
    return result;
}

void MCTS_CPP::worker_loop(py::object root_board, int simulation_limit) {
    // The entire worker loop needs to manage GIL carefully
    // We'll acquire and release it as needed
    
    py::gil_scoped_acquire acquire;
    py::object board = root_board.attr("copy")();
    
    while (this->simulations_done.load(std::memory_order_relaxed) < simulation_limit && !this->stop_threads.load(std::memory_order_relaxed)) {
        try {
            // Reset board state to the root
            while (py::len(board.attr("move_stack")) > py::len(root_board.attr("move_stack"))) {
                board.attr("pop")();
            }

            Node* node = root.get();
            std::vector<Node*> path;
            
            // Selection with Virtual Loss
            while (true) {
                {
                    std::lock_guard<std::mutex> lock(node->node_mtx);
                    node->visit_count += this->VIRTUAL_LOSS;
                    path.push_back(node);

                    if (node->children.empty()) {
                        break;
                    }
                }
                
                Node* next_node = node->select_best_child(this->CPUCT);

                if (next_node == nullptr) break;
                node = next_node;
                
                board.attr("push")(node->move);
            }

            double value;
            py::object outcome = board.attr("outcome")(py::arg("claim_draw")=true);

            if (!outcome.is_none()) {
                auto winner_opt = py::getattr(outcome, "winner").cast<std::optional<bool>>();
                if (!winner_opt.has_value()) value = 0.0; // Draw
                else if (winner_opt.value()) value = 1.0; // White wins
                else value = -1.0; // Black wins
            } else {
                // No GIL is held here while waiting for the NN
                NNResult nn_result = evaluate_position(board);
                value = nn_result.value;
                node->expand(nn_result.policy);
            }
            
            bool is_white_turn = board.attr("turn").cast<bool>();
            double current_value = is_white_turn ? value : -value;

            // Backpropagate the result
            node->backpropagate(current_value, this->VIRTUAL_LOSS);

            this->simulations_done.fetch_add(1, std::memory_order_relaxed);

        } catch (const py::error_already_set& e) {
            std::cerr << "Caught Python exception in worker thread: " << e.what() << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Caught C++ exception in worker thread: " << e.what() << std::endl;
        }
    }
}

py::object MCTS_CPP::search(py::object board, int num_simulations) {
    NNResult root_eval = evaluate_position(board);
    
    py::dict noisy_policy;
    {
        py::gil_scoped_acquire acquire;
        py::list legal_moves = board.attr("legal_moves");
        size_t n_moves = py::len(legal_moves);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::gamma_distribution<> gamma(DIRICHLET_ALPHA, 1.0);
        std::vector<double> noise(n_moves);
        double sum = 0;
        for(size_t i = 0; i < n_moves; ++i) {
            noise[i] = gamma(gen);
            sum += noise[i];
        }
        if (sum > 1e-6) {
            for(size_t i = 0; i < n_moves; ++i) noise[i] /= sum;
        }

        for(size_t i = 0; i < n_moves; ++i) {
            py::object move = legal_moves[i];
            double original_prob = root_eval.policy.contains(move) ? root_eval.policy[move].cast<double>() : 0.0;
            noisy_policy[move] = (1.0 - DIRICHLET_EPSILON) * original_prob + DIRICHLET_EPSILON * noise[i];
        }
    }

    root = std::make_unique<Node>(nullptr, py::none(), 1.0);
    root->expand(noisy_policy);

    this->simulations_done.store(0);
    this->stop_threads.store(false);

    workers.clear();
    for (int i = 0; i < NUM_WORKERS; ++i) {
        workers.emplace_back(&MCTS_CPP::worker_loop, this, board, num_simulations);
    }

    // Release the GIL while the C++ threads are running.
    // This allows other Python threads (like a UI) to run.
    {
        py::gil_scoped_release release;
        for (auto& worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    } // GIL is re-acquired when this scope ends

    py::gil_scoped_acquire acquire;
    double max_visits = -1;
    py::object best_move = py::none();

    for (auto item : root->children) {
        Node* child = static_cast<Node*>(item.second.cast<py::capsule>().get_pointer());
        if (child->visit_count > max_visits) {
            max_visits = child->visit_count;
            best_move = child->move;
        }
    }
    return best_move;
}

void MCTS_CPP::stop_nn_manager() {
    // This function is called from Python to stop the NNManager thread
    // Currently empty as the NNManager is managed entirely from Python
    // This function exists to maintain API compatibility
}
