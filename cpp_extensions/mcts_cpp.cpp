#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <mutex>
#include <atomic>

// Fast UCT calculation
float calc_uct_single(float Q, float N_c, float P, float N_p, float C = 1.5f) {
    if (std::isnan(P)) {
        P = 0.005f; // 1.0 / 200.0
    }
    
    float UCT = Q + P * C * std::sqrt(N_p) / (1.0f + N_c);
    
    if (std::isnan(UCT)) {
        UCT = 0.0f;
    }
    
    return UCT;
}

// Vectorized UCT calculation using SIMD
torch::Tensor calc_uct_vectorized(
    torch::Tensor Q_values,
    torch::Tensor N_values, 
    torch::Tensor P_values,
    float N_p,
    float C = 1.5f) {
    
    auto Q = Q_values.accessor<float, 1>();
    auto N = N_values.accessor<float, 1>();
    auto P = P_values.accessor<float, 1>();
    
    int64_t size = Q_values.size(0);
    auto UCT_values = torch::zeros({size}, torch::kFloat32);
    auto UCT = UCT_values.accessor<float, 1>();
    
    float sqrt_N_p = std::sqrt(N_p);
    
    // Use OpenMP for parallel computation
    #pragma omp parallel for
    for (int64_t i = 0; i < size; i++) {
        float p_val = P[i];
        if (std::isnan(p_val)) {
            p_val = 0.005f;
        }
        
        float uct = Q[i] + p_val * C * sqrt_N_p / (1.0f + N[i]);
        UCT[i] = std::isnan(uct) ? 0.0f : uct;
    }
    
    return UCT_values;
}

// Fast argmax
int64_t fast_argmax(torch::Tensor values) {
    auto accessor = values.accessor<float, 1>();
    int64_t max_idx = 0;
    float max_val = accessor[0];
    
    for (int64_t i = 1; i < values.size(0); i++) {
        if (accessor[i] > max_val) {
            max_val = accessor[i];
            max_idx = i;
        }
    }
    
    return max_idx;
}

// Batch UCT selection for multiple nodes
torch::Tensor batch_uct_select(
    std::vector<torch::Tensor> Q_values_list,
    std::vector<torch::Tensor> N_values_list,
    std::vector<torch::Tensor> P_values_list,
    torch::Tensor N_p_values,
    float C = 1.5f) {
    
    int64_t batch_size = Q_values_list.size();
    auto selected_indices = torch::zeros({batch_size}, torch::kInt64);
    auto indices = selected_indices.accessor<int64_t, 1>();
    auto N_p = N_p_values.accessor<float, 1>();
    
    #pragma omp parallel for
    for (int64_t b = 0; b < batch_size; b++) {
        auto uct_values = calc_uct_vectorized(
            Q_values_list[b], 
            N_values_list[b], 
            P_values_list[b], 
            N_p[b], 
            C
        );
        indices[b] = fast_argmax(uct_values);
    }
    
    return selected_indices;
}

// Thread-safe node statistics update
class AtomicFloat {
private:
    std::atomic<float> value;
public:
    AtomicFloat(float initial = 0.0f) : value(initial) {}
    
    void add(float val) {
        float current = value.load();
        while (!value.compare_exchange_weak(current, current + val));
    }
    
    float load() const {
        return value.load();
    }
};

// C++ Node structure for better memory layout
struct FastNode {
    AtomicFloat N;
    AtomicFloat sum_Q;
    std::vector<float> edge_P;
    std::vector<std::shared_ptr<FastNode>> children;
    std::mutex update_mutex;
    
    FastNode(float initial_Q, const std::vector<float>& move_probs) 
        : N(1.0f), sum_Q(initial_Q), edge_P(move_probs) {
        children.resize(move_probs.size());
    }
    
    void update_stats(float value, bool from_child) {
        float adjustment = from_child ? (1.0f - value) : value;
        N.add(1.0f);
        sum_Q.add(adjustment);
    }
    
    float get_Q() const {
        float n = N.load();
        return n > 0 ? sum_Q.load() / n : 0.0f;
    }
};

// Parallel tree traversal for batch selection
std::vector<std::vector<int64_t>> parallel_select_batch(
    std::shared_ptr<FastNode> root,
    int64_t num_simulations,
    float C = 1.5f) {
    
    std::vector<std::vector<int64_t>> paths(num_simulations);
    
    #pragma omp parallel for
    for (int64_t sim = 0; sim < num_simulations; sim++) {
        std::vector<int64_t> path;
        std::shared_ptr<FastNode> current = root;
        
        while (current && !current->children.empty()) {
            // Calculate UCT values for all children
            int64_t num_children = current->children.size();
            torch::Tensor Q_values = torch::zeros({num_children});
            torch::Tensor N_values = torch::zeros({num_children});
            torch::Tensor P_values = torch::zeros({num_children});
            
            auto Q = Q_values.accessor<float, 1>();
            auto N = N_values.accessor<float, 1>();
            auto P = P_values.accessor<float, 1>();
            
            for (int64_t i = 0; i < num_children; i++) {
                if (current->children[i]) {
                    Q[i] = 1.0f - current->children[i]->get_Q();
                    N[i] = current->children[i]->N.load();
                } else {
                    Q[i] = 0.0f;
                    N[i] = 0.0f;
                }
                P[i] = current->edge_P[i];
            }
            
            auto uct_values = calc_uct_vectorized(
                Q_values, N_values, P_values, 
                current->N.load(), C
            );
            
            int64_t selected = fast_argmax(uct_values);
            path.push_back(selected);
            
            // Move to selected child
            current = current->children[selected];
        }
        
        paths[sim] = path;
    }
    
    return paths;
}

PYBIND11_MODULE(mcts_cpp, m) {
    m.doc() = "C++ optimized MCTS operations";
    
    m.def("calc_uct_single", &calc_uct_single, "Calculate UCT for single edge",
          py::arg("Q"), py::arg("N_c"), py::arg("P"), py::arg("N_p"), py::arg("C") = 1.5f);
    
    m.def("calc_uct_vectorized", &calc_uct_vectorized, "Vectorized UCT calculation",
          py::arg("Q_values"), py::arg("N_values"), py::arg("P_values"), 
          py::arg("N_p"), py::arg("C") = 1.5f);
    
    m.def("fast_argmax", &fast_argmax, "Fast argmax implementation");
    
    m.def("batch_uct_select", &batch_uct_select, "Batch UCT selection",
          py::arg("Q_values_list"), py::arg("N_values_list"), 
          py::arg("P_values_list"), py::arg("N_p_values"), py::arg("C") = 1.5f);
    
    py::class_<FastNode, std::shared_ptr<FastNode>>(m, "FastNode")
        .def(py::init<float, const std::vector<float>&>())
        .def("update_stats", &FastNode::update_stats)
        .def("get_Q", &FastNode::get_Q)
        .def_readonly("N", &FastNode::N)
        .def_readonly("sum_Q", &FastNode::sum_Q)
        .def_readwrite("edge_P", &FastNode::edge_P)
        .def_readwrite("children", &FastNode::children);
    
    m.def("parallel_select_batch", &parallel_select_batch, "Parallel batch selection",
          py::arg("root"), py::arg("num_simulations"), py::arg("C") = 1.5f);
}