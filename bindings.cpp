#include <pybind11/pybind11.h>
#include "mcts_cpp.h"

namespace py = pybind11;

PYBIND11_MODULE(mcts_cpp_engine, m) {
    m.doc() = "High-performance C++ MCTS engine for AlphaZero";

    py::class_<MCTS_CPP>(m, "MCTS_CPP")
        // The constructor signature matches the C++ header and Python call
        .def(py::init<py::object, int, int, double, int, double, double>(),
             py::arg("inference_queue"), // The first argument is the queue
             py::arg("num_workers"),
             py::arg("batch_size"),
             py::arg("cpuct"),
             py::arg("virtual_loss"),
             py::arg("dirichlet_alpha"),
             py::arg("dirichlet_epsilon"))
        .def("search", &MCTS_CPP::search,
             py::arg("board"),
             py::arg("num_simulations"))
        .def("stop", &MCTS_CPP::stop_nn_manager);
}