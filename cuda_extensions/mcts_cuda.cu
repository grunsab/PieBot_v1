#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

// CUDA kernel for parallel UCT calculation
__global__ void calc_uct_kernel(
    const float* __restrict__ Q_values,
    const float* __restrict__ N_values,
    const float* __restrict__ P_values,
    float N_p,
    float C,
    float* __restrict__ UCT_values,
    int size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float P = P_values[idx];
        if (isnan(P)) {
            P = 0.005f;
        }
        
        float sqrt_N_p = sqrtf(N_p);
        float UCT = Q_values[idx] + P * C * sqrt_N_p / (1.0f + N_values[idx]);
        
        UCT_values[idx] = isnan(UCT) ? 0.0f : UCT;
    }
}

// CUDA kernel for batch UCT calculation (multiple nodes at once)
__global__ void batch_calc_uct_kernel(
    const float* __restrict__ Q_values,
    const float* __restrict__ N_values,
    const float* __restrict__ P_values,
    const float* __restrict__ N_p_values,
    float C,
    float* __restrict__ UCT_values,
    const int* __restrict__ node_offsets,
    int num_nodes) {
    
    int node_idx = blockIdx.y;
    int edge_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (node_idx < num_nodes) {
        int start = node_offsets[node_idx];
        int end = node_offsets[node_idx + 1];
        int local_idx = start + edge_idx;
        
        if (local_idx < end) {
            float P = P_values[local_idx];
            if (isnan(P)) {
                P = 0.005f;
            }
            
            float sqrt_N_p = sqrtf(N_p_values[node_idx]);
            float UCT = Q_values[local_idx] + P * C * sqrt_N_p / (1.0f + N_values[local_idx]);
            
            UCT_values[local_idx] = isnan(UCT) ? 0.0f : UCT;
        }
    }
}

// CUDA kernel for finding argmax in segments
__global__ void segmented_argmax_kernel(
    const float* __restrict__ values,
    const int* __restrict__ segment_offsets,
    int* __restrict__ argmax_indices,
    int num_segments) {
    
    int seg_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (seg_idx < num_segments) {
        int start = segment_offsets[seg_idx];
        int end = segment_offsets[seg_idx + 1];
        
        if (start < end) {
            float max_val = values[start];
            int max_idx = 0;
            
            for (int i = 1; i < end - start; i++) {
                if (values[start + i] > max_val) {
                    max_val = values[start + i];
                    max_idx = i;
                }
            }
            
            argmax_indices[seg_idx] = max_idx;
        }
    }
}

// Parallel tree path generation kernel
__global__ void generate_paths_kernel(
    const int* __restrict__ edge_indices,
    const int* __restrict__ node_children,
    const int* __restrict__ node_offsets,
    int* __restrict__ paths,
    int* __restrict__ path_lengths,
    int max_depth,
    int num_paths) {
    
    int path_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (path_idx < num_paths) {
        int current_node = 0;  // Start from root
        int depth = 0;
        
        while (depth < max_depth && current_node >= 0) {
            int node_offset = node_offsets[current_node];
            int num_children = node_offsets[current_node + 1] - node_offset;
            
            if (num_children == 0) break;
            
            // Get selected edge for this path and node
            int selected_edge = edge_indices[path_idx * max_depth + depth];
            if (selected_edge >= num_children) break;
            
            paths[path_idx * max_depth + depth] = selected_edge;
            current_node = node_children[node_offset + selected_edge];
            depth++;
        }
        
        path_lengths[path_idx] = depth;
    }
}

// Batched neural network position encoding kernel
__global__ void batch_encode_positions_kernel(
    const int* __restrict__ piece_positions,  // Flattened piece positions
    const int* __restrict__ castling_rights,  // Castling rights per position
    const int* __restrict__ position_offsets,  // Offsets for each position
    float* __restrict__ encoded_positions,    // Output tensor
    int num_positions,
    int num_planes) {
    
    int pos_idx = blockIdx.z;
    int plane = blockIdx.y;
    int square = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (pos_idx < num_positions && plane < num_planes && square < 64) {
        int output_idx = pos_idx * num_planes * 64 + plane * 64 + square;
        
        if (plane < 12) {
            // Piece planes
            int piece_start = position_offsets[pos_idx];
            int piece_end = position_offsets[pos_idx + 1];
            
            encoded_positions[output_idx] = 0.0f;
            
            for (int i = piece_start; i < piece_end; i += 3) {
                int piece_type = piece_positions[i];
                int piece_square = piece_positions[i + 1];
                
                if (piece_type == plane && piece_square == square) {
                    encoded_positions[output_idx] = 1.0f;
                    break;
                }
            }
        } else {
            // Castling planes
            int castling_idx = pos_idx * 4 + (plane - 12);
            encoded_positions[output_idx] = castling_rights[castling_idx] ? 1.0f : 0.0f;
        }
    }
}

// C++ wrapper functions
torch::Tensor cuda_calc_uct(
    torch::Tensor Q_values,
    torch::Tensor N_values,
    torch::Tensor P_values,
    float N_p,
    float C) {
    
    const int size = Q_values.size(0);
    auto UCT_values = torch::zeros_like(Q_values);
    
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    calc_uct_kernel<<<blocks, threads>>>(
        Q_values.data_ptr<float>(),
        N_values.data_ptr<float>(),
        P_values.data_ptr<float>(),
        N_p,
        C,
        UCT_values.data_ptr<float>(),
        size
    );
    
    return UCT_values;
}

torch::Tensor cuda_batch_calc_uct(
    torch::Tensor Q_values,
    torch::Tensor N_values,
    torch::Tensor P_values,
    torch::Tensor N_p_values,
    torch::Tensor node_offsets,
    float C) {
    
    const int total_edges = Q_values.size(0);
    const int num_nodes = node_offsets.size(0) - 1;
    auto UCT_values = torch::zeros_like(Q_values);
    
    // Find max edges per node for block configuration
    int max_edges = 0;
    auto offsets_cpu = node_offsets.cpu();
    auto offsets_accessor = offsets_cpu.accessor<int, 1>();
    for (int i = 0; i < num_nodes; i++) {
        int edges = offsets_accessor[i + 1] - offsets_accessor[i];
        max_edges = std::max(max_edges, edges);
    }
    
    const int threads = 128;
    const int blocks_x = (max_edges + threads - 1) / threads;
    dim3 blocks(blocks_x, num_nodes);
    
    batch_calc_uct_kernel<<<blocks, threads>>>(
        Q_values.data_ptr<float>(),
        N_values.data_ptr<float>(),
        P_values.data_ptr<float>(),
        N_p_values.data_ptr<float>(),
        C,
        UCT_values.data_ptr<float>(),
        node_offsets.data_ptr<int>(),
        num_nodes
    );
    
    return UCT_values;
}

torch::Tensor cuda_segmented_argmax(
    torch::Tensor values,
    torch::Tensor segment_offsets) {
    
    const int num_segments = segment_offsets.size(0) - 1;
    auto argmax_indices = torch::zeros({num_segments}, 
                                      torch::dtype(torch::kInt32).device(values.device()));
    
    const int threads = 256;
    const int blocks = (num_segments + threads - 1) / threads;
    
    segmented_argmax_kernel<<<blocks, threads>>>(
        values.data_ptr<float>(),
        segment_offsets.data_ptr<int>(),
        argmax_indices.data_ptr<int>(),
        num_segments
    );
    
    return argmax_indices;
}

std::tuple<torch::Tensor, torch::Tensor> cuda_generate_paths(
    torch::Tensor edge_indices,
    torch::Tensor node_children,
    torch::Tensor node_offsets,
    int max_depth,
    int num_paths) {
    
    auto paths = torch::zeros({num_paths, max_depth}, 
                             torch::dtype(torch::kInt32).device(edge_indices.device()));
    auto path_lengths = torch::zeros({num_paths}, 
                                   torch::dtype(torch::kInt32).device(edge_indices.device()));
    
    const int threads = 256;
    const int blocks = (num_paths + threads - 1) / threads;
    
    generate_paths_kernel<<<blocks, threads>>>(
        edge_indices.data_ptr<int>(),
        node_children.data_ptr<int>(),
        node_offsets.data_ptr<int>(),
        paths.data_ptr<int>(),
        path_lengths.data_ptr<int>(),
        max_depth,
        num_paths
    );
    
    return std::make_tuple(paths, path_lengths);
}

torch::Tensor cuda_batch_encode_positions(
    torch::Tensor piece_positions,
    torch::Tensor castling_rights,
    torch::Tensor position_offsets,
    int num_positions,
    int num_planes) {
    
    auto encoded = torch::zeros({num_positions, num_planes, 8, 8}, 
                               torch::dtype(torch::kFloat32).device(piece_positions.device()));
    
    dim3 threads(8, 1, 1);
    dim3 blocks(8, num_planes, num_positions);
    
    batch_encode_positions_kernel<<<blocks, threads>>>(
        piece_positions.data_ptr<int>(),
        castling_rights.data_ptr<int>(),
        position_offsets.data_ptr<int>(),
        encoded.data_ptr<float>(),
        num_positions,
        num_planes
    );
    
    return encoded;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("calc_uct", &cuda_calc_uct, "CUDA UCT calculation");
    m.def("batch_calc_uct", &cuda_batch_calc_uct, "Batch CUDA UCT calculation");
    m.def("segmented_argmax", &cuda_segmented_argmax, "Segmented argmax on GPU");
    m.def("generate_paths", &cuda_generate_paths, "Generate tree paths on GPU");
    m.def("batch_encode_positions", &cuda_batch_encode_positions, "Batch encode positions on GPU");
}