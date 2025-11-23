// main.cu
#include "SongStage3.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <limits>
#include <cstring>

#define CHECK_CUDA(err) do { cudaError_t e = (err); if (e != cudaSuccess) { \
    std::cerr << "CUDA error: " << cudaGetErrorString(e) << " at " << __FILE__ << ":" << __LINE__ << std::endl; exit(1); } } while(0)

int main(int argc, char** argv) {
    const char* trainFile = (argc > 1) ? argv[1] : "letter.scale";
    const char* testFile  = (argc > 2) ? argv[2] : "letter.scale.t";

    std::cout << "Stage3 host: train=" << trainFile << " test=" << testFile << std::endl;

    // Example candidates + distances (normally vienen de Stage2)
    std::vector<int> h_candidates = {10, 20, 30, 40, 50};
    std::vector<float> h_distances = {0.5f, 1.2f, 0.1f, 0.9f, 1.5f};
    int num_cand = (int)h_candidates.size();

    // Device buffers for candidates & distances
    int *d_candidates = nullptr;
    float *d_distances = nullptr;
    CHECK_CUDA(cudaMalloc(&d_candidates, num_cand * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_distances, num_cand * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_candidates, h_candidates.data(), num_cand * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_distances, h_distances.data(), num_cand * sizeof(float), cudaMemcpyHostToDevice));

    // Q buffers (nodes + dists)
    int *d_q_nodes = nullptr;
    float *d_q_dists = nullptr;
    CHECK_CUDA(cudaMalloc(&d_q_nodes, MAX_Q_CAPACITY * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_q_dists, MAX_Q_CAPACITY * sizeof(float)));
    // init q_size = 0
    int h_q_size = 0;
    int *d_q_size = nullptr;
    CHECK_CUDA(cudaMalloc(&d_q_size, sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_q_size, &h_q_size, sizeof(int), cudaMemcpyHostToDevice));

    // top-K arrays: init dists = +inf, nodes = -1
    int *d_topk_nodes = nullptr;
    float *d_topk_dists = nullptr;
    CHECK_CUDA(cudaMalloc(&d_topk_nodes, TOPK_K * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_topk_dists, TOPK_K * sizeof(float)));
    std::vector<int> h_topk_nodes(TOPK_K, -1);
    std::vector<float> h_topk_dists(TOPK_K, std::numeric_limits<float>::infinity());
    CHECK_CUDA(cudaMemcpy(d_topk_nodes, h_topk_nodes.data(), TOPK_K * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_topk_dists, h_topk_dists.data(), TOPK_K * sizeof(float), cudaMemcpyHostToDevice));

    // visited table: init to -1
    int *d_visited = nullptr;
    CHECK_CUDA(cudaMalloc(&d_visited, VISITED_CAPACITY * sizeof(int)));
    std::vector<int> h_visited(VISITED_CAPACITY, -1);
    CHECK_CUDA(cudaMemcpy(d_visited, h_visited.data(), VISITED_CAPACITY * sizeof(int), cudaMemcpyHostToDevice));

    // candidates_processed counter
    int h_processed = 0;
    int *d_processed = nullptr;
    CHECK_CUDA(cudaMalloc(&d_processed, sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_processed, &h_processed, sizeof(int), cudaMemcpyHostToDevice));

    // Fill Stage3Buffers
    Stage3Buffers bufs;
    bufs.candidates = d_candidates;
    bufs.distances = d_distances;
    bufs.num_cand = num_cand;
    bufs.q_nodes = d_q_nodes;
    bufs.q_dists = d_q_dists;
    bufs.q_size = d_q_size;
    bufs.topk_nodes = d_topk_nodes;
    bufs.topk_dists = d_topk_dists;
    bufs.visited_table = d_visited;
    bufs.visited_capacity = VISITED_CAPACITY;
    bufs.candidates_processed = d_processed;
    bufs.max_q_capacity = MAX_Q_CAPACITY;

    // Launch kernel: 1 block per query, choose a reasonable #threads (e.g., 128)
    dim3 blocks(1);
    dim3 threads(128);
    stage3_kernel<<<blocks, threads>>>(bufs);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Read back results: q_size, q content, topk, processed
    CHECK_CUDA(cudaMemcpy(&h_q_size, d_q_size, sizeof(int), cudaMemcpyDeviceToHost));
    std::cout << "q_size = " << h_q_size << std::endl;

    if (h_q_size > 0) {
        std::vector<int> h_q_nodes(h_q_size);
        std::vector<float> h_q_dists(h_q_size);
        CHECK_CUDA(cudaMemcpy(h_q_nodes.data(), d_q_nodes, h_q_size * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_q_dists.data(), d_q_dists, h_q_size * sizeof(float), cudaMemcpyDeviceToHost));
        std::cout << "Q contents:\n";
        for (int i = 0; i < h_q_size; ++i) {
            std::cout << "  q[" << i << "] = (" << h_q_nodes[i] << ", " << h_q_dists[i] << ")\n";
        }
    }

    std::vector<int> h_topk_nodes_out(TOPK_K);
    std::vector<float> h_topk_dists_out(TOPK_K);
    CHECK_CUDA(cudaMemcpy(h_topk_nodes_out.data(), d_topk_nodes, TOPK_K * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_topk_dists_out.data(), d_topk_dists, TOPK_K * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "TopK (pos: node dist):\n";
    for (int i = 0; i < TOPK_K; ++i) {
        std::cout << "  " << i << ": " << h_topk_nodes_out[i] << "  " << h_topk_dists_out[i] << std::endl;
    }

    CHECK_CUDA(cudaMemcpy(&h_processed, d_processed, sizeof(int), cudaMemcpyDeviceToHost));
    std::cout << "candidates_processed = " << h_processed << std::endl;

    // Free device memory
    cudaFree(d_candidates);
    cudaFree(d_distances);
    cudaFree(d_q_nodes);
    cudaFree(d_q_dists);
    cudaFree(d_q_size);
    cudaFree(d_topk_nodes);
    cudaFree(d_topk_dists);
    cudaFree(d_visited);
    cudaFree(d_processed);

    return 0;
}
