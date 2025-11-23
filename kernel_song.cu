#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cassert>

#define DEGREE_LIMIT 4
#define HASH_SIZE 32
#define QUEUE_SIZE 64
#define THREADS_PER_BLOCK 128
#define WARP_SIZE 32
#define TOPK_DEFAULT 10

#define CUDA_CHECK(call) do { cudaError_t err = call; if (err != cudaSuccess) { fprintf(stderr,"CUDA ERROR %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(1);} } while(0)

static inline __device__ float warp_reduce_sum(float v) {
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffff, v, offset);
    return v;
}

// host helpers to read binary files
bool read_graph_bin(const char* path, std::vector<int32_t>& out_graph, int &N, int &K) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    f.read(reinterpret_cast<char*>(&N), sizeof(int));
    f.read(reinterpret_cast<char*>(&K), sizeof(int));
    out_graph.resize((size_t)N * K);
    f.read(reinterpret_cast<char*>(out_graph.data()), out_graph.size() * sizeof(int32_t));
    f.close();
    return true;
}
bool read_dataset_bin(const char* path, std::vector<float>& out_data, int &N, int &D) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    f.read(reinterpret_cast<char*>(&N), sizeof(int));
    f.read(reinterpret_cast<char*>(&D), sizeof(int));
    out_data.resize((size_t)N * D);
    f.read(reinterpret_cast<char*>(out_data.data()), out_data.size() * sizeof(float));
    f.close();
    return true;
}

extern "C"
__global__ void song_mega_kernel(
    const int32_t* __restrict__ d_graph,
    const float*   __restrict__ d_data,
    const int32_t* __restrict__ d_starts,
    int N,
    int graph_degree,
    int dim,
    int max_iters,
    int TOPK,
    int VISITED_SIZE,
    int Q_CAP,
    int32_t* __restrict__ d_out_ids,   // Q * TOPK
    float*   __restrict__ d_out_dists  // Q * TOPK
) {
    const int bidx = blockIdx.x;
    const int tid  = threadIdx.x;
    const int nthreads = blockDim.x;
    const int lane = tid & (WARP_SIZE - 1);
    const int numWarps = nthreads / WARP_SIZE;
    const int warpId = tid / WARP_SIZE;

    extern __shared__ char smem_raw[];
    size_t offset = 0;

    float *s_query = (float*)(smem_raw + offset); offset += sizeof(float) * dim;
    int32_t *s_candidates = (int32_t*)(smem_raw + offset); offset += sizeof(int32_t) * graph_degree;
    float *s_cand_dist = (float*)(smem_raw + offset); offset += sizeof(float) * graph_degree;

    int32_t *s_q_ids = (int32_t*)(smem_raw + offset); offset += sizeof(int32_t) * Q_CAP;
    float *s_q_dists = (float*)(smem_raw + offset); offset += sizeof(float) * Q_CAP;
    int *s_q_size = (int*)(smem_raw + offset); offset += sizeof(int) * 1;

    int32_t *s_topk_ids = (int32_t*)(smem_raw + offset); offset += sizeof(int32_t) * TOPK;
    float *s_topk_dists = (float*)(smem_raw + offset); offset += sizeof(float) * TOPK;
    int *s_topk_size = (int*)(smem_raw + offset); offset += sizeof(int) * 1;

    int32_t *s_visited = (int32_t*)(smem_raw + offset); offset += sizeof(int32_t) * VISITED_SIZE;

    int *s_candidate_count = (int*)(smem_raw + offset); offset += sizeof(int) * 1;
    int *s_now_idx = (int*)(smem_raw + offset); offset += sizeof(int) * 1;

    int start_node = d_starts[bidx];

    for (int i = tid; i < dim; i += nthreads) {
        s_query[i] = d_data[(size_t)start_node * dim + i];
    }

    if (tid == 0) {
        *s_candidate_count = 0;
        *s_q_size = 0;
        *s_topk_size = 0;
        *s_now_idx = -1;
        for (int i=0;i<graph_degree;i++) { s_candidates[i] = -1; s_cand_dist[i] = 1e30f; }
        for (int i=0;i<Q_CAP;i++) { s_q_ids[i] = -1; s_q_dists[i] = 1e30f; }
        for (int i=0;i<TOPK;i++) { s_topk_ids[i] = -1; s_topk_dists[i] = 1e30f; }
        for (int i=0;i<VISITED_SIZE;i++) s_visited[i] = -1;
        s_q_ids[0] = start_node; s_q_dists[0] = 1e30f; *s_q_size = 1;
        int h = (start_node * 2654435761u) & (VISITED_SIZE - 1);
        int p = 0;
        while (p < VISITED_SIZE) {
            int idx = (h + p) & (VISITED_SIZE - 1);
            int32_t old = atomicCAS(&s_visited[idx], -1, start_node);
            if (old == -1 || old == start_node) break;
            p++;
        }
    }
    __syncthreads();

    for (int iter = 0; iter < max_iters; ++iter) {
        if (tid == 0) {
            int qsz = *s_q_size;
            if (qsz == 0) {
                *s_candidate_count = 0;
                *s_now_idx = -1;
            } else {
                int best_pos = 0;
                float bestd = s_q_dists[0];
                for (int i=1;i<qsz;i++) {
                    if (s_q_dists[i] < bestd) { bestd = s_q_dists[i]; best_pos = i; }
                }
                int now = s_q_ids[best_pos];
                s_q_ids[best_pos] = s_q_ids[qsz - 1];
                s_q_dists[best_pos] = s_q_dists[qsz - 1];
                s_q_ids[qsz - 1] = -1;
                s_q_dists[qsz - 1] = 1e30f;
                (*s_q_size)--;
                *s_now_idx = now;
                int base = now * graph_degree;
                int ccount = 0;
                for (int j=0;j<graph_degree;j++) {
                    int32_t v = d_graph[base + j];
                    if (v < 0) continue;
                    int h = (v * 2654435761u) & (VISITED_SIZE - 1);
                    bool seen = false;
                    int probe = 0;
                    while (probe < VISITED_SIZE) {
                        int idx = (h + probe) & (VISITED_SIZE - 1);
                        int32_t val = s_visited[idx];
                        if (val == -1) break;
                        if (val == v) { seen = true; break; }
                        probe++;
                    }
                    if (!seen && ccount < graph_degree) {
                        s_candidates[ccount] = v;
                        s_cand_dist[ccount] = 1e30f;
                        ccount++;
                        probe = 0;
                        while (probe < VISITED_SIZE) {
                            int idx = (h + probe) & (VISITED_SIZE - 1);
                            int32_t old = atomicCAS(&s_visited[idx], -1, v);
                            if (old == -1 || old == v) break;
                            probe++;
                        }
                    }
                }
                *s_candidate_count = ccount;
            }
        }
        __syncthreads();

        int candidate_count = *s_candidate_count;
        if (candidate_count <= 0) break;

        for (int c = warpId; c < candidate_count; c += numWarps) {
            int vid = s_candidates[c];
            const float* vec = d_data + (size_t)vid * dim;
            float partial = 0.0f;
            for (int d = lane; d < dim; d += WARP_SIZE) {
                float diff = vec[d] - s_query[d];
                partial += diff * diff;
            }
            float sum = warp_reduce_sum(partial);
            if (lane == 0) {
                s_cand_dist[c] = sum;
            }
        }
        __syncthreads();

        if (tid == 0) {
            int now = *s_now_idx;
            for (int c = 0; c < candidate_count; ++c) {
                int vid = s_candidates[c];
                float dval = s_cand_dist[c];
                bool skip = false;
                if (*s_topk_size == TOPK) {
                    float worst = -1.0f; int worst_pos = -1;
                    for (int t=0;t<TOPK;t++) {
                        if (s_topk_ids[t] == -1) continue;
                        if (s_topk_dists[t] > worst) { worst = s_topk_dists[t]; worst_pos = t; }
                    }
                    if (worst_pos >= 0 && dval >= worst) skip = true;
                }
                if (skip) continue;
                int qsz = *s_q_size;
                if (qsz < Q_CAP) {
                    s_q_ids[qsz] = vid;
                    s_q_dists[qsz] = dval;
                    (*s_q_size)++;
                } else {
                    float worstq = -1.0f; int worstqpos = -1;
                    for (int i=0;i<Q_CAP;i++) {
                        if (s_q_ids[i] == -1) continue;
                        if (s_q_dists[i] > worstq) { worstq = s_q_dists[i]; worstqpos = i; }
                    }
                    if (worstqpos >= 0 && dval < worstq) {
                        int ev = s_q_ids[worstqpos];
                        s_q_ids[worstqpos] = vid;
                        s_q_dists[worstqpos] = dval;
                        if (ev >= 0) {
                            int h = (ev * 2654435761u) & (VISITED_SIZE - 1);
                            int probe = 0;
                            while (probe < VISITED_SIZE) {
                                int idx = (h + probe) & (VISITED_SIZE - 1);
                                int32_t val = s_visited[idx];
                                if (val == ev) { s_visited[idx] = -1; break; }
                                if (val == -1) break;
                                probe++;
                            }
                        }
                    } else {
                        continue;
                    }
                }
                if (*s_topk_size < TOPK) {
                    int pos = *s_topk_size;
                    s_topk_ids[pos] = vid;
                    s_topk_dists[pos] = dval;
                    (*s_topk_size)++;
                } else {
                    float worst = -1.0f; int worst_pos = -1;
                    for (int t=0;t<TOPK;t++) {
                        if (s_topk_dists[t] > worst) { worst = s_topk_dists[t]; worst_pos = t; }
                    }
                    if (worst_pos >= 0 && dval < worst) {
                        int ev = s_topk_ids[worst_pos];
                        s_topk_ids[worst_pos] = vid;
                        s_topk_dists[worst_pos] = dval;
                        int h = (ev * 2654435761u) & (VISITED_SIZE - 1);
                        int probe = 0;
                        while (probe < VISITED_SIZE) {
                            int idx = (h + probe) & (VISITED_SIZE - 1);
                            int32_t val = s_visited[idx];
                            if (val == ev) { s_visited[idx] = -1; break; }
                            if (val == -1) break;
                            probe++;
                        }
                    }
                }
            }
            if (now >= 0) {
                bool found = false;
                for (int t=0;t<*s_topk_size;t++) if (s_topk_ids[t] == now) { found = true; break; }
                if (!found) {
                    int h = (now * 2654435761u) & (VISITED_SIZE - 1);
                    int probe = 0;
                    while (probe < VISITED_SIZE) {
                        int idx = (h + probe) & (VISITED_SIZE - 1);
                        int32_t val = s_visited[idx];
                        if (val == now) { s_visited[idx] = -1; break; }
                        if (val == -1) break;
                        probe++;
                    }
                }
            }
        }
        __syncthreads();

        if (tid == 0) {
            if (*s_q_size == 0) *s_candidate_count = 0;
        }
        __syncthreads();
        if (*s_candidate_count == 0 && *s_q_size == 0) break;
    }

    if (tid == 0) {
        int tsize = *s_topk_size;
        for (int i=0;i<tsize;i++) {
            int minpos = i;
            for (int j=i+1;j<tsize;j++) if (s_topk_dists[j] < s_topk_dists[minpos]) minpos = j;
            float td = s_topk_dists[i]; s_topk_dists[i] = s_topk_dists[minpos]; s_topk_dists[minpos] = td;
            int ti = s_topk_ids[i]; s_topk_ids[i] = s_topk_ids[minpos]; s_topk_ids[minpos] = ti;
        }
        for (int i=0;i<TOPK;i++) {
            int out_idx = bidx * TOPK + i;
            if (i < tsize) {
                d_out_ids[out_idx] = s_topk_ids[i];
                d_out_dists[out_idx] = s_topk_dists[i];
            } else {
                d_out_ids[out_idx] = -1;
                d_out_dists[out_idx] = 1e30f;
            }
        }
    }
}

// host runner (main)
int main(int argc, char** argv) {
    if (argc < 5) {
        printf("Usage: %s dataset.bin graph_index.bin <start_node> <num_queries>\n", argv[0]);
        return 1;
    }
    const char* dataset_path = argv[1];
    const char* graph_path = argv[2];
    int start_node = atoi(argv[3]);
    int num_queries = atoi(argv[4]);
    if (num_queries < 1) num_queries = 1;

    std::vector<float> h_data;
    int dataN, dataD;
    if (!read_dataset_bin(dataset_path, h_data, dataN, dataD)) {
        fprintf(stderr, "Failed to read dataset: %s\n", dataset_path); return 1;
    }
    std::vector<int32_t> h_graph;
    int graphN, graphK;
    if (!read_graph_bin(graph_path, h_graph, graphN, graphK)) {
        fprintf(stderr, "Failed to read graph: %s\n", graph_path); return 1;
    }

    int graph_degree = graphK;
    int dim = dataD;
    int TOPK = TOPK_DEFAULT;
    int VISITED_SIZE = 64;
    int Q_CAP = 4 * TOPK;

    int vsz = 1;
    while (vsz < 4*TOPK) vsz <<= 1;
    VISITED_SIZE = vsz;

    int Q = num_queries;
    std::vector<int32_t> h_starts(Q);
    for (int i = 0; i < Q; ++i) h_starts[i] = (start_node + i < dataN) ? (start_node + i) : start_node;

    int32_t *d_graph = nullptr; float *d_data = nullptr; int32_t *d_starts = nullptr;
    int32_t *d_out_ids = nullptr; float *d_out_dists = nullptr;

    size_t graph_bytes = (size_t)graphN * graph_degree * sizeof(int32_t);
    size_t data_bytes = (size_t)dataN * dim * sizeof(float);
    size_t starts_bytes = (size_t)Q * sizeof(int32_t);
    size_t out_ids_bytes = (size_t)Q * TOPK * sizeof(int32_t);
    size_t out_dists_bytes = (size_t)Q * TOPK * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_graph, graph_bytes));
    CUDA_CHECK(cudaMalloc(&d_data, data_bytes));
    CUDA_CHECK(cudaMalloc(&d_starts, starts_bytes));
    CUDA_CHECK(cudaMalloc(&d_out_ids, out_ids_bytes));
    CUDA_CHECK(cudaMalloc(&d_out_dists, out_dists_bytes));

    CUDA_CHECK(cudaMemcpy(d_graph, h_graph.data(), graph_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), data_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_starts, h_starts.data(), starts_bytes, cudaMemcpyHostToDevice));

    size_t shared_size = 0;
    shared_size += sizeof(float) * dim;
    shared_size += sizeof(int32_t) * graph_degree;
    shared_size += sizeof(float) * graph_degree;
    shared_size += sizeof(int32_t) * Q_CAP;
    shared_size += sizeof(float) * Q_CAP;
    shared_size += sizeof(int) * 1;
    shared_size += sizeof(int32_t) * TOPK;
    shared_size += sizeof(float) * TOPK;
    shared_size += sizeof(int) * 1;
    shared_size += sizeof(int32_t) * VISITED_SIZE;
    shared_size += sizeof(int) * 1;
    shared_size += sizeof(int) * 1;

    dim3 grid(Q); dim3 block(THREADS_PER_BLOCK);
    song_mega_kernel<<<grid, block, shared_size>>>(
        d_graph, d_data, d_starts,
        dataN, graph_degree, dim,
        2048,
        TOPK, VISITED_SIZE, Q_CAP,
        d_out_ids, d_out_dists
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int32_t> h_out_ids((size_t)Q * TOPK);
    std::vector<float> h_out_dists((size_t)Q * TOPK);
    CUDA_CHECK(cudaMemcpy(h_out_ids.data(), d_out_ids, out_ids_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_out_dists.data(), d_out_dists, out_dists_bytes, cudaMemcpyDeviceToHost));

    for (int q = 0; q < Q; ++q) {
        printf("Query %d Top-%d results:\n", q, TOPK);
        for (int k = 0; k < TOPK; ++k) {
            int idx = h_out_ids[q * TOPK + k];
            float dist = h_out_dists[q * TOPK + k];
            if (idx < 0) {
                printf("  [%d] -\n", k);
            } else {
                printf("  [%d] id=%d dist_sq=%.6f\n", k, idx, dist);
            }
        }
    }

    cudaFree(d_graph); cudaFree(d_data); cudaFree(d_starts); cudaFree(d_out_ids); cudaFree(d_out_dists);
    return 0;
}