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
#define TOPK_DEFAULT 8

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


// ---------------- HELPERS DEVICE ----------------

// Inserta id en tabla visited con linear probing y atomicCAS.
// Devuelve true si insertó (o ya presente), false si tabla llena.
__device__ bool visited_insert(int32_t *visited_table, int table_size, int id) {
    if (id < 0) return false;
    unsigned int h = (unsigned int)id;
    unsigned int idx = h & (table_size - 1); // table_size debe ser potencia de 2 para esto
    for (int probe = 0; probe < table_size; ++probe) {
        int cur = atomicCAS(&visited_table[idx], -1, id);
        if (cur == -1) {
            // inserted (slot was empty)
            return true;
        }
        if (cur == id) {
            // already present
            return true;
        }
        // else continue probing
        idx = (idx + 1) & (table_size - 1);
    }
    // table full / can't insert
    return false;
}

// Comprueba si id está en visited (no usa atomics: lectura eventual consistente)
__device__ bool visited_contains(int32_t *visited_table, int table_size, int id) {
    if (id < 0) return false;
    unsigned int h = (unsigned int)id;
    unsigned int idx = h & (table_size - 1);
    for (int probe = 0; probe < table_size; ++probe) {
        int cur = visited_table[idx];
        if (cur == -1) return false; // empty -> not found
        if (cur == id) return true;
        idx = (idx + 1) & (table_size - 1);
    }
    return false;
}

// Elimina id de visited (marca tombstone = -2). Devuelve true si eliminado.
__device__ bool visited_remove(int32_t *visited_table, int table_size, int id) {
    if (id < 0) return false;
    unsigned int h = (unsigned int)id;
    unsigned int idx = h & (table_size - 1);
    for (int probe = 0; probe < table_size; ++probe) {
        int cur = visited_table[idx];
        if (cur == -1) return false; // not found
        if (cur == id) {
            // attempt to replace id -> tombstone (-2)
            int prev = atomicCAS(&visited_table[idx], id, -2);
            return (prev == id);
        }
        idx = (idx + 1) & (table_size - 1);
    }
    return false;
}

// ---------------- KERNEL ----------------

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
        visited_insert(s_visited, VISITED_SIZE, start_node);
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
                    if (!visited_contains(s_visited, VISITED_SIZE, v) && ccount < graph_degree) {
                        s_candidates[ccount] = v;
                        s_cand_dist[ccount] = 1e30f;
                        visited_insert(s_visited, VISITED_SIZE, v);
                        ccount++;
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

         // ---------------- Stage 3: Data Structure Maintenance  ----------------
        if (tid == 0) {
            for (int i = 0; i < candidate_count; ++i) {
                int cand = s_candidates[i];
                float dist = s_cand_dist[i];

                if (cand < 0) continue;

                // Selected insertion: compare con el peor (último) en topk (topk_dists sorted asc)
                float worst_dist = s_topk_dists[TOPK - 1];
                if (!(dist < worst_dist)) {
                    // peor o igual -> ignorar (no marcar visited)
                    continue;
                }

                // Encontrar posición de inserción
                int insert_pos = TOPK;
                for (int p = 0; p < TOPK; ++p) {
                    if (dist < s_topk_dists[p]) { insert_pos = p; break; }
                }

                int evicted_node = -1;
                if (insert_pos < TOPK) {
                    // Guardar el id que será expulsado (el último antes del shift)
                    evicted_node = s_topk_ids[TOPK - 1];

                    // Shift para hacer espacio (de derecha a izquierda)
                    for (int s = TOPK - 1; s > insert_pos; --s) {
                        s_topk_dists[s] = s_topk_dists[s - 1];
                        s_topk_ids[s] = s_topk_ids[s - 1];
                    }
                    // Insertar nuevo elemento en insert_pos
                    s_topk_dists[insert_pos] = dist;
                    s_topk_ids[insert_pos] = cand;
                    if (*s_topk_size < TOPK) {
                        (*s_topk_size)++;
                    }
                }

                // Insertar en Q (bounded)
                int q_idx = atomicAdd(s_q_size, 1);
                if (q_idx < Q_CAP) {
                    s_q_ids[q_idx] = cand;
                    s_q_dists[q_idx] = dist;
                } else {
                    // revertir incremento y descartar candidato
                    atomicAdd(s_q_size, -1);
                    // opcional: podrías intentar insertar en topk aun si Q overflow; omitimos.
                    // Si hicimos inserción en topk, y hubo eviction, revertir topk? dejamos como está.
                    continue;
                }
/*
                // Insertar en visited (si no estaba)
                bool vinserted = visited_insert(s_visited, VISITED_SIZE, cand);
                (void)vinserted;
*/
                // Si hubo evicted_node válido y distinto de -1, eliminarlo de visited.
                // Esto mantiene la invariante visited ⊆ (Q ∪ topK).
                if (evicted_node >= 0) {
                    // intentamos remover; visited_remove usa atomicCAS así que es seguro.
                    visited_remove(s_visited, VISITED_SIZE, evicted_node);
                }

                // Incrementar contador de procesados
                //atomicAdd(s_candidate_count, 1);
            } // for candidates
        } // if thread 0
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

    printf("Dataset: N=%d, D=%d\n", dataN, dataD);
    printf("Graph: N=%d, DEGREE=%d\n", graphN, graphK);

    int graph_degree = graphK;
    int dim = dataD;
    int TOPK = TOPK_DEFAULT;
    int VISITED_SIZE = 4096;
    int Q_CAP = 4 * TOPK;

    int vsz = 1;
    while (vsz < 4*TOPK) vsz <<= 1;
    VISITED_SIZE = vsz;


    std::cout << "Vecinos de start_node " << start_node << ": ";
    for(int i = 0; i < graphK; ++i) {
        std::cout << h_graph[start_node*graphK + i] << " ";
    }
    std::cout << "\n";

    int Q = num_queries;
    std::vector<int32_t> h_starts(Q);
    for (int i = 0; i < Q; ++i) h_starts[i] = (start_node + i < dataN) ? (start_node + i) : start_node;

    std::cout << "Vecinos de start_node " << start_node << ": ";
    for(int i = 0; i < graphK; ++i) {
        std::cout << h_graph[start_node*graphK + i] << " ";
    }
    std::cout << "\n";


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

    // launch kernel
    printf("Launching kernel: blocks=%d threads=%d shared=%.2f KB\n", Q, THREADS_PER_BLOCK, shared_size / 1024.0f);

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