#ifndef SONG_STAGE3_H
#define SONG_STAGE3_H

// --------------------------------------------------------
// CONFIGURACIÓN (fuera de extern "C")
// --------------------------------------------------------
constexpr int MAX_Q_CAPACITY = 1024;
constexpr int = 8;
constexpr int VISITED_CAPACITY = 4096;

// --------------------------------------------------------
// Struct POD para los buffers
// --------------------------------------------------------
typedef struct Stage3Buffers {
    const int*   candidates;    // [num_cand]
    const float* distances;     // [num_cand]
    int    num_cand;

    int*   q_nodes;     // [max_q_capacity]
    float* q_dists;     // [max_q_capacity]
    int*   q_size;      // pointer to single int in device memory

    int*   topk_nodes;  // [TOPK_K]
    float* topk_dists;  // [TOPK_K]

    int*   visited_table;   // [visited_capacity] (int) -1 empty, -2 tombstone
    int    visited_capacity;

    int*   candidates_processed;    // pointer to single int

    int    max_q_capacity;
} Stage3Buffers;


// --------------------------------------------------------
// Declaraciones host (pueden ir con extern "C")
// --------------------------------------------------------
#ifdef __cplusplus
extern "C" {
#endif

void run_stage3_gpu(const char* trainFile, const char* testFile);

#ifdef __cplusplus
}
#endif


// --------------------------------------------------------
// Declaraciones CUDA (__global__ y __device__)
// NO usar extern "C"
// --------------------------------------------------------
__global__ void stage3_kernel(Stage3Buffers buffers);

__device__ bool visited_insert(int* table, int table_size, int id);
__device__ bool visited_contains(int* table, int table_size, int id);
__device__ bool visited_remove(int* table, int table_size, int id);

#endif // SONG_STAGE3_H
