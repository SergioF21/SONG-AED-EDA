// Compilar: nvcc -std=c++14 -O3 stage3_final.cu -o stage3_final

#include "SongStage3.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <algorithm>
#include <iostream>


#define CHECK_CUDA(err) do { cudaError_t e = (err); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1);} } while(0)



// ---------------- HELPERS DEVICE ----------------

// Inserta id en tabla visited con linear probing y atomicCAS.
// Devuelve true si insertó (o ya presente), false si tabla llena.
__device__ bool visited_insert(int *visited_table, int table_size, int id) {
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
__device__ bool visited_contains(int *visited_table, int table_size, int id) {
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
__device__ bool visited_remove(int *visited_table, int table_size, int id) {
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

// ---------------- SHARED/DATA STRUCT (pointers to device memory) ----------------
// No colocamos estructuras complejas en shared memory aquí; el kernel usa punteros que apuntan
// a buffers ya asignados en device/global.



// ---------------- KERNEL PRINCIPAL (Stage 3) ----------------
// Se asume 1 block por consulta; thread 0 actúa como controlador secuencial.
__global__ void stage3_kernel(Stage3Buffers bufs) {
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int i = 0; i < bufs.num_cand; ++i) {
            int cand = bufs.candidates[i];
            float dist = bufs.distances[i];

            if (cand < 0) continue;

            // Selected insertion: compare con el peor (último) en topk (topk_dists sorted asc)
            float worst_dist = bufs.topk_dists[TOPK_K - 1];
            if (!(dist < worst_dist)) {
                // peor o igual -> ignorar (no marcar visited)
                continue;
            }

            // Encontrar posición de inserción
            int insert_pos = TOPK_K;
            for (int p = 0; p < TOPK_K; ++p) {
                if (dist < bufs.topk_dists[p]) { insert_pos = p; break; }
            }

            int evicted_node = -1;
            if (insert_pos < TOPK_K) {
                // Guardar el id que será expulsado (el último antes del shift)
                evicted_node = bufs.topk_nodes[TOPK_K - 1];

                // Shift para hacer espacio (de derecha a izquierda)
                for (int s = TOPK_K - 1; s > insert_pos; --s) {
                    bufs.topk_dists[s] = bufs.topk_dists[s - 1];
                    bufs.topk_nodes[s] = bufs.topk_nodes[s - 1];
                }
                // Insertar nuevo elemento en insert_pos
                bufs.topk_dists[insert_pos] = dist;
                bufs.topk_nodes[insert_pos] = cand;
            }

            // Insertar en Q (bounded)
            int q_idx = atomicAdd(bufs.q_size, 1);
            if (q_idx < bufs.max_q_capacity) {
                bufs.q_nodes[q_idx] = cand;
                bufs.q_dists[q_idx] = dist;
            } else {
                // revertir incremento y descartar candidato
                atomicAdd(bufs.q_size, -1);
                // opcional: podrías intentar insertar en topk aun si Q overflow; omitimos.
                // Si hicimos inserción en topk, y hubo eviction, revertir topk? dejamos como está.
                continue;
            }

            // Insertar en visited (si no estaba)
            bool vinserted = visited_insert(bufs.visited_table, bufs.visited_capacity, cand);
            (void)vinserted;

            // Si hubo evicted_node válido y distinto de -1, eliminarlo de visited.
            // Esto mantiene la invariante visited ⊆ (Q ∪ topK).
            if (evicted_node >= 0) {
                // intentamos remover; visited_remove usa atomicCAS así que es seguro.
                visited_remove(bufs.visited_table, bufs.visited_capacity, evicted_node);
            }

            // Incrementar contador de procesados
            atomicAdd(bufs.candidates_processed, 1);
        } // for candidates
    } // if thread 0

    __syncthreads();
}


void run_stage3_gpu(const char* trainFile, const char* testFile) {
    std::cout << "[Stage3] Loading " << trainFile << std::endl;
    std::cout << "[Stage3] Loading " << testFile << std::endl;

    // TODO: aquí llamas tus parsers y kernels.
    // Ejemplo:
    // load_dataset(trainFile);
    // load_testset(testFile);
    // launch_gpu_kernels();
    // write_results();

    std::cout << "[Stage3] GPU processing completed." << std::endl;
}