/*
ETAPA 3:
1. Sincronizar hilos (__syncthreads())
2. El mismo hijo de la etapa 1 toma el control de nuevo
3. Lee los resultados del array dist.
4. Para cada candidato y su distancia, actualiza las estructuras en memoria compartida: q (la cola de visita), topk (los resultados) y visited (el conjunto de invitaods). 
5. Aqui es donde se aplican las optimizaciones de memoria del siguiente paso:
    - Bounded Priority queue
    - Selected Insertion
    - Visited Delition
*/

#include "step3.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <limits.h>
#include <vector>
#include <algorithm>    
#include "graph.h"
#include "structs/sharedMemory.h"
#include "utils.h"
#include <iostream>
#include <thread>
#include <unordered_set>
#include <chrono>
#include <unordered_map>
#include <device_launch_parameters.h>




__global__
void step3_kernel(sharedMemory smem) {
    // Hilo que ejecuta la etapa 3
    int idx = threadIdx.x;
    if (idx == 0) {
        // Leer los resultados del array dist
        for (int i = 0; i < smem.q_size[0]; i++) {
            int candidate = smem.q[i][0];
            int distance = smem.q[i][1];

            // Actualizar la cola de visita (q)
            int q_index = atomicAddWrapper(smem.q_size, 1);
            smem.q[q_index][0] = candidate;
            smem.q[q_index][1] = distance;

            // Actualizar los resultados top-k
            for (int j = 0; j < DEGREE_LIMIT; j++) {
                if (distance < smem.topk[j][0]) {
                    // Desplazar los elementos mayores
                    for (int k = DEGREE_LIMIT - 1; k > j; k--) {
                        smem.topk[k][0] = smem.topk[k - 1][0];
                        smem.topk[k][1] = smem.topk[k - 1][1];
                    }
                    // Insertar el nuevo elemento
                    smem.topk[j][0] = candidate;
                    smem.topk[j][1] = distance;
                    break;
                }
            }

            // Actualizar el conjunto de visited
            smem.visited[candidate] = true;

            // Incrementar el contador de candidatos procesados
            atomicAddWrapper(smem.candidates_processed, 1);
        }
    }
};

int main() {
    // Ejemplo de uso del kernel step3_kernel
    const int num_candidates = 5;
    int h_dist[num_candidates] = {10, 20, 5, 15, 25};
    int h_candidates[num_candidates] = {1, 2, 3, 4, 5};

    int *d_dist, *d_candidates;
    int *d_q, *d_q_size;
    int *d_topk;
    bool *d_visited;

    int h_q_size = 0;
    const int k = 3;
    const int max_nodes = 100;

    cudaMalloc((void**)&d_dist, num_candidates * sizeof(int));
    cudaMalloc((void**)&d_candidates, num_candidates * sizeof(int));
    cudaMalloc((void**)&d_q, max_nodes * sizeof(int));
    cudaMalloc((void**)&d_q_size, sizeof(int));
    cudaMalloc((void**)&d_topk, k * sizeof(int));
    cudaMalloc((void**)&d_visited, max_nodes * sizeof(bool));

    cudaMemcpy(d_dist, h_dist, num_candidates * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_candidates, h_candidates, num_candidates * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_q_size, &h_q_size, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_visited, 0, max_nodes * sizeof(bool));

    sharedMemory smem;
    smem.q[0] = d_q;
    smem.q_size = d_q_size;
    smem.topk[0] = d_topk;
    smem.visited = d_visited;
    int h_candidates_processed = 0;
    cudaMalloc((void**)&smem.candidates_processed, sizeof(int));
    cudaMemcpy(smem.candidates_processed, &h_candidates_processed, sizeof(int), cudaMemcpyHostToDevice);
    // Lanzar el kernel

    step3_kernel(smem);

    cudaDeviceSynchronize();

    // Liberar memoria
    cudaFree(d_dist);
    cudaFree(d_candidates);
    cudaFree(d_q);
    cudaFree(d_q_size);
    cudaFree(d_topk);
    cudaFree(d_visited);
    cudaFree(smem.candidates_processed);
    return 0;
}

