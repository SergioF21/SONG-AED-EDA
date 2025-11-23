#ifndef SONG_STAGE_1_CUDA_H
#define SONG_STAGE_1_CUDA_H

#include <iostream>
#include <vector>
#include <fstream>
#include <cuda_runtime.h>

// --- PARÁMETROS GPU ---
#define DEGREE_LIMIT 4
#define HASH_SIZE 32
#define QUEUE_SIZE 10
#define THREADS_PER_BLOCK 256

// --- MANEJO DE ERRORES CUDA ---
#define CUDA_CHECK(call)                                                     \
    do                                                                       \
    {                                                                        \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess)                                              \
        {                                                                    \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)           \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// --- ESTRUCTURAS DE DATOS ---

// Memoria compartida dentro del kernel
struct SharedMemoryGPU
{
    int cola_q[QUEUE_SIZE];
    int tamano_cola;
    int tabla_visited[HASH_SIZE];
    int lista_candidatos[DEGREE_LIMIT];
    int conteo_candidatos;
};

// Variables en device memory
static int *d_grafo = nullptr;
static int num_nodos_gpu = 0;

// --- FUNCIONES HOST ---

// Cargar grafo a GPU
inline void cargarGrafoGPU(const std::string &filename)
{
    std::cout << "[GPU Stage 1] Cargando grafo a GPU: " << filename << std::endl;

    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "[ERROR] No se encontró " << filename << std::endl;
        exit(1);
    }

    int n_points, k;
    file.read(reinterpret_cast<char *>(&n_points), sizeof(int));
    file.read(reinterpret_cast<char *>(&k), sizeof(int));

    std::cout << "    -> Nodos: " << n_points << ", Grado: " << k << std::endl;

    if (k != DEGREE_LIMIT)
    {
        std::cerr << "[ERROR] Grado incompatible" << std::endl;
        exit(1);
    }

    num_nodos_gpu = n_points;
    int total_elements = n_points * k;

    // Leer datos del grafo
    std::vector<int> grafo_host(total_elements);
    file.read(reinterpret_cast<char *>(grafo_host.data()), total_elements * sizeof(int));
    file.close();

    // Copiar a GPU
    CUDA_CHECK(cudaMalloc(&d_grafo, total_elements * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_grafo, grafo_host.data(),
                          total_elements * sizeof(int),
                          cudaMemcpyHostToDevice));

    std::cout << "[GPU Stage 1] Grafo cargado en GPU Global Memory ("
              << (total_elements * sizeof(int)) / 1024.0 << " KB)" << std::endl;
}

// Liberar memoria GPU
inline void liberarGrafoGPU()
{
    if (d_grafo)
    {
        CUDA_CHECK(cudaFree(d_grafo));
        d_grafo = nullptr;
    }
}

// --- KERNELS CUDA ---

// Kernel: Etapa 1 - Localización de Candidatos
__global__ void kernel_localizacion_candidatos(
    int *grafo,
    int num_nodos,
    int nodo_inicio,
    int *output_candidatos,
    int *output_count)
{
    // Allocate shared memory
    __shared__ SharedMemoryGPU shared;

    // Solo el primer hilo inicializa
    if (threadIdx.x == 0)
    {
        // Inicializar estructuras
        for (int i = 0; i < HASH_SIZE; i++)
        {
            shared.tabla_visited[i] = 0;
        }
        for (int i = 0; i < DEGREE_LIMIT; i++)
        {
            shared.lista_candidatos[i] = 0;
        }

        // Setup inicial
        shared.cola_q[0] = nodo_inicio;
        shared.tamano_cola = 1;
        shared.conteo_candidatos = 0;

        // Marcar nodo inicial como visitado
        shared.tabla_visited[nodo_inicio % HASH_SIZE] = 1;
    }

    __syncthreads();

    // PASO 1: Thread 0 extrae nodo de la cola
    int now_idx = -1;
    if (threadIdx.x == 0)
    {
        if (shared.tamano_cola > 0)
        {
            now_idx = shared.cola_q[0];
            shared.tamano_cola--;
        }
    }

    __syncthreads();

    if (now_idx == -1)
        return;

    // PASO 2: Acceso paralelo a vecinos (cada thread procesa un vecino)
    if (threadIdx.x < DEGREE_LIMIT)
    {
        int index_memoria = now_idx * DEGREE_LIMIT + threadIdx.x;

        if (index_memoria < num_nodos * DEGREE_LIMIT)
        {
            int vecino_id = grafo[index_memoria];

            // PASO 3: Verificar si ya fue visitado
            int hash_idx = vecino_id % HASH_SIZE;

            // Acceso atómico para evitar race conditions
            int ya_visitado = atomicCAS(&shared.tabla_visited[hash_idx], 0, 1);

            // PASO 4: Si no estaba visitado, agregarlo a candidatos
            if (ya_visitado == 0)
            {
                int pos = atomicAdd(&shared.conteo_candidatos, 1);
                if (pos < DEGREE_LIMIT)
                {
                    shared.lista_candidatos[pos] = vecino_id;
                }
            }
        }
    }

    __syncthreads();

    // PASO 5: Thread 0 copia resultados a memoria global
    if (threadIdx.x == 0)
    {
        *output_count = shared.conteo_candidatos;
        for (int i = 0; i < shared.conteo_candidatos; i++)
        {
            output_candidatos[i] = shared.lista_candidatos[i];
        }
    }
}

// --- FUNCIÓN PRINCIPAL GPU ---

inline std::vector<int> runSongSimulationGPU()
{
    std::cout << "\n=== EJECUTANDO STAGE 1 EN GPU ===" << std::endl;

    // 1. Cargar grafo a GPU
    cargarGrafoGPU("graph_index.bin");

    // 2. Preparar memoria para resultados
    int *d_candidatos;
    int *d_count;
    CUDA_CHECK(cudaMalloc(&d_candidatos, DEGREE_LIMIT * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));

    // 3. Configurar ejecución del kernel
    int nodo_inicio = 0;
    dim3 blockDim(THREADS_PER_BLOCK);
    dim3 gridDim(1); // Un solo bloque para esta simulación

    std::cout << "[GPU] Lanzando kernel con " << THREADS_PER_BLOCK << " threads..." << std::endl;

    // 4. Ejecutar kernel
    kernel_localizacion_candidatos<<<gridDim, blockDim>>>(
        d_grafo,
        num_nodos_gpu,
        nodo_inicio,
        d_candidatos,
        d_count);

    // Sincronizar y verificar errores
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    std::cout << "[GPU] Kernel ejecutado exitosamente" << std::endl;

    // 5. Copiar resultados de vuelta al host
    int count_host;
    std::vector<int> candidatos_host(DEGREE_LIMIT);

    CUDA_CHECK(cudaMemcpy(&count_host, d_count, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(candidatos_host.data(), d_candidatos,
                          DEGREE_LIMIT * sizeof(int), cudaMemcpyDeviceToHost));

    // 6. Preparar vector de salida
    std::vector<int> candidatos_encontrados;
    for (int i = 0; i < count_host && i < DEGREE_LIMIT; i++)
    {
        candidatos_encontrados.push_back(candidatos_host[i]);
    }

    // 7. Mostrar resultados
    std::cout << "\n=== RESULTADO FINAL GPU STAGE 1 ===" << std::endl;
    std::cout << "Candidatos encontrados: [ ";
    for (int candidato : candidatos_encontrados)
    {
        std::cout << candidato << " ";
    }
    std::cout << "]" << std::endl;
    std::cout << "Total: " << candidatos_encontrados.size() << " candidatos" << std::endl;

    // 8. Liberar memoria
    CUDA_CHECK(cudaFree(d_candidatos));
    CUDA_CHECK(cudaFree(d_count));
    liberarGrafoGPU();

    return candidatos_encontrados;
}

// Alias para mantener compatibilidad con el main
inline std::vector<int> runSongSimulation()
{
    return runSongSimulationGPU();
}

#endif // SONG_STAGE_1_CUDA_H