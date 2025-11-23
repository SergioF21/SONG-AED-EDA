#ifndef SONG_STAGE_2_CUDA_H
#define SONG_STAGE_2_CUDA_H

#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <cuda_runtime.h>

// --- PARÁMETROS GPU ---
#define WARP_SIZE 32
#define THREADS_PER_BLOCK 256
#define MAX_CANDIDATES 16

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

// --- VARIABLES GLOBALES GPU ---
static float *d_dataset = nullptr;
static float *d_query_point = nullptr;
static int dataset_n_points = 0;
static int dataset_dim = 0;

// --- FUNCIONES HOST ---

// Cargar dataset a GPU
inline void cargarDatasetGPU(const std::string &filename)
{
    std::cout << "\n[GPU Stage 2] Cargando dataset a GPU: " << filename << std::endl;

    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "[ERROR] No se pudo abrir " << filename << std::endl;
        exit(1);
    }

    // Leer header
    file.read(reinterpret_cast<char *>(&dataset_n_points), sizeof(int));
    file.read(reinterpret_cast<char *>(&dataset_dim), sizeof(int));

    std::cout << "[GPU Stage 2] Dataset info:" << std::endl;
    std::cout << "        -> Puntos: " << dataset_n_points << std::endl;
    std::cout << "        -> Dimensiones: " << dataset_dim << std::endl;

    // Leer datos
    size_t total_elements = dataset_n_points * dataset_dim;
    std::vector<float> data_host(total_elements);
    file.read(reinterpret_cast<char *>(data_host.data()),
              total_elements * sizeof(float));
    file.close();

    // Copiar a GPU
    CUDA_CHECK(cudaMalloc(&d_dataset, total_elements * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_dataset, data_host.data(),
                          total_elements * sizeof(float),
                          cudaMemcpyHostToDevice));

    std::cout << "[GPU Stage 2] Dataset cargado en GPU Global Memory ("
              << (total_elements * sizeof(float)) / (1024.0 * 1024.0) << " MB)" << std::endl;
}

// Inicializar query point en GPU
inline void inicializarQueryPointGPU(int query_id)
{
    if (query_id >= dataset_n_points)
    {
        std::cerr << "[ERROR] Query ID fuera de rango" << std::endl;
        exit(1);
    }

    std::cout << "[GPU Stage 2] Copiando Query Point " << query_id
              << " a GPU Shared Memory" << std::endl;

    // Allocar memoria para query point
    CUDA_CHECK(cudaMalloc(&d_query_point, dataset_dim * sizeof(float)));

    // Copiar query point específico desde dataset
    size_t offset = query_id * dataset_dim;
    CUDA_CHECK(cudaMemcpy(d_query_point,
                          d_dataset + offset,
                          dataset_dim * sizeof(float),
                          cudaMemcpyDeviceToDevice));
}

// Liberar memoria GPU
inline void liberarMemoriaStage2GPU()
{
    if (d_dataset)
    {
        CUDA_CHECK(cudaFree(d_dataset));
        d_dataset = nullptr;
    }
    if (d_query_point)
    {
        CUDA_CHECK(cudaFree(d_query_point));
        d_query_point = nullptr;
    }
}

// --- KERNELS CUDA ---

// Device function: Warp Shuffle Reduction
__device__ float warpReduceSum(float val)
{
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
    {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Kernel: Calcular distancias L2 para múltiples candidatos
__global__ void kernel_bulk_distance_computation(
    float *dataset,          // Dataset completo en global memory
    float *query_point,      // Query point en global memory
    int *candidatos,         // IDs de candidatos
    int num_candidatos,      // Número de candidatos
    int dim,                 // Dimensionalidad
    float *output_distancias // Distancias de salida
)
{
    // Shared memory para query point (acceso rápido)
    extern __shared__ float shared_query[];

    // Cargar query point a shared memory (coalescencia)
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
    {
        shared_query[i] = query_point[i];
    }
    __syncthreads();

    // Cada bloque procesa un candidato
    int candidato_idx = blockIdx.x;

    if (candidato_idx >= num_candidatos)
        return;

    int candidate_id = candidatos[candidato_idx];

    // Calcular distancia L2 cuadrada usando todos los threads del bloque
    float partial_sum = 0.0f;

    // Cada thread procesa una porción de las dimensiones
    int tid = threadIdx.x;
    int stride = blockDim.x;

    const float *candidate_vec = dataset + candidate_id * dim;

    // Calcular suma parcial de diferencias al cuadrado
    for (int d = tid; d < dim; d += stride)
    {
        float diff = shared_query[d] - candidate_vec[d];
        partial_sum += diff * diff;
    }

    // Reducción a nivel de warp
    partial_sum = warpReduceSum(partial_sum);

    // Reducción final en shared memory
    __shared__ float warp_sums[32]; // Máximo 32 warps por bloque

    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    if (lane == 0)
    {
        warp_sums[warp_id] = partial_sum;
    }
    __syncthreads();

    // Thread 0 hace la reducción final
    if (threadIdx.x == 0)
    {
        float total_sum = 0.0f;
        int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

        for (int i = 0; i < num_warps; i++)
        {
            total_sum += warp_sums[i];
        }

        // Calcular distancia L2 (raíz cuadrada)
        output_distancias[candidato_idx] = sqrtf(total_sum);
    }
}

// --- FUNCIÓN PRINCIPAL GPU ---

inline std::vector<float> runBulkDistanceComputationGPU(
    const std::vector<int> &candidatos,
    int num_candidatos,
    int query_id = 0)
{
    std::cout << "\n=== EJECUTANDO STAGE 2 EN GPU ===" << std::endl;
    std::cout << "[GPU Stage 2] Procesando " << num_candidatos << " candidatos..." << std::endl;

    // 1. Cargar dataset a GPU
    cargarDatasetGPU("dataset.bin");

    // 2. Inicializar query point
    inicializarQueryPointGPU(query_id);

    // 3. Preparar candidatos en GPU
    int *d_candidatos;
    CUDA_CHECK(cudaMalloc(&d_candidatos, num_candidatos * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_candidatos, candidatos.data(),
                          num_candidatos * sizeof(int),
                          cudaMemcpyHostToDevice));

    // 4. Preparar salida
    float *d_distancias;
    CUDA_CHECK(cudaMalloc(&d_distancias, num_candidatos * sizeof(float)));

    // 5. Configurar kernel
    dim3 blockDim(THREADS_PER_BLOCK);
    dim3 gridDim(num_candidatos); // Un bloque por candidato

    size_t shared_mem_size = dataset_dim * sizeof(float);

    std::cout << "[GPU Stage 2] Configuración del kernel:" << std::endl;
    std::cout << "        -> Bloques: " << gridDim.x << std::endl;
    std::cout << "        -> Threads por bloque: " << blockDim.x << std::endl;
    std::cout << "        -> Shared memory: " << shared_mem_size / 1024.0 << " KB" << std::endl;
    std::cout << "        -> Dimensiones: " << dataset_dim << std::endl;

    // 6. Ejecutar kernel
    std::cout << "[GPU Stage 2] Lanzando kernel..." << std::endl;

    kernel_bulk_distance_computation<<<gridDim, blockDim, shared_mem_size>>>(
        d_dataset,
        d_query_point,
        d_candidatos,
        num_candidatos,
        dataset_dim,
        d_distancias);

    // Sincronizar y verificar errores
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    std::cout << "[GPU Stage 2] Kernel ejecutado exitosamente" << std::endl;

    // 7. Copiar resultados de vuelta al host
    std::vector<float> distancias_host(num_candidatos);
    CUDA_CHECK(cudaMemcpy(distancias_host.data(), d_distancias,
                          num_candidatos * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // 8. Mostrar resultados
    std::cout << "\n=== RESUMEN GPU STAGE 2: DISTANCIAS CALCULADAS ===" << std::endl;
    std::cout << std::left << std::setw(15) << "Candidato ID"
              << std::setw(20) << "Distancia L2" << std::endl;
    std::cout << std::string(35, '-') << std::endl;

    for (int i = 0; i < num_candidatos; i++)
    {
        std::cout << std::left << std::setw(15) << candidatos[i]
                  << std::fixed << std::setprecision(6)
                  << std::setw(20) << distancias_host[i] << std::endl;
    }

    std::cout << "\n[Output para Stage 3] Array de distancias listo" << std::endl;

    // 9. Liberar memoria
    CUDA_CHECK(cudaFree(d_candidatos));
    CUDA_CHECK(cudaFree(d_distancias));
    liberarMemoriaStage2GPU();

    return distancias_host;
}

// Alias para compatibilidad
inline std::vector<float> runBulkDistanceComputation(
    const std::vector<int> &candidatos,
    int num_candidatos,
    int query_id = 0)
{
    return runBulkDistanceComputationGPU(candidatos, num_candidatos, query_id);
}

#endif // SONG_STAGE_2_CUDA_H