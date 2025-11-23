#ifndef SONG_STAGE_2_BULK_DISTANCE_H
#define SONG_STAGE_2_BULK_DISTANCE_H

#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>

// --- PARÁMETROS ---
const int WARP_SIZE = 32; // Simulación de 32 hilos por warp (GPU)

// --- ESTRUCTURAS DE DATOS ---

// Dataset cargado en memoria global (simulando GPU global memory)
struct DatasetGPU
{
    int n_points;
    int dim;
    std::vector<float> data; // Datos aplanados row-major
};

static DatasetGPU dataset_gpu;
static std::vector<float> query_point; // Punto de consulta en shared memory

// --- FUNCIONES AUXILIARES ---

// Cargar dataset desde archivo binario
inline void cargarDatasetParaDistancias(const std::string &filename)
{
    std::cout << "\n[Stage 2] Cargando dataset desde: " << filename << "..." << std::endl;

    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "[ERROR] No se pudo abrir " << filename << std::endl;
        exit(1);
    }

    file.read(reinterpret_cast<char *>(&dataset_gpu.n_points), sizeof(int));
    file.read(reinterpret_cast<char *>(&dataset_gpu.dim), sizeof(int));

    dataset_gpu.data.resize(dataset_gpu.n_points * dataset_gpu.dim);
    file.read(reinterpret_cast<char *>(dataset_gpu.data.data()),
              dataset_gpu.data.size() * sizeof(float));

    file.close();

    std::cout << "[Stage 2] Dataset cargado:" << std::endl;
    std::cout << "        -> Puntos: " << dataset_gpu.n_points << std::endl;
    std::cout << "        -> Dimensiones: " << dataset_gpu.dim << std::endl;
}

// Inicializar query point (simula copia a shared memory)
inline void inicializarQueryPoint(int query_id)
{
    if (query_id >= dataset_gpu.n_points)
    {
        std::cerr << "[ERROR] Query ID fuera de rango" << std::endl;
        exit(1);
    }

    query_point.resize(dataset_gpu.dim);

    // Copiar datos del query point desde memoria global
    int offset = query_id * dataset_gpu.dim;
    for (int i = 0; i < dataset_gpu.dim; ++i)
    {
        query_point[i] = dataset_gpu.data[offset + i];
    }

    std::cout << "[Stage 2] Query Point " << query_id << " copiado a Shared Memory" << std::endl;
}

// ============================================================
// === ETAPA 2 - BULK DISTANCE COMPUTATION ===
// ============================================================

// Función para calcular distancia L2 cuadrada entre dos vectores
// Simula el trabajo de UN HILO en el warp
inline float calcular_distancia_parcial_thread(const float *vec_a,
                                               const float *vec_b,
                                               int start_dim,
                                               int end_dim)
{
    float partial_sum = 0.0f;

    for (int d = start_dim; d < end_dim; ++d)
    {
        float diff = vec_a[d] - vec_b[d];
        partial_sum += diff * diff;
    }

    return partial_sum;
}

// Reducción a nivel de warp (simula warp shuffle reduction)
inline float warp_reduction(const std::vector<float> &partial_distances)
{
    float total = 0.0f;

    std::cout << "    [Warp Reduction] Agregando distancias parciales:" << std::endl;

    for (size_t i = 0; i < partial_distances.size(); ++i)
    {
        total += partial_distances[i];
        std::cout << "        Hilo " << i << ": " << std::fixed
                  << std::setprecision(4) << partial_distances[i] << std::endl;
    }

    return total;
}

// Función principal de Bulk Distance Computation
inline std::vector<float> ejecutarEtapa2_Noemi(const std::vector<int> &candidatos,
                                               int num_candidatos)
{
    std::cout << "\n--- INICIO ETAPA 2: BULK DISTANCE COMPUTATION (Lógica de Noemi) ---" << std::endl;
    std::cout << "[Stage 2] Procesando " << num_candidatos << " candidatos..." << std::endl;

    std::vector<float> distancias_finales;
    distancias_finales.reserve(num_candidatos);

    // Calcular cuántas dimensiones procesa cada hilo
    int dims_per_thread = (dataset_gpu.dim + WARP_SIZE - 1) / WARP_SIZE;

    std::cout << "[Stage 2] Configuración de paralelización:" << std::endl;
    std::cout << "        -> Hilos por warp: " << WARP_SIZE << std::endl;
    std::cout << "        -> Dimensiones por hilo: ~" << dims_per_thread << std::endl;
    std::cout << std::endl;

    // Para cada candidato en la lista
    for (int c = 0; c < num_candidatos; ++c)
    {
        int candidate_id = candidatos[c];

        std::cout << "=== Procesando Candidato " << candidate_id << " ===" << std::endl;

        // Validar ID
        if (candidate_id >= dataset_gpu.n_points || candidate_id < 0)
        {
            std::cerr << "[ERROR] Candidato " << candidate_id << " fuera de rango" << std::endl;
            distancias_finales.push_back(1e9); // Distancia infinita
            continue;
        }

        // Obtener el vector del candidato desde memoria global
        const float *candidate_vec = &dataset_gpu.data[candidate_id * dataset_gpu.dim];
        const float *query_vec = query_point.data();

        std::cout << "  [Memoria Global] Leyendo vector del Candidato " << candidate_id << std::endl;

        // SIMULACIÓN DE PARALELIZACIÓN: Dividir el trabajo entre hilos
        std::vector<float> partial_distances;
        partial_distances.reserve(WARP_SIZE);

        std::cout << "  [Hilos del Warp] Calculando distancias parciales..." << std::endl;

        for (int thread_id = 0; thread_id < WARP_SIZE; ++thread_id)
        {
            int start_dim = thread_id * dims_per_thread;
            int end_dim = std::min(start_dim + dims_per_thread, dataset_gpu.dim);

            if (start_dim >= dataset_gpu.dim)
            {
                partial_distances.push_back(0.0f);
                continue;
            }

            // Cada hilo calcula su parte de la distancia
            float partial = calcular_distancia_parcial_thread(query_vec,
                                                              candidate_vec,
                                                              start_dim,
                                                              end_dim);
            partial_distances.push_back(partial);
        }

        // Reducción de warp: agregar todas las distancias parciales
        float distancia_total = warp_reduction(partial_distances);

        // Tomar la raíz cuadrada para obtener distancia L2
        float distancia_l2 = std::sqrt(distancia_total);

        distancias_finales.push_back(distancia_l2);

        std::cout << "  [RESULTADO] Distancia L2 = " << std::fixed
                  << std::setprecision(6) << distancia_l2 << std::endl;
        std::cout << std::endl;
    }

    std::cout << "--- FIN ETAPA 2 ---" << std::endl;
    return distancias_finales;
}

// Función de visualización de resultados
inline void mostrarResultadosEtapa2(const std::vector<int> &candidatos,
                                    const std::vector<float> &distancias)
{
    std::cout << "\n=== RESUMEN ETAPA 2: DISTANCIAS CALCULADAS ===" << std::endl;
    std::cout << std::left << std::setw(15) << "Candidato ID"
              << std::setw(20) << "Distancia L2" << std::endl;
    std::cout << std::string(35, '-') << std::endl;

    for (size_t i = 0; i < candidatos.size(); ++i)
    {
        std::cout << std::left << std::setw(15) << candidatos[i]
                  << std::fixed << std::setprecision(6)
                  << std::setw(20) << distancias[i] << std::endl;
    }

    std::cout << "\n[Output para Etapa 3] Array de distancias listo para mantenimiento" << std::endl;
}

// Función wrapper principal para llamar desde main
inline std::vector<float> runBulkDistanceComputation(const std::vector<int> &candidatos,
                                                     int num_candidatos,
                                                     int query_id = 0)
{
    // 1. Cargar dataset
    cargarDatasetParaDistancias("dataset.bin");

    // 2. Inicializar query point
    inicializarQueryPoint(query_id);

    // 3. Ejecutar cálculo de distancias
    std::vector<float> distancias = ejecutarEtapa2_Noemi(candidatos, num_candidatos);

    // 4. Mostrar resultados
    mostrarResultadosEtapa2(candidatos, distancias);

    // 5. Retornar distancias para Stage 3
    return distancias;
}

#endif