#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <fstream>

// --- CONFIGURACIÓN ---
// Estos valores deben coincidir con la generación de datos
#define DEGREE_LIMIT 4 
#define QUEUE_SIZE 10
#define HASH_SIZE 32

// Macro para verificar errores de CUDA de forma limpia
#define CHECK_CUDA(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        printf("Error CUDA: %s:%d, ", __FILE__, __LINE__); \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

// --- KERNEL DE GPU (Etapa 1: Localización de Candidatos) ---
// Este código se ejecuta en la tarjeta gráfica
__global__ void song_stage1_candidate_locating(
    int* d_graph,          // Memoria Global (VRAM): El grafo completo
    int* d_candidates_out, // Output para validación
    int start_node         // Nodo desde donde empezamos a buscar
) {
    // Memoria Compartida (L1 Cache - Muy rápida)
    // Gestionada explícitamente para reducir latencia
    __shared__ int s_queue[QUEUE_SIZE];      
    __shared__ int s_queue_size;
    __shared__ int s_visited[HASH_SIZE];     
    __shared__ int s_candidates[DEGREE_LIMIT]; 
    __shared__ int s_candidate_count;

    // 1. SETUP INICIAL (Hilo 0 prepara el entorno)
    if (threadIdx.x == 0) {
        s_candidate_count = 0; 
        
        // Inicializamos la cola con el nodo de inicio
        s_queue[0] = start_node; 
        s_queue_size = 1;

        // Limpiamos la tabla hash (Visited)
        for(int i=0; i<HASH_SIZE; i++) s_visited[i] = 0;
        
        // Marcamos el inicio como visitado
        s_visited[start_node % HASH_SIZE] = 1; 
    }
    __syncthreads(); // Sincronización de bloque

    // 2. LÓGICA PRINCIPAL (Single Thread Execution para esta etapa)
    if (threadIdx.x == 0) {
        int now_idx = -1;

        // PASO A: Extraer nodo de la cola
        if (s_queue_size > 0) {
            now_idx = s_queue[0]; 
            s_queue_size--;
        }

        if (now_idx != -1) {
            // PASO B: Leer vecinos desde Memoria Global (Lenta -> Rápida)
            // Aquí ocurre el acceso coalescente si usáramos más hilos, 
            // pero por lógica algorítmica usamos 1 hilo maestro.
            for (int i = 0; i < DEGREE_LIMIT; ++i) {
                
                // Aritmética de punteros para ubicar vecinos en el array plano
                int neighbor_id = d_graph[now_idx * DEGREE_LIMIT + i];

                // Validar fin de lista de adyacencia (padding -1)
                // Nota: En producción real, verificamos límites de array también.
                if (neighbor_id == -1) break;

                // PASO C: Filtrado con Bloom Filter / Hash Table
                int hash_idx = neighbor_id % HASH_SIZE;
                bool is_visited = (s_visited[hash_idx] == 1);

                if (!is_visited) {
                    // PASO D: Agregar a lista de candidatos
                    if (s_candidate_count < DEGREE_LIMIT) {
                        s_candidates[s_candidate_count] = neighbor_id;
                        s_candidate_count++;
                        s_visited[hash_idx] = 1;
                        
                        // Escribir a memoria global solo para validación (demo)
                        d_candidates_out[s_candidate_count-1] = neighbor_id;
                    }
                }
            }
        }
    }
    __syncthreads(); 
    // Aquí iniciaría la Etapa 2 (Cálculo de distancias en paralelo)
}

// --- FUNCIÓN HOST: Cargar Grafo Binario ---
std::vector<int> load_graph_bin(const std::string& filename, int& n_points) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: No se pudo abrir " << filename << std::endl;
        exit(1);
    }

    int k_val;
    file.read(reinterpret_cast<char*>(&n_points), sizeof(int));
    file.read(reinterpret_cast<char*>(&k_val), sizeof(int));

    if (k_val != DEGREE_LIMIT) {
        std::cerr << "Error: El grado del grafo (" << k_val << ") no coincide con el compilado." << std::endl;
        exit(1);
    }

    int total_size = n_points * k_val;
    std::vector<int> graph_data(total_size);
    file.read(reinterpret_cast<char*>(graph_data.data()), total_size * sizeof(int));
    return graph_data;
}

int main() {
    std::cout << "=== SONG CUDA IMPLEMENTATION (STAGE 1) ===" << std::endl;

    // 1. Cargar datos reales en el Host (CPU RAM)
    int num_nodes;
    std::vector<int> h_graph = load_graph_bin("graph_index.bin", num_nodes);
    
    size_t bytes_graph = h_graph.size() * sizeof(int);
    std::cout << "Grafo cargado: " << num_nodes << " nodos. (" << bytes_graph / 1024 << " KB)" << std::endl;

    // 2. Asignar memoria en el Device (GPU VRAM)
    int* d_graph;
    int* d_candidates_out;
    
    CHECK_CUDA(cudaMalloc((void**)&d_graph, bytes_graph));
    CHECK_CUDA(cudaMalloc((void**)&d_candidates_out, DEGREE_LIMIT * sizeof(int)));
    CHECK_CUDA(cudaMemset(d_candidates_out, 0, DEGREE_LIMIT * sizeof(int)));

    // 3. Transferencia Host -> Device (El cuello de botella real)
    CHECK_CUDA(cudaMemcpy(d_graph, h_graph.data(), bytes_graph, cudaMemcpyHostToDevice));

    // 4. Ejecución del Kernel
    // Lanzamos 1 bloque con 32 hilos (1 warp) para procesar una consulta de prueba
    int start_node = 0;
    std::cout << "Lanzando kernel para nodo inicial: " << start_node << std::endl;
    
    song_stage1_candidate_locating<<<1, 32>>>(d_graph, d_candidates_out, start_node);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 5. Recuperar resultados Device -> Host
    std::vector<int> h_candidates(DEGREE_LIMIT);
    CHECK_CUDA(cudaMemcpy(h_candidates.data(), d_candidates_out, DEGREE_LIMIT * sizeof(int), cudaMemcpyDeviceToHost));

    // Mostrar validación
    std::cout << "Candidatos encontrados (Output de GPU): [ ";
    for(int x : h_candidates) {
        if(x != 0) std::cout << x << " ";
    }
    std::cout << "]" << std::endl;

    // Limpieza
    cudaFree(d_graph);
    cudaFree(d_candidates_out);
    
    return 0;
}