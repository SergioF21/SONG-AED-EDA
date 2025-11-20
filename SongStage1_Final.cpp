#include <iostream>
#include <vector>
#include <iomanip>
#include <fstream>  // Necesario para leer archivos
#include <cstdlib>  // Para exit()

// --- PARÁMETROS DE LA SIMULACIÓN ---
// Deben coincidir con lo que generó GraphBuilder.cpp
const int DEGREE_LIMIT = 4;  // K=4 (Grado fijo)
const int HASH_SIZE = 32;    // Tamaño de la tabla hash simulada
const int QUEUE_SIZE = 10;   // Tamaño de la cola de prioridad

// --- ESTRUCTURAS DE DATOS ---

// 1. Memoria Global (El Grafo gigante real)
std::vector<int> memoria_global_grafo;
int num_nodos_total = 0; // Para saber cuántos nodos cargamos

// FUNCIÓN NUEVA: Carga los datos reales generados por GraphBuilder
void cargarGrafoReal(const std::string& filename) {
    std::cout << "[Setup] Buscando archivo de grafo: " << filename << "..." << std::endl;
    
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[ERROR CRÍTICO] No se encontró " << filename << std::endl;
        std::cerr << "Tip: Ejecuta primero el GraphBuilder.cpp para generar este archivo." << std::endl;
        exit(1);
    }

    // 1. Leer encabezado (N y K)
    int n_points_leidos, k_leidos;
    file.read(reinterpret_cast<char*>(&n_points_leidos), sizeof(int));
    file.read(reinterpret_cast<char*>(&k_leidos), sizeof(int));

    std::cout << "[Setup] Archivo encontrado." << std::endl;
    std::cout << "        -> Nodos totales: " << n_points_leidos << std::endl;
    std::cout << "        -> Grado (K): " << k_leidos << std::endl;

    // Validación de seguridad
    if (k_leidos != DEGREE_LIMIT) {
        std::cerr << "[ERROR] El grado del grafo (" << k_leidos << ") no coincide con DEGREE_LIMIT (" 
                  << DEGREE_LIMIT << ") de la simulación." << std::endl;
        exit(1);
    }

    num_nodos_total = n_points_leidos;

    // 2. Leer la lista de adyacencia completa
    // El tamaño total es N * K enteros
    int total_ints = n_points_leidos * k_leidos;
    memoria_global_grafo.resize(total_ints);
    
    file.read(reinterpret_cast<char*>(memoria_global_grafo.data()), total_ints * sizeof(int));
    file.close();

    std::cout << "[Setup] Grafo cargado en RAM (Simulando Memoria Global GPU)." << std::endl;
}

// --- MEMORIA COMPARTIDA SIMULADA (Tu espacio de trabajo) ---
struct MemoriaCompartida {
    int cola_q[QUEUE_SIZE];      
    int tamano_cola = 0;
    
    int tabla_visited[HASH_SIZE]; 
    
    int lista_candidatos[DEGREE_LIMIT]; 
    int conteo_candidatos = 0;

    MemoriaCompartida() {
        for(int i=0; i<HASH_SIZE; i++) tabla_visited[i] = 0;
        for(int i=0; i<DEGREE_LIMIT; i++) lista_candidatos[i] = 0;
    }
};

// ============================================================
// === TU PARTE: ETAPA 1 - LOCALIZACIÓN DE CANDIDATOS ===
// ============================================================
void ejecutarEtapa1_Mariel(MemoriaCompartida& shared_mem) {
    std::cout << "\n--- INICIO ETAPA 1 (Logica de Mariel) ---" << std::endl;

    int now_idx = -1;

    // PASO 1: Extraer el mejor vértice de la cola Q
    if (shared_mem.tamano_cola > 0) {
        now_idx = shared_mem.cola_q[0]; 
        shared_mem.tamano_cola--;       
        std::cout << "[Hilo 0] Extrae nodo {" << now_idx << "} de la Cola Q." << std::endl;
    } else {
        return;
    }

    // PASO 2: Acceder a Memoria Global (Grafo Real)
    std::cout << "[Hilo 0] Accediendo a Memoria Global para leer vecinos..." << std::endl;

    for (int i = 0; i < DEGREE_LIMIT; ++i) {
        // Aritmética de punteros para grafo aplanado
        int index_memoria = now_idx * DEGREE_LIMIT + i;
        
        // Seguridad: no leer fuera de memoria
        if (index_memoria >= memoria_global_grafo.size()) break;

        int vecino_id = memoria_global_grafo[index_memoria];

        // PASO 3: Comprobar Visited (Memoria Compartida)
        int hash_idx = vecino_id % HASH_SIZE;
        bool ya_visitado = (shared_mem.tabla_visited[hash_idx] == 1);

        std::cout << "    -> Vecino: " << vecino_id;
        
        if (ya_visitado) {
            std::cout << " [YA VISITADO] -> Ignorar." << std::endl;
        } else {
            // PASO 4: Añadir a Lista de Candidatos
            if (shared_mem.conteo_candidatos < DEGREE_LIMIT) {
                shared_mem.lista_candidatos[shared_mem.conteo_candidatos] = vecino_id;
                shared_mem.conteo_candidatos++;
                shared_mem.tabla_visited[hash_idx] = 1;
                std::cout << " [AGREGADO] -> Pasa a Etapa 2." << std::endl;
            }
        }
    }
}

int main() {
    // 1. CARGAR EL GRAFO REAL (Generado por GraphBuilder)
    // Asegúrate de que "graph_index.bin" esté en la misma carpeta
    cargarGrafoReal("graph_index.bin");
    
    MemoriaCompartida memoria;
    
    // 2. SETUP INICIAL DE LA BÚSQUEDA
    // Elegimos un nodo arbitrario para empezar (ej. Nodo 0)
    // En una búsqueda real, este sería el "Entry Point" del grafo
    int nodo_inicio = 0; 
    
    memoria.cola_q[0] = nodo_inicio;
    memoria.tamano_cola = 1;
    
    // Marcamos el nodo inicial como visitado en tu hash table
    memoria.tabla_visited[nodo_inicio % HASH_SIZE] = 1;

    std::cout << "\n[Simulacion] Iniciando busqueda desde el Nodo " << nodo_inicio << std::endl;

    // 3. EJECUTAR TU LÓGICA
    ejecutarEtapa1_Mariel(memoria);

    // 4. RESULTADOS
    std::cout << "\n=== RESULTADO FINAL (Input para Noemi) ===" << std::endl;
    std::cout << "Candidatos encontrados: [ ";
    for(int i=0; i<memoria.conteo_candidatos; i++) {
        std::cout << memoria.lista_candidatos[i] << " ";
    }
    std::cout << "]" << std::endl;

    return 0;
}