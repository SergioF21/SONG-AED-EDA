#ifndef SONG_STAGE_1_FINAL_H
#define SONG_STAGE_1_FINAL_H

#include <iostream>
#include <vector>
#include <iomanip>
#include <fstream> 
#include <cstdlib> 

// --- PARÁMETROS DE LA SIMULACIÓN ---
const int DEGREE_LIMIT = 4;  
const int HASH_SIZE = 32;    
const int QUEUE_SIZE = 10;   

// --- ESTRUCTURAS DE DATOS ---

// Variables globales estáticas para evitar errores de linker
static std::vector<int> memoria_global_grafo;
static int num_nodos_total = 0; 

// FUNCIÓN: Carga los datos reales generados por GraphBuilder
inline void cargarGrafoReal(const std::string& filename) {
    std::cout << "[Simulacion] Buscando archivo de grafo: " << filename << "..." << std::endl;
    
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[ERROR CRÍTICO] No se encontró " << filename << std::endl;
        std::cerr << "Tip: Asegúrate de que GraphBuilder se ejecutó primero." << std::endl;
        exit(1);
    }

    int n_points_leidos, k_leidos;
    file.read(reinterpret_cast<char*>(&n_points_leidos), sizeof(int));
    file.read(reinterpret_cast<char*>(&k_leidos), sizeof(int));

    std::cout << "[Simulacion] Archivo encontrado." << std::endl;
    std::cout << "        -> Nodos totales: " << n_points_leidos << std::endl;
    std::cout << "        -> Grado (K): " << k_leidos << std::endl;

    if (k_leidos != DEGREE_LIMIT) {
        std::cerr << "[ERROR] El grado del grafo (" << k_leidos << ") no coincide con DEGREE_LIMIT (" 
                  << DEGREE_LIMIT << ") de la simulación." << std::endl;
        exit(1);
    }

    num_nodos_total = n_points_leidos;

    int total_ints = n_points_leidos * k_leidos;
    memoria_global_grafo.resize(total_ints);
    
    file.read(reinterpret_cast<char*>(memoria_global_grafo.data()), total_ints * sizeof(int));
    file.close();

    std::cout << "[Simulacion] Grafo cargado en RAM (Simulando Memoria Global GPU)." << std::endl;
}

// --- MEMORIA COMPARTIDA SIMULADA ---
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
// === ETAPA 1 - LOCALIZACIÓN DE CANDIDATOS ===
// ============================================================
inline void ejecutarEtapa1_Mariel(MemoriaCompartida& shared_mem) {
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

// Función principal encapsulada
inline void runSongSimulation() {
    // 1. CARGAR EL GRAFO REAL (Generado previamente)
    cargarGrafoReal("graph_index.bin");
    
    MemoriaCompartida memoria;
    
    // 2. SETUP INICIAL DE LA BÚSQUEDA
    int nodo_inicio = 0; 
    
    memoria.cola_q[0] = nodo_inicio;
    memoria.tamano_cola = 1;
    
    // Marcamos el nodo inicial como visitado
    memoria.tabla_visited[nodo_inicio % HASH_SIZE] = 1;

    std::cout << "\n[Simulacion] Iniciando busqueda desde el Nodo " << nodo_inicio << std::endl;

    // 3. EJECUTAR LÓGICA
    ejecutarEtapa1_Mariel(memoria);

    // 4. RESULTADOS
    std::cout << "\n=== RESULTADO FINAL (Input para Noemi) ===" << std::endl;
    std::cout << "Candidatos encontrados: [ ";
    for(int i=0; i<memoria.conteo_candidatos; i++) {
        std::cout << memoria.lista_candidatos[i] << " ";
    }
    std::cout << "]" << std::endl;
}

#endif