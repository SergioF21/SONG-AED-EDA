#include <iostream>
#include <vector>
#include <iomanip> // Para que se vea bonito el output

// --- PARÁMETROS DE LA SIMULACIÓN ---
// Estos valores simulan las restricciones que tendrías en una GPU real
const int DEGREE_LIMIT = 4;  // Cada nodo tiene máximo 4 vecinos (Grafo de grado fijo)
const int HASH_SIZE = 32;    // Tamaño de la tabla hash simulada
const int QUEUE_SIZE = 10;   // Tamaño de la cola de prioridad

// --- ESTRUCTURAS MOCK (Simulando el trabajo de tus compañeros) ---

// 1. Simulación de la Memoria Global (El Grafo gigante de Dario)
// Usamos un vector plano para simular la memoria contigua de la GPU
std::vector<int> memoria_global_grafo;

// Función auxiliar para inicializar el grafo de Dario (Mock)
void inicializarGrafoMock() {
    // Grafo de 5 nodos. Formato: [Vecino1, Vecino2, Vecino3, Vecino4]
    // -1 significa "espacio vacío" (padding)
    std::vector<int> datos = {
        1, 2, 3, -1,  // Vecinos del Nodo 0
        0, 4, -1, -1, // Vecinos del Nodo 1
        0, -1, -1, -1,// Vecinos del Nodo 2
        0, 4, -1, -1, // Vecinos del Nodo 3
        1, 3, -1, -1  // Vecinos del Nodo 4
    };
    memoria_global_grafo = datos;
}

// --- MEMORIA COMPARTIDA SIMULADA (El espacio de trabajo de Sergio) ---
// En C++ normal son variables globales o de clase
struct MemoriaCompartida {
    int cola_q[QUEUE_SIZE];      // Cola de prioridad
    int tamano_cola = 0;
    
    int tabla_visited[HASH_SIZE]; // Tabla Hash (Visited)
    
    int lista_candidatos[DEGREE_LIMIT]; // Tu output para Noemi
    int conteo_candidatos = 0;

    // Constructor para limpiar la memoria (simula __syncthreads inicial)
    MemoriaCompartida() {
        for(int i=0; i<HASH_SIZE; i++) tabla_visited[i] = 0;
        for(int i=0; i<DEGREE_LIMIT; i++) lista_candidatos[i] = 0;
    }
};

// ============================================================
// === TU PARTE: ETAPA 1 - LOCALIZACIÓN DE CANDIDATOS ===
// ============================================================
// Esta función representa el "Hilo 0" que toma el control
void ejecutarEtapa1_Mariel(MemoriaCompartida& shared_mem) {
    
    std::cout << "\n--- INICIO ETAPA 1 (Lógica de Mariel) ---" << std::endl;

    int now_idx = -1;

    // PASO 1: Extraer el mejor vértice de la cola Q (Memoria Compartida)
    if (shared_mem.tamano_cola > 0) {
        now_idx = shared_mem.cola_q[0]; // Sacamos el primer elemento
        shared_mem.tamano_cola--;       // Reducimos el tamaño
        
        std::cout << "[Paso 1] Hilo extrae nodo {" << now_idx << "} de la Cola Q." << std::endl;
    } else {
        std::cout << "[Error] La cola estaba vacía." << std::endl;
        return;
    }

    // PASO 2: Acceder a Memoria Global (El Grafo de Dario)
    std::cout << "[Paso 2] Leyendo vecinos del nodo " << now_idx << " desde Memoria Global..." << std::endl;

    for (int i = 0; i < DEGREE_LIMIT; ++i) {
        // Calculamos la posición exacta en el array gigante
        int index_memoria = now_idx * DEGREE_LIMIT + i;
        int vecino_id = memoria_global_grafo[index_memoria];

        // Si es -1, terminamos (no hay más vecinos)
        if (vecino_id == -1) break;

        // PASO 3: Comprobar Visited (Memoria Compartida de Sergio)
        // Usamos el ID como hash simple
        int hash_idx = vecino_id % HASH_SIZE;
        bool ya_visitado = (shared_mem.tabla_visited[hash_idx] == 1);

        std::cout << "    -> Vecino detectado: " << vecino_id;
        
        if (ya_visitado) {
            std::cout << " [YA VISITADO] -> Se ignora." << std::endl;
        } else {
            // PASO 4: Añadir a Lista de Candidatos (Para Noemi)
            std::cout << " [NUEVO] -> Procesando..." << std::endl;
            
            if (shared_mem.conteo_candidatos < DEGREE_LIMIT) {
                // 1. Agregamos a la lista de candidatos
                shared_mem.lista_candidatos[shared_mem.conteo_candidatos] = vecino_id;
                shared_mem.conteo_candidatos++;

                // 2. Marcamos como visitado en la tabla Hash (Bloom Filter logic)
                shared_mem.tabla_visited[hash_idx] = 1;

                std::cout << "       [ACCION] Agregado " << vecino_id << " a lista de Candidatos." << std::endl;
            } else {
                std::cout << "       [FULL] Lista de candidatos llena." << std::endl;
            }
        }
    }
    std::cout << "--- FIN ETAPA 1 ---" << std::endl;
}

// --- MAIN (Para probar tu código en Visual Studio) ---
int main() {
    // 1. Inicializar los "Mocks" (Datos falsos para probar)
    inicializarGrafoMock();
    
    MemoriaCompartida memoria;
    
    // 2. Preparamos el escenario inicial
    // Simulamos que ya hay un nodo (Nodo 0) en la cola para empezar
    memoria.cola_q[0] = 0;
    memoria.tamano_cola = 1;
    memoria.tabla_visited[0] = 1; // El nodo 0 ya se considera visitado

    std::cout << "Estado Inicial: Cola Q tiene al nodo {0}. Grafo cargado." << std::endl;

    // 3. EJECUTAR TU LÓGICA
    ejecutarEtapa1_Mariel(memoria);

    // 4. MOSTRAR RESULTADOS (Lo que mostrarías en tu Demo)
    std::cout << "\n=== RESULTADO FINAL (Input para Noemi - Etapa 2) ===" << std::endl;
    std::cout << "Lista 'Candidates' en Memoria Compartida:" << std::endl;
    
    if (memoria.conteo_candidatos == 0) {
        std::cout << "(Vacía)" << std::endl;
    } else {
        std::cout << "[ ";
        for(int i=0; i<memoria.conteo_candidatos; i++) {
            std::cout << memoria.lista_candidatos[i] << " ";
        }
        std::cout << "]" << std::endl;
    }

    std::cout << "\nValidación: El nodo 0 tenía vecinos 1, 2 y 3." << std::endl;
    std::cout << "El algoritmo debió encontrar 1, 2 y 3." << std::endl;

    return 0;
}