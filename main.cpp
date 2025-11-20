#include <iostream>
#include <vector>
#include <iomanip>
#include <fstream> // Necesario para leer archivos
#include <cstdlib> // Para exit()
#include "bulkDistanceComputation.h"
#include "SongStage1_Final.h"

int main()
{
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
    // 4. EJECUTAR ETAPA 2 - NOEMI (Bulk Distance Computation)
    ejecutarEtapa2_Noemi(memoria, nodo_inicio);

    // 4. RESULTADOS
    std::cout << "\n=== RESULTADO FINAL (Input para Noemi) ===" << std::endl;
    std::cout << "Candidatos encontrados: [ ";
    for (int i = 0; i < memoria.conteo_candidatos; i++)
    {
        std::cout << memoria.lista_candidatos[i] << " ";
    }
    std::cout << "]" << std::endl;

    return 0;
}