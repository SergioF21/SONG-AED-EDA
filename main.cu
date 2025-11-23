#include <iostream>
#include <vector>
#include "adjacency_graph.h"
#include "SongStage1.cuh"
#include "SongStage2.cu"
#include "SongStage3.cuh"

int main()
{
    // Configuración para acelerar I/O en consola (opcional)
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    std::cout << "=============================================" << std::endl;
    std::cout << "     PROYECTO INTEGRADO: GRAPH + SIMULATION  " << std::endl;
    std::cout << "=============================================" << std::endl;
    std::cout << "\n";

    // PASO 1: Construir el grafo y generar los archivos binarios
    runGraphBuilder();

    std::cout << "\n---------------------------------------------\n";
    std::cout << "   Grafo construido. Iniciando Simulacion... " << std::endl;
    std::cout << "---------------------------------------------\n\n";

    // PASO 2: Ejecutar Stage 1 y OBTENER los candidatos reales
    std::vector<int> candidatos_stage1 = load_graph_bin("graph_index.bin", /*n_points*/ *(new int));
    process_stage1("dataset.bin", candidatos_stage1);

    std::cout << "\n---------------------------------------------\n";
    std::cout << "   Stage 1 completada. Iniciando Stage 2... " << std::endl;
    std::cout << "---------------------------------------------\n\n";

    // PASO 3: Ejecutar Stage 2 con los candidatos REALES de Stage 1
    int query_id = 0; // El mismo nodo desde donde empezó la búsqueda
    std::vector<float> distancias = runBulkDistanceComputation(
        candidatos_stage1,
        candidatos_stage1.size(),
        query_id);

    std::cout << "\n=============================================" << std::endl;
    std::cout << "          EJECUCION COMPLETADA               " << std::endl;
    std::cout << "     (Etapas 1 y 2 integradas con éxito)    " << std::endl;
    std::cout << "=============================================" << std::endl;

    
    // PASO 4: Ejecutar Stage 3 con los resultados de Stage 2
    std::cout << "\n---------------------------------------------\n";
    std::cout << "   Iniciando Stage 3 con resultados Stage 2..." << std::endl;
    std::cout << "---------------------------------------------\n\n";

    run_stage3_gpu("dataset.bin", "query.bin");
    



    return 0;
}