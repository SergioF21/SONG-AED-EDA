#include <iostream>
#include <vector>
#include "GraphBuilder.h"
#include "SongStage1.cu"
#include "SongStage2.cu"

int main()
{
    std::cout << "=============================================" << std::endl;
    std::cout << "   PROYECTO SONG - VERSIÓN GPU (CUDA)       " << std::endl;
    std::cout << "   Etapas 1 y 2 completamente en GPU        " << std::endl;
    std::cout << "=============================================" << std::endl;
    std::cout << "\n";

    // =========================================================
    // PASO 1: Construcción del Grafo (CPU)
    // =========================================================
    std::cout << "--- PASO 1: Construcción del Grafo (CPU) ---" << std::endl;
    runGraphBuilder();

    std::cout << "\n---------------------------------------------\n";
    std::cout << "   Grafo construido. Iniciando GPU...       " << std::endl;
    std::cout << "---------------------------------------------\n\n";

    // =========================================================
    // PASO 2: Stage 1 - Localización de Candidatos (GPU)
    // =========================================================
    std::cout << "--- PASO 2: Stage 1 - Localización (GPU) ---" << std::endl;
    std::vector<int> candidatos_stage1 = runSongSimulationGPU();

    if (candidatos_stage1.empty())
    {
        std::cerr << "[ERROR] No se encontraron candidatos en Stage 1" << std::endl;
        return 1;
    }

    std::cout << "\n---------------------------------------------\n";
    std::cout << "   Stage 1 GPU completada. Iniciando Stage 2" << std::endl;
    std::cout << "---------------------------------------------\n\n";

    // =========================================================
    // PASO 3: Stage 2 - Bulk Distance Computation (GPU)
    // =========================================================
    std::cout << "--- PASO 3: Stage 2 - Distancias (GPU) ---" << std::endl;
    int query_id = 0; // Mismo nodo desde donde empezó la búsqueda

    std::vector<float> distancias = runBulkDistanceComputationGPU(
        candidatos_stage1,
        candidatos_stage1.size(),
        query_id);

    // =========================================================
    // RESUMEN FINAL
    // =========================================================
    std::cout << "\n=============================================" << std::endl;
    std::cout << "     PIPELINE GPU COMPLETADO EXITOSAMENTE   " << std::endl;
    std::cout << "=============================================" << std::endl;
    std::cout << "\nResumen de ejecución:" << std::endl;
    std::cout << "  ✓ Stage 1: " << candidatos_stage1.size() << " candidatos encontrados" << std::endl;
    std::cout << "  ✓ Stage 2: " << distancias.size() << " distancias calculadas" << std::endl;
    std::cout << "\nDatos listos para Stage 3 (Mantenimiento de estructuras)" << std::endl;

    return 0;
}