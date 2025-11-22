#include <iostream>
#include "GraphBuilder.h"
#include "SongStage1_Final.h"

int main() {
    // Configuración para acelerar I/O en consola (opcional)
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    std::cout << "=============================================" << std::endl;
    std::cout << "     PROYECTO INTEGRADO: GRAPH + SIMULATION  " << std::endl;
    std::cout << "=============================================" << std::endl;
    std::cout << "\n";

    // PASO 1: Construir el grafo y generar los archivos binarios
    // Esto reemplaza la ejecución aislada de GraphBuilder.cpp
    runGraphBuilder();

    std::cout << "\n---------------------------------------------\n";
    std::cout << "   Grafo construido. Iniciando Simulacion... " << std::endl;
    std::cout << "---------------------------------------------\n\n";

    // PASO 2: Ejecutar la simulación de Song Stage 1
    // Esto reemplaza la ejecución aislada de SongStage1_Final.cpp
    runSongSimulation();

    std::cout << "\n=============================================" << std::endl;
    std::cout << "          EJECUCION COMPLETADA               " << std::endl;
    std::cout << "=============================================" << std::endl;

    return 0;
}