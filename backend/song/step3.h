/*4. Para cada candidato y su distancia, actualiza las estructuras en memoria compartida: q (la cola de visita), topk (los resultados) y visited (el conjunto de invitados).        
5. Aqui es donde se aplican las optimizaciones de memoria del siguiente paso:
    - Bounded Priority queue
    - Selected Insertion
    - Visited Delition
*/
#ifndef STEP3_H
#define STEP3_H
#include <string>
#include <vector>
#include "graph.h"
#include "structs/sharedMemory.h"


void step3(const std::string& graph_file, const std::vector<int>& candidates, const std::vector<int>& distances, int k);

int atomicAddWrapper(int* address, int val);







#endif // STEP3_H