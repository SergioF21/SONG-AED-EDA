const int DEGREE_LIMIT = 4;
const int HASH_SIZE = 32;
const int Q_SIZE = 10;

struct sharedMemory
{
    int *q[Q_SIZE];  // Cola de visita
    bool *visited;   // Conjunto de visited
    int *q_size;     // Tamaño actual de la cola de visita
    int *topk[DEGREE_LIMIT];       // Resultados top-k
    int *candidates_processed; // Número de candidatos procesados

    sharedMemory(){
        for (int i = 0; i < Q_SIZE; i++) {
            q[i] = nullptr;
        }
        visited = nullptr;
        q_size = nullptr;
        for (int i = 0; i < DEGREE_LIMIT; i++) {
            topk[i] = nullptr;
        }
        candidates_processed = nullptr;
    }
};
