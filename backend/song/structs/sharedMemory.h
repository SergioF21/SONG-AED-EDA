const int DEGREE_LIMIT = 4;
const int HASH_SIZE = 32;
const int Q_SIZE = 10;

struct sharedMemory {
    int* dist;        // distancias de Stage 2
    int* cand;        // candidatos de Stage 1
    int num_cand;     // número de candidatos
    
    int (*q)[2];      // cola q: (node, distance)
    int* q_size;

    int (*topk)[2];   // top-k resultados
    bool* visited;
    int* candidates_processed;
};

