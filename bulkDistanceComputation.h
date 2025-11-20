// ============================================================
// === ETAPA 2 - BULK DISTANCE COMPUTATION (Lógica de Noemi) ===
// ============================================================

#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>
#include "SongStage1_Final.h"

extern int dataset_n;
extern int dataset_dim;
extern std::vector<float> memoria_global_dataset;

extern const int DEGREE_LIMIT;
extern const int QUEUE_SIZE;

// ------------------------------------------------------------
// Calcular distancia L2
// ------------------------------------------------------------
float l2_distance(const float *a, const float *b, int dim)
{
    float sum = 0;
    for (int i = 0; i < dim; i++)
    {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

// ------------------------------------------------------------
// Insertar ordenadamente en la Cola Q
// ------------------------------------------------------------
void insertarEnColaOrdenada(
    int id, float dist,
    int cola_q[], float distancias_q[],
    int &tamano_q)
{
    if (tamano_q < QUEUE_SIZE)
    {
        cola_q[tamano_q] = id;
        distancias_q[tamano_q] = dist;
        tamano_q++;
    }

    // Ordenar por distancia ascendente
    for (int i = 0; i < tamano_q - 1; i++)
    {
        for (int j = i + 1; j < tamano_q; j++)
        {
            if (distancias_q[j] < distancias_q[i])
            {
                std::swap(distancias_q[i], distancias_q[j]);
                std::swap(cola_q[j], cola_q[i]);
            }
        }
    }
}

// ------------------------------------------------------------
// Estructura compatible con la etapa 1
// ------------------------------------------------------------
struct MemoriaCompartida
{
    int cola_q[QUEUE_SIZE];
    float distancias_q[QUEUE_SIZE];
    int tamano_cola = 0;

    int tabla_visited[32];
    int lista_candidatos[DEGREE_LIMIT];
    int conteo_candidatos = 0;

    MemoriaCompartida()
    {
        for (int i = 0; i < QUEUE_SIZE; i++)
            distancias_q[i] = 1e9;
        for (int i = 0; i < DEGREE_LIMIT; i++)
            lista_candidatos[i] = -1;
    }
};

// ------------------------------------------------------------
// === ETAPA 2 (NOEMI): CALCULAR DISTANCIAS DE CANDIDATOS ===
// ------------------------------------------------------------
void ejecutarEtapa2_Noemi(MemoriaCompartida &mem, int query_id)
{
    std::cout << "\n--- INICIO ETAPA 2 (Noemi) ---\n";

    const float *query_vec =
        &memoria_global_dataset[query_id * dataset_dim];

    for (int i = 0; i < mem.conteo_candidatos; i++)
    {

        int cand_id = mem.lista_candidatos[i];
        const float *cand_vec =
            &memoria_global_dataset[cand_id * dataset_dim];

        float dist = l2_distance(query_vec, cand_vec, dataset_dim);

        std::cout << "[Etapa 2] Dist(query=" << query_id
                  << ", cand=" << cand_id
                  << ") = " << dist << "\n";

        insertarEnColaOrdenada(
            cand_id, dist,
            mem.cola_q, mem.distancias_q,
            mem.tamano_cola);
    }
}
