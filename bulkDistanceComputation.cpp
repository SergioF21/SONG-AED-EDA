#include <vector>
#include <cmath>
#include <omp.h> // Para paralelización en CPU con OpenMP
#include <iostream>

// -------------------------------------------------------------
// BULK DISTANCE COMPUTATION STAGE
// -------------------------------------------------------------
//
// Input:
// - query p (vector)
// - candidates = [id1, id2, id3, ...]
// - data = matriz con los vectores del dataset (data[id] = vector)
//
// Output:
// - dist[] = distancias entre p y cada data[id]
//
// Parallel for each candidate:
//     dist[i] = distance(p, data[candidate[i]])
//
// -------------------------------------------------------------

// Función para calcular la distancia euclidiana entre dos vectores
float euclidean_distance(const std::vector<float> &a, const std::vector<float> &b)
{
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); ++i)
    {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

// Etapa 2: Bulk Distance Computation
std::vector<float> bulk_distance_computation(
    const std::vector<float> &query,             // query p
    const std::vector<std::vector<float>> &data, // dataset
    const std::vector<int> &candidates)          // lista de candidatos
{
    std::vector<float> distances(candidates.size());

// Ejecutamos en paralelo el cálculo de distancias
#pragma omp parallel for
    for (int i = 0; i < (int)candidates.size(); ++i)
    {
        distances[i] = euclidean_distance(query, data[candidates[i]]);
    }

    return distances;
}

int main()
{
    // Query vector
    std::vector<float> query = {1.0, 2.0, 3.0};

    // Dataset simulado (4 vectores)
    std::vector<std::vector<float>> data = {
        {1.0, 2.0, 3.0}, // id 0
        {2.0, 3.0, 4.0}, // id 1
        {5.0, 5.0, 5.0}, // id 2
        {0.0, 1.0, 1.0}  // id 3
    };

    // Candidatos (IDs del dataset)
    std::vector<int> candidates = {0, 1, 2, 3};

    // Llamada a la función
    std::vector<float> distances = bulk_distance_computation(query, data, candidates);

    // Mostrar resultados
    std::cout << "Distancias calculadas:" << std::endl;
    for (size_t i = 0; i < distances.size(); ++i)
    {
        std::cout << "  Candidate " << candidates[i] << " -> " << distances[i] << std::endl;
    }

    return 0;
}
