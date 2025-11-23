#ifndef GRAPH_BUILDER_H
#define GRAPH_BUILDER_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <cstdlib>

// --- ESTRUCTURAS DE DATOS ---
struct Dataset {
    int n_points;
    int dim;
    std::vector<float> data;   // Datos aplanados (row-major)
    std::vector<int> labels;
};

struct Neighbor {
    int id;
    float distance;
    
    // Sobrecarga para ordenar
    bool operator<(const Neighbor& other) const {
        return distance < other.distance;
    }
};

// --- FUNCIONES AUXILIARES ---

// Dividir strings (split)
inline std::vector<std::string> split(const std::string &s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

// Cargar archivo LibSVM
inline Dataset load_libsvm_file(const std::string& filename, int dim) {
    std::cout << "[GraphBuilder] Cargando dataset: " << filename << "..." << std::endl;
    std::ifstream file(filename);
    
    // Si no encuentra el archivo, creamos datos dummy para que NO falle la demo
    if (!file.is_open()) {
        std::cerr << "[AVISO] No se encontro " << filename << ". Generando datos aleatorios..." << std::endl;
        Dataset dummy;
        dummy.n_points = 100; // 100 puntos de prueba
        dummy.dim = dim;
        dummy.data.resize(dummy.n_points * dim);
        for(int i=0; i<dummy.data.size(); i++) dummy.data[i] = (float)(rand() % 100) / 100.0f;
        return dummy;
    }

    Dataset ds;
    ds.dim = dim;
    std::string line;
    
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::vector<std::string> parts = split(line, ' ');
        
        try { ds.labels.push_back(std::stoi(parts[0])); } catch (...) { continue; }

        std::vector<float> vec(dim, 0.0f);
        for (size_t i = 1; i < parts.size(); ++i) {
            std::vector<std::string> pair = split(parts[i], ':');
            if (pair.size() == 2) {
                int idx = std::stoi(pair[0]);
                float val = std::stof(pair[1]);
                if (idx - 1 < dim && idx - 1 >= 0) vec[idx - 1] = val;
            }
        }
        ds.data.insert(ds.data.end(), vec.begin(), vec.end());
    }
    ds.n_points = ds.labels.size();
    std::cout << "[GraphBuilder] Carga completada: " << ds.n_points << " puntos." << std::endl;
    return ds;
}

// Guardar dataset en binario
inline void save_data_binary(const Dataset& ds, const std::string& filename) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) return;
    out.write(reinterpret_cast<const char*>(&ds.n_points), sizeof(int));
    out.write(reinterpret_cast<const char*>(&ds.dim), sizeof(int));
    out.write(reinterpret_cast<const char*>(ds.data.data()), ds.data.size() * sizeof(float));
    out.close();
    std::cout << "-> Archivo creado: " << filename << std::endl;
}

// --- ALGORITMO FUERZA BRUTA ---
inline float compute_l2_sq(const float* a, const float* b, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

inline void build_and_save_graph_manual(const Dataset& ds, int K, const std::string& filename) {
    std::cout << "[GraphBuilder] Construyendo grafo (Algoritmo Manual)..." << std::endl;

    std::ofstream out(filename, std::ios::binary);
    out.write(reinterpret_cast<const char*>(&ds.n_points), sizeof(int));
    out.write(reinterpret_cast<const char*>(&K), sizeof(int));

    std::vector<int> adj_flat;
    adj_flat.reserve(ds.n_points * K);

    for (int i = 0; i < ds.n_points; ++i) {
        std::vector<Neighbor> candidates;
        candidates.reserve(ds.n_points);

        const float* vec_i = &ds.data[i * ds.dim];

        for (int j = 0; j < ds.n_points; ++j) {
            if (i == j) continue; 

            const float* vec_j = &ds.data[j * ds.dim];
            float dist = compute_l2_sq(vec_i, vec_j, ds.dim);
            
            candidates.push_back({j, dist});
        }

        if (candidates.size() > K) {
            std::partial_sort(candidates.begin(), candidates.begin() + K, candidates.end());
        } else {
            std::sort(candidates.begin(), candidates.end());
        }

        for (int k = 0; k < K && k < candidates.size(); ++k) {
            adj_flat.push_back(candidates[k].id);
        }

        if (i % 100 == 0) std::cout << "\rProcesando nodo " << i << "/" << ds.n_points << std::flush;
    }
    std::cout << std::endl;

    out.write(reinterpret_cast<const char*>(adj_flat.data()), adj_flat.size() * sizeof(int));
    out.close();
    std::cout << "- Archivo creado: " << filename << std::endl;
}

// Función principal encapsulada para ser llamada desde main.cpp
inline void runGraphBuilder(int argc, char** argv) {
    if (argc < 1) {
        std::cerr << "Uso: " << argv[0] << "  [dim] [K]" << std::endl;
        return;
    }


    std::cout << "=== GENERADOR DE GRAFO (GraphBuilder) ===" << std::endl;
    
    std::string input_file = "datasets/letter.scale"; 
    int dim = std::stoi(argv[1]);
    int K = std::stoi(argv[2]); 

    // 1. Cargar
    Dataset ds = load_libsvm_file(input_file, dim);

    // 2. Guardar Data
    save_data_binary(ds, "dataset.bin");

    // 3. Construir Grafo
    build_and_save_graph_manual(ds, K, "graph_index.bin");

    std::cout << "=== PROCESO DE CONSTRUCCION TERMINADO ===" << std::endl;
}

#endif
int main(int argc, char** argv) {
    runGraphBuilder(argc, argv);
    return 0;
}   