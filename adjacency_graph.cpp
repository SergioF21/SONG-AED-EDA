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

#include <faiss/IndexFlat.h>

// --- ESTRUCTURAS DE DATOS ---
struct Dataset {
    int n_points;
    int dim;
    std::vector<float> data;
    std::vector<int> labels;
};

struct Neighbor {
    int id;
    float distance;
    
    bool operator<(const Neighbor& other) const {
        return distance < other.distance;
    }
};

// --- FUNCIONES AUXILIARES ---

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
    
    // Si no encuentra archivo → dataset dummy
    if (!file.is_open()) {
        std::cerr << "[AVISO] No se encontro " << filename << ". Generando datos aleatorios..." << std::endl;
        Dataset dummy;
        dummy.n_points = 100;
        dummy.dim = dim;
        dummy.data.resize(dummy.n_points * dim);
        for(int i = 0; i < dummy.data.size(); i++)
            dummy.data[i] = (float)(rand() % 100) / 100.0f;
        return dummy;
    }

    Dataset ds;
    ds.dim = dim;
    std::string line;
    
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::vector<std::string> parts = split(line, ' ');
        
        try { ds.labels.push_back(std::stoi(parts[0])); }
        catch (...) { continue; }

        std::vector<float> vec(dim, 0.0f);
        for (size_t i = 1; i < parts.size(); ++i) {
            std::vector<std::string> pair = split(parts[i], ':');
            if (pair.size() == 2) {
                int idx = std::stoi(pair[0]);
                float val = std::stof(pair[1]);
                if (idx - 1 < dim && idx - 1 >= 0)
                    vec[idx - 1] = val;
            }
        }
        ds.data.insert(ds.data.end(), vec.begin(), vec.end());
    }

    ds.n_points = ds.labels.size();
    std::cout << "[GraphBuilder] Cargados " << ds.n_points << " puntos." << std::endl;
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

// --- CREACIÓN DEL GRAFO USANDO FAISS ---
inline void build_and_save_graph_faiss(const Dataset& ds, int K, const std::string& filename) {
    std::cout << "[GraphBuilder] Construyendo grafo con FAISS..." << std::endl;

    // Índice L2
    faiss::IndexFlatL2 index(ds.dim);
    index.add(ds.n_points, ds.data.data());

    // Buscar K+1 (incluye self-match)
    std::vector<faiss::idx_t> I(ds.n_points * (K + 1));
    std::vector<float>       D(ds.n_points * (K + 1));

    index.search(ds.n_points, ds.data.data(), K + 1, D.data(), I.data());

    // Guardar archivo
    std::ofstream out(filename, std::ios::binary);
    out.write(reinterpret_cast<const char*>(&ds.n_points), sizeof(int));
    out.write(reinterpret_cast<const char*>(&K), sizeof(int));

    std::vector<int> adj_flat;
    adj_flat.reserve(ds.n_points * K);

    for (int i = 0; i < ds.n_points; i++) {
        int base = i * (K + 1);
        for (int k = 1; k < K + 1; k++) {
            adj_flat.push_back((int)I[base + k]);
        }
    }

    out.write(reinterpret_cast<const char*>(adj_flat.data()), adj_flat.size() * sizeof(int));
    out.close();

    std::cout << "- Archivo creado: " << filename << std::endl;
}

// --- FUNCIÓN PRINCIPAL ---
inline void runGraphBuilder() {
    std::cout << "=== GENERADOR DE GRAFO (GraphBuilder + FAISS) ===" << std::endl;
    
    std::string input_file = "letter.scale"; 
    int dim = 16;
    int K = 16;

    Dataset ds = load_libsvm_file(input_file, dim);

    save_data_binary(ds, "dataset.bin");

    build_and_save_graph_faiss(ds, K, "graph_index.bin");

    std::cout << "=== PROCESO DE CONSTRUCCION TERMINADO ===" << std::endl;
}

#endif


int main() {
    runGraphBuilder();
    return 0;
}
