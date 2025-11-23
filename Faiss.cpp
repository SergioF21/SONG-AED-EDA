#include <faiss/IndexFlat.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <chrono>


#include <faiss/IndexFlat.h>

const int DEGREE_LIMIT = 16;

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

    save_data_binary(ds, "dataset_faiss.bin");

    build_and_save_graph_faiss(ds, K, "graph_index_faiss.bin");

    std::cout << "=== PROCESO DE CONSTRUCCION TERMINADO ===" << std::endl;
}

int main() {

    
    runGraphBuilder();

    // data de graph_index_faiss.bin debería ser cargado aqui
    
    std::ifstream ds_file("dataset_faiss.bin", std::ios::binary);
    if (!ds_file.is_open()) {
        std::cerr << "[ERROR] No se encontró dataset_faiss.bin" << std::endl;
        return 1;
    }
    int n_points_file, dim_file;
    ds_file.read(reinterpret_cast<char*>(&n_points_file), sizeof(int));
    ds_file.read(reinterpret_cast<char*>(&dim_file), sizeof(int));

    float* data = new float[n_points_file * dim_file];
    
    ds_file.read(reinterpret_cast<char*>(data), n_points_file * dim_file * sizeof(float));
    ds_file.close();

    int n_queries = std::min(16, n_points_file);
    float* query_vectors = new float[n_queries * dim_file];
    for (int i = 0; i < n_queries * dim_file; i++)
        query_vectors[i] = data[i];

    for (int i = 0; i < n_queries * dim_file; i++)
        query_vectors[i] = data[i]; // solo un ejemplo

    faiss::IndexFlatL2 gt_index(dim_file);
    
    gt_index.add(n_points_file, data);

    int K = DEGREE_LIMIT;
    
    std::vector<float> gt_dist(n_queries * K);

    std::vector<faiss::idx_t> gt_ids(n_queries * K);

    auto start = std::chrono::high_resolution_clock::now();

    gt_index.search(n_queries, query_vectors, K, gt_dist.data(), gt_ids.data());

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end - start;

    std::cout << "[INFO] Tiempo de búsqueda FAISS para " << n_queries << " consultas: " << diff.count() << " segundos." << std::endl;

    for (int i = 0; i < n_queries; i++) {
        std::cout << "Consulta " << i << " vecinos más cercanos:" << std::endl;
        for (int k = 0; k < K; k++) {
            int idx = i * K + k;
            std::cout << "  ID: " << gt_ids[idx] << ", Distancia: " << std::fixed << std::setprecision(4) << gt_dist[idx] << std::endl;
        }
    }

    delete[] data;
    delete[] query_vectors;

    std::cout << "[INFO] Búsqueda completada." << std::endl;

    return 0;
}