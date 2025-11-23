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

// Helper para escribir JSON para el frontend
void write_results_json(const char* filename, double time_seconds, int K, int num_queries, 
                        const std::vector<int64_t>& ids, const std::vector<float>& dists) {
    std::ofstream out(filename);
    out << "{\n";
    out << "  \"execution_time\": " << time_seconds << ",\n";
    out << "  \"num_queries\": " << num_queries << ",\n";
    out << "  \"k\": " << K << ",\n";
    out << "  \"queries\": [\n";
    
    for (int q = 0; q < num_queries; ++q) {
        out << "    {\n";
        out << "      \"query_id\": " << q << ",\n";
        out << "      \"results\": [";
        for (int k = 0; k < K; ++k) {
            int idx = ids[q * K + k];
            float d = dists[q * K + k];
            out << "{\"id\": " << idx << ", \"distance\": " << d << "}";
            if (k < K - 1) out << ", ";
        }
        out << "]\n";
        out << "    }" << (q < num_queries - 1 ? "," : "") << "\n";
    }
    out << "  ]\n";
    out << "}\n";
    out.close();
    std::cout << "[JSON] Resultados guardados en " << filename << std::endl;
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
    
    std::string input_file = "datasets/letter.scale"; 
    int dim = 16;
    int K = 16;

    Dataset ds = load_libsvm_file(input_file, dim);

    save_data_binary(ds, "dataset_faiss.bin");

    build_and_save_graph_faiss(ds, K, "graph_index_faiss.bin");

    std::cout << "=== PROCESO DE CONSTRUCCION TERMINADO ===" << std::endl;
}

int main(int argc, char** argv) {
    // Argumentos: ./faiss_demo dataset_faiss.bin <start_node> <num_queries> <K>
    if (argc < 5) {
        std::cerr << "Uso: " << argv[0] << " dataset_faiss.bin <start_node> <num_queries> <K>" << std::endl;
        return 1;
    }
    std::string dataset_path = argv[1];
    int start_node = std::atoi(argv[2]);
    int num_queries = std::atoi(argv[3]);
    int K = std::atoi(argv[4]);
    if (num_queries < 1) num_queries = 1;
    if (K < 1) K = 1;
    // 1. Cargar Dataset (Lectura manual de binario)
    std::ifstream ds_file(dataset_path, std::ios::binary);
    if (!ds_file.is_open()) {
        std::cerr << "[ERROR] No se encontró " << dataset_path << std::endl;
        return 1;
    }
    int n_points_file, dim_file;
    ds_file.read(reinterpret_cast<char*>(&n_points_file), sizeof(int));
    ds_file.read(reinterpret_cast<char*>(&dim_file), sizeof(int));
    // Usamos vector en lugar de new[] para manejo automático de memoria (RAII)
    // Pero si prefieres new[], aquí está la lógica equivalente segura:
    float* data = new float[n_points_file * dim_file];
    ds_file.read(reinterpret_cast<char*>(data), n_points_file * dim_file * sizeof(float));
    ds_file.close();
    // 2. Preparar Queries (Simuladas desde el mismo dataset)
    if (start_node >= n_points_file) start_node = 0;
    // Ajustar num_queries si se pasa del final
    if (start_node + num_queries > n_points_file) num_queries = n_points_file - start_node;
    float* query_vectors = new float[num_queries * dim_file];
    for (int i = 0; i < num_queries; ++i) {
        int point_idx = start_node + i;
        for (int d = 0; d < dim_file; ++d) {
            query_vectors[i * dim_file + d] = data[point_idx * dim_file + d];
        }
    }
    std::cout << "Indexando " << n_points_file << " vectores de dimension " << dim_file << "..." << std::endl;
    // 3. Construir Índice FAISS (Flat L2 = Brute Force Exacto)
    faiss::IndexFlatL2 index(dim_file);
    index.add(n_points_file, data);
    // 4. Buscar
    std::vector<float> D(num_queries * K);
    std::vector<faiss::idx_t> I(num_queries * K);
    std::cout << "Ejecutando busqueda FAISS (CPU Baseline)..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    index.search(num_queries, query_vectors, K, D.data(), I.data());
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "[INFO] Tiempo FAISS: " << diff.count() << " s" << std::endl;
    // 5. Guardar JSON
    write_results_json("frontend_results_faiss.json", diff.count(), K, num_queries, I, D);
    // --- LIMPIEZA DE MEMORIA (IMPORTANTE) ---
    delete[] data;
    delete[] query_vectors;
    return 0;
}