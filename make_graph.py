import numpy as np
import faiss
import struct

def load_libsvm_file(filename, dim):
    data = []
    labels = []

    with open(filename, "r") as f:
        for line in f:
            parts = line.strip().split()
            label = int(parts[0])
            labels.append(label)

            vec = np.zeros(dim, dtype=np.float32)

            for item in parts[1:]:
                idx, val = item.split(":")
                vec[int(idx)-1] = float(val)

            data.append(vec)

    return np.array(data, dtype=np.float32), np.array(labels)

class FAISSGraphBuilder:
    def __init__(self, data, K=16, metric="L2"):
        self.data = data.astype(np.float32)
        self.metric = metric.lower()
        self.K = K
        self.n_points = len(data)
        self.dim = data.shape[1]

        # Preprocesamiento para métricas
        if self.metric == "cos":
            # Normaliza para que cos = dot product
            faiss.normalize_L2(self.data)

    def build_graph(self):
        print(f"Construyendo grafo con FAISS ({self.metric}) para {self.n_points} puntos...")

        # Selección de índice según la métrica
        if self.metric == "l2":
            index = faiss.IndexFlatL2(self.dim)
        elif self.metric == "ip" or self.metric == "cos":
            index = faiss.IndexFlatIP(self.dim)
        else:
            raise ValueError("Métricas válidas: L2, IP, COS")

        index.add(self.data)

        distances, indices = index.search(self.data, self.K + 1)

        adjacency_list = indices[:, 1:].astype(np.int32)

        print("Construcción completada!")
        return adjacency_list

    def save_graph_binary(self, adjacency_list, filename):
        with open(filename, 'wb') as f:
            f.write(struct.pack('II', self.n_points, self.K))
            adjacency_list.tofile(f)
        print(f"Grafo guardado: {filename}")

    def save_data_binary(self, filename):
        with open(filename, 'wb') as f:
            f.write(struct.pack('II', self.n_points, self.dim))
            self.data.tofile(f)
        print(f"Datos guardados: {filename}")


if __name__ == "__main__":
    data, labels = load_libsvm_file("letter.scale.t", dim=16)

    builder = FAISSGraphBuilder(data, K=16)
    adjacency_list = builder.build_graph()

    builder.save_graph_binary(adjacency_list, "graph_index.bin")
    builder.save_data_binary("dataset.bin")