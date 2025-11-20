#include"graph.h"
#include<fstream>
#include<sstream>
#include<iostream>

using namespace std;

vector<int> load_libsvm_file(const string& filename, Graph& graph, bool directed){
    ifstream file(filename);
    if(!file.is_open()){
        cerr << "Error opening file: " << filename << endl;
        exit(EXIT_FAILURE);
    }

    string line;
    int max_node_id = -1;
    vector<int> labels;

    while(getline(file, line)){
        istringstream iss(line);
        int node_id;
        iss >> node_id;

        if(node_id > max_node_id){
            max_node_id = node_id;
        }

        int label;
        iss >> label;
        labels.push_back(label);

        int neighbor_id;
        while(iss >> neighbor_id){
            graph.add_edge(node_id, neighbor_id, directed);
        }
    }

    file.close();
    return labels;
}