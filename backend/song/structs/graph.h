#pragma once
#include<vector>
#include<algorithm>
#include<queue>
#include<stdlib.h>
#include<limits>
#include"config.h"
#include<random>
#include<unordered_set>

class Graph{
public:
    Graph(): num_nodes(0), num_edges(0) {}
    void add_vertex(idx_t vertex_id, std::vector<std::pair<int, value_t>> &neighbors){
        adj_list.push_back(std::vector<int>());
        num_nodes++;
    }   
    void add_edge(int src, int dst, bool directed);
    const std::vector<int>& get_neighbors(int node_id) const;
    int get_num_nodes() const { return num_nodes; }
    int get_num_edges() const { return num_edges; }
private:
    std::vector<std::vector<int>> adj_list;
    int num_nodes;      
    int num_edges;
};