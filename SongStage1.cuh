#ifndef SONG_STAGE1_H
#define SONG_STAGE1_H

#include <cuda_runtime.h>
#include <vector>
#include <string>
void process_stage1(const std::string& inputFile, std::vector<int>& outputData);

__global__ void song_stage1_candidate_locating(const int* graph, int* candidates_out, int start_node);

std::vector<int> load_graph_bin(const std::string& filename, int& n_points);
#endif // SONG_STAGE3_H