#pragma once
#include<random>
#include<vector>
#include"config.h"

typedef float bithash_t;

class BitHash {
public:
    std::vector<bithash_t> hash_matrix;
    int p, k;

    BitHash() {}

    BitHash(int p, int k, int seed = 123): p(p), k(k) {
        std::default_random_engine generator(seed);
        std::normal_distribution<bithash_t> distribution(0.0, 1.0);
        hash_matrix.resize(p*k);
        for (int i = 0; i < p*k; i++) {
            hash_matrix[i] = static_cast<bithash_t>(distribution(generator) > 0 ? 1 : 0);
        }
    }

    std::vector<bool> hash2vecbool(const std::vector<std::pair<int,value_t>>& point) const {
        std::vector<bool> result(p*k);
        for (int i = 0; i < p*k; i++) {
            result[i] = hash_matrix[i] > 0 ? true : false;
        }
        return result;
    }

    uint8_t hash2uint8(const std::vector<std::pair<int, value_t>>& point) {
        uint8_t result = 0;
        for (int i = 0; i < 8 && i < p*k; i++) {
            if (hash_matrix[i] > 0) {
                result |= (1 << i);
            }
        }
        return result;
    }

    std::vector<std::pair<int, value_t>> hash2kv(const std::vector<std::pair<int, data_value_t>> &point){
        std::vector<std::pair<int, value_t>> result;
        for (int i = 0; i < p*k; i++) {
            result.emplace_back(i, static_cast<value_t>(hash_matrix[i]));
        }
        return result;
    }
    
};