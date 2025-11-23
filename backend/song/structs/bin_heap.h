#pragma once

// [begin, end)
template<class T>
void push_heap(T* begin, T*end) {
    T* now = end - 1;
    int parent = (now - begin - 1) / 2;
    while(parent >=0){
        if (*(begin + parent) < *now){
            T temp = *(begin + parent);
            *(begin + parent) = *now;
            *now = temp;
            now = begin + parent;
            parent = (now - begin - 1) / 2;
        } else {
            break;
        }
    }
}

template<class T>
T pop_heap(T* begin, T* end) {
    T ret = *begin;
    *(begin) = *(end - 1);
    T* now = begin;
    int left = 2 * (now - begin) + 1;
    int right = 2 * (now - begin) + 2;
    while (left < (end - begin)) {
        if (right < (end - begin) && *(begin + right) > *(begin + left)) {
            if (*(begin + right) > *now) {
                T temp = *(begin + right);
                *(begin + right) = *now;
                *now = temp;
                now = begin + right;
            } else {
                break;
            }
        } else {
            if (*(begin + left) > *now) {
                T temp = *(begin + left);
                *(begin + left) = *now;
                *now = temp;
                now = begin + left;
            } else {
                break;
            }
        }
        left = 2 * (now - begin) + 1;
        right = 2 * (now - begin) + 2;
    }
    return ret;
}