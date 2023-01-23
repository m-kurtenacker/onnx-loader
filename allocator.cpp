#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>

#include<map>
#include<vector>
#include<iostream>

#define CACHE_SIZE 200

typedef struct {
    void * data;
    size_t size;
    bool taken;
} Buffer;

int num_cache_items = 0;
Buffer buffer_list[CACHE_SIZE];

std::vector<Buffer> buffered_items;

static int num_allocs;

void print_num_allocs_cpp() {
    std::cout << num_allocs << std::endl;
}

void * get_buffer_cpp(size_t min_size) {
    for (auto& it : buffered_items) {
        if (!it.taken && it.size == min_size) {
            it.taken = true;
            return it.data;
        }
    }

    num_allocs++;
    void * data = malloc(min_size);
    Buffer new_item;
    new_item.data = data;
    new_item.size = min_size;
    new_item.taken = true;

    buffered_items.push_back(new_item);
    return new_item.data;
}

void release_buffer_cpp(void * data) {
    for (auto& it : buffered_items) {
        if (it.data == data) {
            assert(it.taken);
            it.taken = false;
            return;
        }
    }

    assert(false);
}

extern "C" {

void * get_buffer_old(size_t min_size);
void release_buffer_old(void * data);

void * get_buffer(size_t min_size) {
    void * data = get_buffer_cpp(min_size);
    //void * data = get_buffer_old(min_size);
    return data;
}

void release_buffer(void * data) {
    release_buffer_cpp(data);
    //release_buffer_old(data);
}

void * get_buffer_old(size_t min_size) {
    int best_candidate = -1;
    size_t best_candidate_size = 1000000;

    for (int i = 0; i < num_cache_items; i++) {
        if (buffer_list[i].taken)
            continue;
        if (buffer_list[i].size < min_size)
            continue;
        if (buffer_list[i].size < best_candidate_size) {
            best_candidate = i;
            best_candidate_size = buffer_list[i].size;
        }
    }

    if (best_candidate != -1) {
        buffer_list[best_candidate].taken = true;
        return buffer_list[best_candidate].data;
    } else {
        num_allocs++;
        void * data = malloc(min_size);
        buffer_list[num_cache_items].data = data;
        buffer_list[num_cache_items].size = min_size;
        buffer_list[num_cache_items].taken = true;
        num_cache_items++;
        assert(num_cache_items < CACHE_SIZE);
        return buffer_list[num_cache_items - 1].data;
    }
}

void release_buffer_old(void * data) {
    int candidate = -1;

    for (int i = 0; i < num_cache_items; i++) {
        if (buffer_list[i].data == data) {
            candidate = i;
            break;
        }
    }

    if (candidate != -1) {
        buffer_list[candidate].taken = false;
    } else {
        free(data);
    }
}

void print_num_allocs(void) {
    print_num_allocs_cpp();
}
}
