#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <string>

#include "config.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef union {
    struct {
        int magic;
        int length;
    };
    char data[8];
} label_header;

typedef union {
    struct {
        int magic;
        int length;
        int rows;
        int cols;
    };
    char data[16];
} image_header;

void read_labels (char * filename, int ** labels) {
    FILE * labels_file = fopen(filename, "r");

    label_header header;

    for (int i = 0; i < 8; i++) {
        char d = fgetc(labels_file);
        int endian_fixed_address = i - (i % 4) + (3 - i % 4);
        header.data[endian_fixed_address] = d;
    }

    printf("%d %d\n", header.magic, header.length);

    assert(header.magic == 2049);
    //Only supported label format: unsigned byte, 1 dimension

    *labels = (int*)malloc(sizeof(int) * header.length);
    for (int i = 0; i < header.length; i++) {
        char d = fgetc(labels_file);
        (*labels)[i] = d;
    }

    fclose(labels_file);
}

void read_images(char * filename, char ** images) {
    FILE * images_file = fopen(filename, "r");

    image_header header;

    for (int i = 0; i < 16; i++) {
        char d = fgetc(images_file);
        int endian_fixed_address = i - (i % 4) + (3 - i % 4);
        header.data[endian_fixed_address] = d;
    }

    printf("%d %d %d %d\n", header.magic, header.length, header.rows, header.cols);

    assert(header.magic == 2051);
    //Currently only supported type: unsigned byte, 3 dimensions

    *images = (char*)malloc(sizeof(char) * header.length * header.rows * header.cols);

    for (int i = 0; i < header.length * header.rows * header.cols; i++) {
        char d = fgetc(images_file);
        (*images)[i] = d;
    }

    fclose(images_file);
}

size_t * read_idx(char * filename, char ** images) {
    FILE * images_file = fopen(filename, "r");

    image_header header;

    for (int i = 0; i < 16; i++) {
        char d = fgetc(images_file);
        int endian_fixed_address = i - (i % 4) + (3 - i % 4);
        header.data[endian_fixed_address] = d;
    }

    printf("%d %d %d %d\n", header.magic, header.length, header.rows, header.cols);

    assert(header.magic == 2051);
    //Currently only supported type: unsigned byte, 3 dimensions

    *images = (char*)malloc(sizeof(char) * header.length * header.rows * header.cols);

    for (int i = 0; i < header.length * header.rows * header.cols; i++) {
        char d = fgetc(images_file);
        (*images)[i] = d;
    }

    fclose(images_file);

    size_t * sizes = (size_t*) malloc(sizeof(size_t) * 4);
    sizes[0] = header.length;
    sizes[1] = header.rows;
    sizes[2] = header.cols;
    sizes[3] = 0;

    return sizes;
}

void write_idx(char * filename, size_t * sizes, char * images) {
    FILE * images_file = fopen(filename, "w");

    image_header header;

    header.magic = 2051;
    header.length = sizes[0];
    header.rows = sizes[1];
    header.cols = sizes[2];

    assert(sizes[3] == 0);

    for (int i = 0; i < 16; i++) {
        int endian_fixed_address = i - (i % 4) + (3 - i % 4);
        char d = header.data[endian_fixed_address];
        fputc(d, images_file);
    }

    for (int i = 0; i < header.length * header.rows * header.cols; i++) {
        char d = images[i];
        fputc(d, images_file);
    }

    fclose(images_file);
}

void read_cmake_labels (int ** labels) {
    read_labels(LABEL_FILE_PATH, labels);
}

void read_cmake_images (char ** images) {
    read_images(IMAGE_FILE_PATH, images);
}


void read_train_labels (int ** labels) {
    read_labels((PROJECT_ROOT "/mnist-data/train-labels-idx1-ubyte"), labels);
}

void read_train_images (char ** images) {
    read_images((PROJECT_ROOT "/mnist-data/train-images-idx3-ubyte"), images);
}


void read_test_labels (int ** labels) {
    read_labels((PROJECT_ROOT "/mnist-data/t10k-labels-idx1-ubyte"), labels);
}

void read_test_images (char ** images) {
    read_images((PROJECT_ROOT "/mnist-data/t10k-images-idx3-ubyte"), images);
}

int rv_lane_id() { return 0; }

#ifdef __cplusplus
}
#endif
