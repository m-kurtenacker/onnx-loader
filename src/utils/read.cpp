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

    union {
        char data[4];
        int value;
    } magic;

    for (int i = 0; i < 4; i++) {
        char d = fgetc(images_file);
        int endian_fixed_address = i - (i % 4) + (3 - i % 4);
        magic.data[endian_fixed_address] = d;
    }

    int header_length = magic.data[0];

    union {
        char data[4];
        int value;
    } dimension [header_length];

    for (int i = 0; i < header_length; i++) {
        for (int j = 0; j < 4; j++) {
            char d = fgetc(images_file);
            int endian_fixed_address = j - (j % 4) + (3 - j % 4);
            dimension[i].data[endian_fixed_address] = d;
        }
    }

    int number_elements = 1;
    for (int i = 0; i < header_length; i++) {
        number_elements *= dimension[i].value;
    }

    printf("%d", magic.value);
    for (int i = 0; i < header_length; i++) {
        printf(" %d", dimension[i].value);
    }
    printf("\n");

    if (magic.value >> 8 == 0x8) {
        *images = (char*)malloc(sizeof(char) * number_elements);

        for (int i = 0; i < sizeof(char) * number_elements; i++) {
            char d = fgetc(images_file);
            (*images)[i] = d;
        }
    } else if (magic.value >> 8 == 0xd) {
        *images = (char*) malloc(sizeof(float) * number_elements);

        for (int i = 0; i < sizeof(float) * number_elements; i++) {
            char d = fgetc(images_file);
            int endian_fixed_address = i - (i % 4) + (3 - i % 4);
            (*images)[endian_fixed_address] = d;
        }
    } else {
        assert(false && "Cannot import this type right now!");
    }

    fclose(images_file);

    size_t * sizes = (size_t*) malloc(sizeof(size_t) * (header_length + 1));
    for (int i = 0; i < header_length; i++) {
        sizes[i] = dimension[i].value;
    }
    sizes[header_length] = 0;

    return sizes;
}

void write_idx(char * filename, size_t * sizes, char * images, char type) {
    FILE * images_file = fopen(filename, "w");

    size_t number_elements = 1;
    int num_dims = 0;
    while (sizes[num_dims]) {
        number_elements *= sizes[num_dims];
        num_dims++;
    }

    union {
        char data[4];
        int value;
    } magic;

    magic.value = type << 8;
    magic.data[0] = num_dims;

    for (int i = 0; i < 4; i++) {
        int endian_fixed_address = i - (i % 4) + (3 - i % 4);
        unsigned char d = magic.data[endian_fixed_address];
        fputc(d, images_file);
    }

    for (int i = 0; i < num_dims; i++) {
        union {
            char data[4];
            int value;
        } dimension;

        dimension.value = sizes[i];

        for (int j = 0; j < 4; j++) {
            int endian_fixed_address = j - (j % 4) + (3 - j % 4);
            unsigned char d = dimension.data[endian_fixed_address];
            fputc(d, images_file);
        }
    }

    if (magic.value >> 8 == 0x8) {
        for (int i = 0; i < number_elements; i++) {
            unsigned char d = images[i];
            fputc(d, images_file);
        }
    } else if (magic.value >> 8 == 0xd) {
        for (int i = 0; i < sizeof(float) * number_elements; i++) {
            int endian_fixed_address = i - (i % 4) + (3 - i % 4);
            unsigned char d = images[endian_fixed_address];
            fputc(d, images_file);
        }
    } else {
        assert(false && "Cannot write this type right now!");
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
