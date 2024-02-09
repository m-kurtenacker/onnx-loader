#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

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

void read_train_labels (int ** labels) {
    FILE * labels_file = fopen("mnist-data/train-labels-idx1-ubyte", "r");

    label_header header;

    for (int i = 0; i < 8; i++) {
        char d = fgetc(labels_file);
        int endian_fixed_address = i - (i % 4) + (3 - i % 4);
        header.data[endian_fixed_address] = d;
    }

    printf("%d %d\n", header.magic, header.length);

    assert(header.magic == 2049);

    *labels = (int*)malloc(sizeof(int) * header.length);
    for (int i = 0; i < header.length; i++) {
        char d = fgetc(labels_file);
        (*labels)[i] = d;
    }

    fclose(labels_file);
}

void read_test_labels (int ** labels) {
    FILE * labels_file = fopen("mnist-data/t10k-labels-idx1-ubyte", "r");

    label_header header;

    for (int i = 0; i < 8; i++) {
        char d = fgetc(labels_file);
        int endian_fixed_address = i - (i % 4) + (3 - i % 4);
        header.data[endian_fixed_address] = d;
    }

    printf("%d %d\n", header.magic, header.length);

    assert(header.magic == 2049);

    *labels = (int*)malloc(sizeof(int) * header.length);
    for (int i = 0; i < header.length; i++) {
        char d = fgetc(labels_file);
        (*labels)[i] = d;
    }

    fclose(labels_file);
}

void read_train_images (char ** images) {
    FILE * images_file = fopen("mnist-data/train-images-idx3-ubyte", "r");

    image_header header;

    for (int i = 0; i < 16; i++) {
        char d = fgetc(images_file);
        int endian_fixed_address = i - (i % 4) + (3 - i % 4);
        header.data[endian_fixed_address] = d;
    }

    printf("%d %d %d %d\n", header.magic, header.length, header.rows, header.cols);

    assert(header.magic == 2051);

    *images = (char*)malloc(sizeof(char) * header.length * header.rows * header.cols);
    for (int i = 0; i < header.length * header.rows * header.cols; i++) {
        char d = fgetc(images_file);
        (*images)[i] = d;
    }

    fclose(images_file);
}

void read_test_images (char ** images) {
    FILE * images_file = fopen("mnist-data/t10k-images-idx3-ubyte", "r");

    image_header header;

    for (int i = 0; i < 16; i++) {
        char d = fgetc(images_file);
        int endian_fixed_address = i - (i % 4) + (3 - i % 4);
        header.data[endian_fixed_address] = d;
    }

    printf("%d %d %d %d\n", header.magic, header.length, header.rows, header.cols);

    assert(header.magic == 2051);

    *images = (char*)malloc(sizeof(char) * header.length * header.rows * header.cols);
    for (int i = 0; i < header.length * header.rows * header.cols; i++) {
        char d = fgetc(images_file);
        (*images)[i] = d;
    }

    fclose(images_file);
}

int rv_lane_id() { return 0; }

void read_images(char * filename, char ** images) {
    FILE * images_file = fopen(filename, "r");

    image_header header;

    for (int i = 0; i < 16; i++) {
        char d = fgetc(images_file);
        header.data[i] = d;
    }

    printf("%d %d %d %d\n", header.magic, header.length, header.rows, header.cols);

    assert(header.magic == 2052);

    *images = (char*)malloc(sizeof(char) * header.length * header.rows * header.cols * 3);

    for (int i = 0; i < header.length * header.rows * header.cols * 3; i++) {
        char d = fgetc(images_file);
        (*images)[i] = d;
    }

    fclose(images_file);
}

#ifdef __cplusplus
}
#endif
