#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "NNC.h"


const size_t DATASET_SIZE_1 = 4;
const size_t DATASET_SIZE_2 = 3;
const size_t FIRST_LAYER_SIZE = 2;

int main(int argc, char **argv){
    srand(6565);

    Matrix *matrix1 = MATRIX(3,2);
    Matrix *matrix2 = MATRIX(2,3);
    Matrix* dataset = open_dataset("dataset2");

    print_matrix(dataset);
    // print_matrix(transpose(dataset));

    return 0;
}
