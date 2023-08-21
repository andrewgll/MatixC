#include "NNLinAlg.h"


int main(int argc, char **argv){
    srand(6565);

    Matrix *matrix1 = MATRIX(3,2);
    Matrix *matrix2 = MATRIX(2,3);
    Matrix* dataset = open_dataset("./datasets/dataset");

    Matrix* mat = rand_matrix(2,3);
    print_matrix(mat);
    return 0;
}
