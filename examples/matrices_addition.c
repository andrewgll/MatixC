#include "../mx.h"

int main(void){
    float arr[] = {1,2,3};
    float arr2[] = {2,3,4};
    Matrix* matrix1 = MATRIX_FROM_ARRAY(arr);
    Matrix* matrix2 = MATRIX_FROM_ARRAY(arr2);

    Matrix* sum = ADD(matrix1, matrix2);
    PRINTM(sum);

    Matrix* u = MATRIX(1,2);
    AT(u,0,0) = -6;
    AT(u,0,1) = 8;

    
    Matrix* w = MATRIX(1,2);
    AT(w,0,0) = 8;
    AT(w,0,1) = 6;
    Matrix* uw_add = ADD(u,w);

    PRINTM(uw_add);
    return 0;
}
