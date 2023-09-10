#include "../mx.h"

int main(void){
    float* array = (float*)malloc(sizeof(float)*3);
    array[0]=4;
    array[1]=2;
    float* array2 = (float*)malloc(sizeof(float)*3);
    array2[0]=-1;
    array2[1]=2;
    Matrix* m = MATRIX_FROM(array, 2,1);
    Matrix* m2 = MATRIX_FROM(array2, 2,1);
    Matrix* m_transposed = TRANSPOSE_COPY(m);
    // perpendicular
    Matrix* mdotm2 = DOT(m_transposed,m2);
    PRINTM(m);
    PRINTM(m2);
    PRINTM(mdotm2);
    return 0;
}
