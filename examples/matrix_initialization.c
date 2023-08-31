#include "../mx.h"

int main(void){
    dtype* array = (dtype*)malloc(sizeof(dtype)*3);
    array[0]=4;
    array[1]=2;
    dtype* array2 = (dtype*)malloc(sizeof(dtype)*3);
    array2[0]=-1;
    array2[1]=2;
    Matrix* m = MATRIX_FROM(array, 2,1);
    Matrix* m2 = MATRIX_FROM(array2, 2,1);
    Matrix* m_transposed = TRANSPOSE_COPY(m);
    // perpendicular
    Matrix* mdotm2 = mx_dot(m_transposed,m2);
    mx_print(m);
    mx_print(m2);
    mx_print(mdotm2);
    return 0;
}
