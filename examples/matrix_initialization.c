#include "../mx.h"

int main(void){
    dtype* array = (dtype*)malloc(sizeof(dtype)*3);
    array[0]=1;
    array[1]=2;
    array[2]=3;
    Matrix* m = MATRIX_FROM(array, 3,1);
    Matrix* d = mx_dot(m,m);
    mx_print(m);
    mx_print(d);
    mx_free(m);
    return 0;
}
