#include "../mx.h"

int main(void){
    dtype* array = (dtype*)malloc(sizeof(dtype)*3);
    array[0]=1;
    array[1]=2;
    array[2]=3;
    Matrix* m = MATRIX_FROM(array, 3,1);
    mx_print(m);
    mx_free(m);
    mx_print(m);
    return 0;
}
