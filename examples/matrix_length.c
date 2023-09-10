#include "../mx.h"

int main(void){
    float* array = (float*)malloc(sizeof(float)*3);
    array[0]=1;
    array[1]=2;
    array[2]=3;
    Matrix* m = MATRIX_FROM(array,3,1);
    float length = mx_length(m);
    printf("%f", length);
    return 0;
}
