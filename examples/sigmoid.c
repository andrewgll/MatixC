#include "../mx.h"

int main(void){
    float array[] = {0,2,-2,4,-4};
    Matrix* mat = MATRIX_FROM_ARRAY(array);
    mx_apply_function(mat, sigmoidf);
    PRINTM(mat);
    return 0;
}
