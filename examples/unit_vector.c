#include "../mx.h"

int main(void){
    Matrix* m = MATRIX_RAND(1,3);
    mx_print(m);
    Matrix* m_unit = UNIT_VECTOR_FROM(m);
    dtype length = mx_length(m_unit);
    printf("%f\n",length);
    mx_free(m);
    mx_free(m_unit);
    return 0;
}
