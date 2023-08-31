#include "../mx.h"

int main(void){
    Matrix* m = MATRIX(3,1);
    AT(m,0,0) = 12;
    AT(m,1,0) = 23;
    AT(m,2,0) = 511;
    Matrix* m1 = MATRIX(3,1);
    AT(m1,0,0) = 9;
    AT(m1,1,0) = -1;
    AT(m1,2,0) = -123;
    dtype cosine = mx_cosine_between_two_vectors(m,m1);
    printf("%f", cosine);
    return 0;
}
