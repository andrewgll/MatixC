#include "../mx.h"

int main(void){
    // Schwarz inequality |v * w| < ||v|||*||w||
    Matrix* rand1 = MATRIX(1,3);
    Matrix* rand2 = MATRIX(1,3);

    AT(rand1,0,0) = 1;
    AT(rand1,0,1) = 2;
    AT(rand1,0,2) = 4;

    AT(rand2,0,0) = 4;
    AT(rand2,0,1) = 2;
    AT(rand2,0,2) = 13;
    
    Matrix* dot = mx_dot(rand1,rand2);
    dtype length1 = mx_length(rand1);
    dtype length2 = mx_length(rand2);
    dtype result = length1*length2;
    // dot should be less 
    printf("%f\n", AT(dot,0,0));
    // as result ratio of dot/length1*length2 must be <= 1,
    // that's why cosine is never bigger then 1.
    // cos = |v * w| / ||v|||*||w||
    // if ratio is 0, then it is 90 degree angle.
    printf("%f\n", result);
}
