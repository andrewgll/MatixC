#include "../mx.h"
int main(void){    
    Matrix* matrix1 = mx_arrange(2, 3, 1);   // Produces [[1,2,3], [4,5,6]]
    Matrix* matrix2 = TRANSPOSE_VIEW(matrix1); // Should produce [[1,4], [2,5], [3,6]]
    Matrix* result = mx_dot(matrix1, matrix2);
    mx_free(matrix1);
    mx_free(matrix2);
    mx_print(result);
    mx_free(result);
}
