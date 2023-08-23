#include "nnc.h"

Matrix* gradient_descent(Matrix* X, Matrix* y, Matrix* W, double alpha, size_t iterations){
    int m = X->rows;
    double scaling_factor = -2.0 / m;
    for (size_t i = 0; i < iterations; i++) {
        Matrix *y_pred = mat_dot(X, W);
        Matrix *R = mat_subtract(y, y_pred);

        Matrix *X_transposed = mat_transpose(X);
        Matrix *gradient = mat_dot(X_transposed, R);
        Matrix *gradient_scaled = mat_scale(gradient, scaling_factor);
        Matrix *update = mat_scale(gradient_scaled, alpha);

        Matrix *new_weight = mat_subtract(W, update);
        free_mat(W);
        W = NULL;
        W = MATRIX_VIEW(new_weight);
        print_mat(W);
        free_mat(y_pred);
        y_pred = NULL;
        free_mat(R);
        R=NULL;
        free_mat(X_transposed);
        X_transposed=NULL;
        free_mat(gradient);
        gradient=NULL;
        free_mat(gradient_scaled);
        gradient_scaled=NULL;
        free_mat(update);
        update=NULL;
        free_mat(new_weight);
        new_weight=NULL;
    }
    return W;
}

int main(){
    srand(6);
    Matrix* matrix = mat_arrange(3,3, 0);
    Matrix* matrix2 = mat_arrange(3,3, 0);
    Matrix* matrix3 = MATRIX_WITH(3,3,2);
    print_mat(matrix3);
    printf("%f ", AT(matrix3,0,1));
    printf("%ld ", sizeof(matrix->container->data[0]));
    printf("%ld ", sizeof(AT(matrix,0,0)));
    printf("%ld ", sizeof(dtype));
    dtype value = 1;
    dtype value2 = 2;
    printf("%f ", value-value2);
    return 0;
}

