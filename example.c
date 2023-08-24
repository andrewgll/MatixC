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
        W = MATRIX_COPY(new_weight);
        free_mat(X_transposed);
        free_mat(new_weight);
        print_mat(W);
        free_mat(y_pred);
        free_mat(R);
        free_mat(gradient);
        free_mat(gradient_scaled);
        free_mat(update);
    }
    return W;
}


int main(){

    //example of gradient descent and logical operators
    srand(6);
    // AND dataset
    Matrix* dataset = open_dataset("datasets/AND");
    Matrix* X = COL_SLICE(dataset,0,1);
    Matrix* Y = COL_SLICE(dataset,2,2);
    Matrix* W = mat_rand(2,1);
    Matrix* result = gradient_descent(X,Y,W,0.1,20);
    
    for(size_t i = 0; i < X->rows; i++){
        Matrix* sample = ROW_SLICE(X,i,i);
        Matrix* result = mat_dot(sample, W);
        int value = 0;
        if(AT(result, 0, 0) > 0.5){
            value = 1;
        }
        printf("%f %f = %d\n", AT(sample,0,0), AT(sample,0,1), value);
    }
    
    free_mat(dataset);
    free_mat(X);
    free_mat(Y);
    free_mat(result);
    return 0;
}

