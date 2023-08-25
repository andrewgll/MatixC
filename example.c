#include "mx.h"

Matrix* gradient_descent(Matrix* X, Matrix* y, Matrix* W, double alpha, size_t iterations){
    int m = X->rows;
    Matrix *transform = MATRIX_WITH(2,1,1);
    double scaling_factor = -2.0 / m;
    for (size_t i = 0; i < iterations; i++) {
        Matrix *y_pred = mx_dot(X, W);
        Matrix* y_pred_transformed = mx_dot(y_pred, transform);
        Matrix *R = mx_subtract(y, y_pred_transformed);
        Matrix *X_transposed = mx_transpose(X);
        Matrix *gradient = mx_dot(X_transposed, R);
        Matrix *gradient_scaled = mx_scale(gradient, scaling_factor);
        Matrix *update = mx_scale(gradient_scaled, alpha);

        Matrix *new_weight = mx_subtract(W, update);
        mx_free(W);
        W = MATRIX_COPY(new_weight);
        mx_free(X_transposed);
        mx_free(new_weight);
        mx_print(W);
        mx_free(y_pred);
        mx_free(R);
        mx_free(y_pred_transformed);
        mx_free(gradient);
        mx_free(gradient_scaled);
        mx_free(update);
    }
    return W;
}


int main(void){

    //example of gradient descent and logical operators
    srand(6);
    // AND dataset
    Matrix* dataset = open_dataset("datasets/XOR");
    Matrix* X = COL_SLICE(dataset,0,1);
    Matrix* Y = COL_SLICE(dataset,2,2);
    Matrix* W = mx_rand(2,2);
    Matrix* result = gradient_descent(X,Y,W,0.1,20);
    
    for(size_t i = 0; i < X->rows; i++){
        Matrix* sample = ROW_SLICE(X,i,i);
        Matrix* result = mx_dot(sample, W);
        int value = 0;
        if(AT(result, 0, 0) > 0.5){
            value = 1;
        }
        printf("%f %f = %d\n", AT(sample,0,0), AT(sample,0,1), value);
    }
    
    mx_free(dataset);
    mx_free(X);
    mx_free(Y);
    mx_free(result);
    return 0;
}

