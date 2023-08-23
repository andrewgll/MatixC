#include "nnc.h"

double gradient_descent(Matrix* X, Matrix* y, Matrix* W, double alpha, size_t iterations){
    int m = X->rows;
    double scaling_factor = -2.0 / m;
    for (int i = 0; i < iterations; i++) {
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

}


int main(int argc, char **argv){
    srand(6);
    // Matrix* dataset = open_dataset("datasets/dataset");
    // Matrix* X = COL_SLICE(dataset, 0,1);
    // Matrix* Y = COL_SLICE(dataset, 2,2);
    // Matrix* W = mat_rand(X->cols, 1);
    // print_mat(X);
    // print_mat(Y);
    // print_mat(W);
    // gradient_descent(X,Y,W,0.01,10);

    // free_mat(dataset);
    // free_mat(X);
    // free_mat(Y);
    // free_mat(W);
    Matrix* rand = mat_rand(3,3);
    Matrix* X = COL_SLICE(rand, 0,1);
    Matrix* Y = ROW_SLICE(rand, 0,1);

    free_mat(rand);
    free_mat(X);
    free_mat(Y);
    // Matrix* rand1 = mat_rand(3,3);
    // print_mat(rand);
    // print_mat(rand1);

    // Matrix* subt = mat_subtract(rand, rand1);
    // print_mat(subt);

    return 0;
}

