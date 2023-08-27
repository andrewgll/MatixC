#include "mx.h" 

dtype forward(Matrix* X, Matrix** NN, size_t nn_size){
    Matrix* y_pred = MATRIX_COPY(X);
    for(size_t j = 0; j < nn_size; j++){
        Matrix* new_pred = mx_dot(y_pred, NN[j]); // create a new prediction with the dot product
        mx_free(y_pred); // free the old prediction
        y_pred = new_pred; // update y_pred to point to the new prediction
        apply_function(y_pred, sigmoidf); // apply sigmoid to each element of matrix
    }
    return AT(y_pred,0,0);
}

Matrix** gradient_descent(Matrix* X, Matrix* Y, Matrix** NN, size_t nn_size, size_t iterations, double alpha){

    double scaling_factor = -2.0 / X->rows;
    for(size_t i = 0; i < iterations; i++){
        // 4x1
        Matrix* y_pred = MATRIX(X->rows,1);
        for(size_t j = 0; j< X->rows; j++){
            Matrix* X_batch = ROW_SLICE(X, j,j);
            AT(y_pred,j,0) = forward(X_batch, NN, nn_size);
            mx_free(X_batch);
        }

        
        // 4x1 - 4x1
        Matrix* R = mx_subtract(y_pred, Y);
        Matrix* X_batch_transposed = mx_transpose(X);
        // 2x4 4x1
        Matrix* gradient = mx_dot(X_batch_transposed,R);
        Matrix* update = mx_scale(gradient,scaling_factor*alpha);

        for(size_t j = 0; j < nn_size; j++){
            Matrix* new_w = mx_subtract(NN[j], update); // create a new prediction with the dot product
            mx_free(y_pred); // free the old prediction
            // y_pred = new_pred; // update y_pred to point to the new prediction
            
        }
        
    }
    return NN;

}


int main(void){

    //example of gradient descent and logical operators
    srand(611);
    // AND dataset
    Matrix* dataset = open_dataset("datasets/AND");
    Matrix* X = COL_SLICE(dataset,0,1);
    Matrix* Y = COL_SLICE(dataset,2,2);
    size_t nn_layers = 2;
    Matrix** NN = (Matrix**)malloc(sizeof(Matrix*)*nn_layers);
    NN[0] = MATRIX_RAND(2,2);
    NN[1] = MATRIX_RAND(2,1);
    // Matrix** result_matrix = gradient_descent(X,Y,NN, nn_layers, 200, 0.1);

    for(size_t i = 0; i < X->rows; i++){
        Matrix* batch = ROW_SLICE(X,i,i);
        dtype value =  forward(batch, NN, nn_layers);
        printf("%f %f = %f\n",AT(batch,0,0),AT(batch,0,1), value);
        mx_free(batch);
    }
    mx_free(dataset);
    mx_free(X);
    mx_free(Y);
    return 0;
}

