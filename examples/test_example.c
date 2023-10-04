#include "../mx.h"

float forward(NN* nn){
    for(size_t i = 0; i < nn->count; ++i){
        DOT(nn->as[i+1], nn->as[i], nn->ws[i]);
        ADD(nn->as[i+1], nn->bs[i]);
        mx_apply_function(nn->as[i+1], sigmoidf);
    }
    return SCALAR(nn->as[nn->count]);
}
int main(void){
    size_t arch[] = {1,1};
    NN* nn= NN(arch);
    NN* g= NN(arch);
    mx_nn_set_to_rand(nn,0,1);
    Matrix* m = open_dataset("./datasets/AND");
    Matrix* x = COL_SLICE(m,0,1);
    Matrix* y = COL_SLICE(m,2,2);

        return 0;
}

void gradient_descent(NN* nn, Matrix* X, Matrix* Y, float learning_rate, int num_iterations) {
    int m = X->rows;  // number of examples

    for (int iter = 0; iter < num_iterations; ++iter) {
        // Forward propagation
        nn->as[0] = X;
        forward(nn);

        // Compute the cost (MSE)
        Matrix* cost = SUBTRACT_TO_COPY(nn->as[1], Y);  // cost = A2 - Y
        // For MSE, the derivative dZ2 is also (A2 - Y)
        Matrix* dZ2 = cost;

        // dW2 = (1/m) * A1' * dZ2
        Matrix* dW2;
        DOT(dW2, nn->as[0], dZ2);
        // Scale dW2 by (1/m)
        SCALAR_DOT(dW2, 1.0/m);

        // dB2 = average of dZ2 across all examples
        Matrix* dB2 = AVERAGE(dZ2); // assuming AVERAGE function exists

        // dZ1 = dZ2 * W2' (transpose of W2)
        Matrix* dZ1;
        DOT(dZ2, nn->ws[1], dZ1);  // assuming DOT also handles transpose

        // dW1 = (1/m) * X' * dZ1
        Matrix* dW1;
        DOT(X, dZ1, dW1);
        // Scale dW1 by (1/m)
        SCALAR_DOT(dW1, 1.0/m);

        // dB1 = average of dZ1 across all examples
        Matrix* dB1 = AVERAGE(dZ1);

        // Update weights and biases
        SUBTRACT(nn->ws[0], dW1);  // in-place subtraction
        SUBTRACT(nn->bs[0], dB1);

        SUBTRACT(nn->ws[1], dW2);
        SUBTRACT(nn->bs[1], dB2);

        // Free temporary matrices
        MX_FREE(cost);
        MX_FREE(dW1);
        MX_FREE(dW2);
        MX_FREE(dB1);
        MX_FREE(dB2);
        MX_FREE(dZ1);
        MX_FREE(dZ2);
    }
}

