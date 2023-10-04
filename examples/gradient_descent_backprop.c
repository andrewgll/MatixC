#include "mx.h"

float forward(NN* nn){
    for(size_t i = 0; i < nn->count; ++i){
        DOT(nn->as[i+1], nn->as[i], nn->ws[i]);
        ADD(nn->as[i+1], nn->bs[i]);
        mx_apply_sigmoid(nn->as[i+1]);
    }
    return SCALAR(nn->as[nn->count]);
}

float dcost(NN* nn, NN* g, Matrix* ti, Matrix* to){
    assert(ti->rows==to->rows);
    assert(to->cols == nn->as[nn->count]->cols); 
    

}

int main(void){
    size_t arch[] = {2,2,1};
    NN* nn = NN(arch);
    NN* g = NN(arch);
    mx_nn_set_to_rand(nn,0,1);
    mx_nn_set_to_rand(g,0,1);
    Matrix* xor_data = open_dataset("./datasets/XOR");
    Matrix* x;
    COL_SLICE(x, xor_data, 0);
    Matrix* y;
    COL_SLICE(y, xor_data, 2);

    return 0;
}