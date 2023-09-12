#include "../mx.h"

int main(void){
    size_t arch_t[] = {4,3,2};
    NN* arch = NN(arch_t);
    PRINTNN(arch);
    mx_nn_free(arch);
    return 0;
}
