#include "../mx.h"
#include <time.h>

float forward_xor(NN *xor){
    for(size_t i = 0; i < xor->count; ++i){
        DOT(xor->as[i+1], xor->as[i],xor->ws[i]);
        ADD(xor->as[i+1],xor->bs[i]);
        mx_apply_function(xor->as[i+1], sigmoidf);
    }
    return SCALAR(xor->as[xor->count]);
}

float cost(NN* m, Matrix* ti, Matrix* to){
    assert(ti->rows==to->rows);
    assert(to->cols == m->as[m->count]->cols); 
    size_t n = ti->rows;
    float c = 0;
    for(size_t i = 0; i< n; ++i){
        Matrix* x = ROW_SLICE(ti,i,i);
        Matrix* y = ROW_SLICE(to,i,i);
        mx_free(m->as[0]);
        m->as[0] = x;
        float result = forward_xor(m);
        size_t q = to->cols;
        for(size_t j = 0; j < q; ++j){
            float d = result - AT(y,0,j);
            c += d*d;
        }
        mx_free(y);
    }
    return c/n;
}

void finite_difference(NN* m, NN* g,float eps, Matrix* ti, Matrix* to){
    float saved;
    float c = cost(m, ti,to);

    for(size_t d = 0; d < m->count; ++d){
        for(size_t i=0; i< m->ws[d]->rows;++i){
            for(size_t j = 0; j < m->ws[d]->cols; ++j){
                saved = AT(m->ws[d],i,j);
                AT(m->ws[d],i,j) += eps;
                AT(g->ws[d],i,j) = (cost(m, ti, to)-c)/eps;
                AT(m->ws[d],i,j) = saved;            
            }
        }
        for(size_t i=0; i< m->bs[d]->rows;++i){
            for(size_t j = 0; j < m->bs[d]->cols; ++j){
                saved = AT(m->bs[d],i,j);
                AT(m->bs[d],i,j) += eps;
                AT(g->bs[d],i,j) = (cost(m, ti, to)-c)/eps;
                AT(m->bs[d],i,j) = saved;            
            }
        }
    }
}

void learn(NN* m, NN* g, float rate){
    for(size_t d = 0; d < m->count; ++d){
        for(size_t i=0; i< m->ws[d]->rows;++i){
            for(size_t j = 0; j < m->ws[d]->cols; ++j){
                AT(m->ws[d],i,j) -= rate*AT(g->ws[d],i,j);
            }
        }
        for(size_t i=0; i< m->bs[d]->rows;++i){
            for(size_t j = 0; j < m->bs[d]->cols; ++j){
                AT(m->bs[d],i,j) -= rate*AT(g->bs[d],i,j);
            }
        } 
    }
}

int main(void){
    srand(time(NULL));

    size_t arch[] = {2,2,1};  // 2 input neuron 2 hidden 1 output
    NN* xor = NN(arch);         // initilize NN 
    NN* gradient = NN(arch);    
    mx_nn_set_to_rand(xor,0,1); // randomize weights and biases
    mx_nn_set_to_rand(gradient,0,1);
    Matrix* xor_data = open_dataset("./datasets/XOR"); // get data from dataset
    Matrix* ti = COL_SLICE(xor_data,0,1);       // Slice x
    Matrix* to = COL_SLICE(xor_data,2,2);       // Slice y

    float eps = 1e-1;
    float rate = 1;
    for(size_t i = 0; i<20000; ++i){
        finite_difference(xor,gradient,eps, ti, to);
        learn(xor,gradient,rate);
        printf("%zu cost = %f\n",i, cost(xor,ti,to));
    }
    for(size_t i = 0; i < 2; ++i){
        for(size_t j = 0; j < 2; ++j){
            AT(xor->as[0],0,0) = i;
            AT(xor->as[0],0,1) = j;
            forward_xor(xor);
            float y = SCALAR(xor->as[xor->count]);
            printf("%zu ^ %zu = %f\n", i,j, round(y));
        }
    }
    PRINTNN(xor);
    mx_nn_free(xor);
    mx_nn_free(gradient);
    mx_free(xor_data);
    mx_free(ti);
    mx_free(to);
}
