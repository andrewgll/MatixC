#include "../mx.h"
#include <time.h>


// Basic NN model with two layers
typedef struct{
    // input data
    Matrix* a0;

    //  layer 1
    Matrix* w1;
    Matrix* b1;
    Matrix* a1;
    
    // layer 2
    Matrix* w2;
    Matrix* b2;
    Matrix* a2;
    
}XOR;

XOR xor_alloc(void){
    XOR xor;
    xor.a0 = MATRIX(1,2);
    xor.w1 = MATRIX(2,2);
    xor.b1 = MATRIX(1,2);
    xor.a1 = MATRIX(1,2);
    xor.w2 = MATRIX(2,1);
    xor.b2 = MATRIX(1,1);
    xor.a2 = MATRIX(1,1);
    mx_set_to_rand(xor.w1,0,1);
    mx_set_to_rand(xor.b1,0,1);
    mx_set_to_rand(xor.w2,0,1);
    mx_set_to_rand(xor.b2,0,1);
    return xor;
}

float forward_xor(XOR *xor){
    mx_free(xor->a1);
    xor->a1 = DOT(xor->a0,xor->w1);
    ADD(xor->a1,xor->b1);
    mx_apply_function(xor->a1, sigmoidf);

    mx_free(xor->a2);
    xor->a2 = DOT(xor->a1, xor->w2);
    ADD(xor->a2,xor->b2);
    mx_apply_function(xor->a2, sigmoidf);
    return SCALAR(xor->a2);
}

float cost(XOR* m, Matrix* ti, Matrix* to){
    assert(ti->rows==to->rows);
    assert(to->cols == m->a2->cols); 
    size_t n = ti->rows;
    float c = 0;
    for(size_t i = 0; i< n; ++i){
        Matrix* x = ROW_SLICE(ti,i,i);
        Matrix* y = ROW_SLICE(to,i,i);
        mx_free(m->a0);
        m->a0 = x;
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

void finite_difference(XOR* m, XOR* g,float eps, Matrix* ti, Matrix* to){
    float saved;
    float c = cost(m, ti,to);

    for(size_t i=0; i< m->w1->rows;++i){
        for(size_t j = 0; j < m->w1->cols; ++j){
            saved = AT(m->w1,i,j);
            AT(m->w1,i,j) += eps;
            AT(g->w1,i,j) = (cost(m, ti, to)-c)/eps;
            AT(m->w1,i,j) = saved;            
        }
    }
    for(size_t i=0; i< m->b1->rows;++i){
        for(size_t j = 0; j < m->b1->cols; ++j){
            saved = AT(m->b1,i,j);
            AT(m->b1,i,j) += eps;
            AT(g->b1,i,j) = (cost(m, ti, to)-c)/eps;
            AT(m->b1,i,j) = saved;            
        }
    }
    for(size_t i=0; i< m->w2->rows;++i){
        for(size_t j = 0; j < m->w2->cols; ++j){
            saved = AT(m->w2,i,j);
            AT(m->w2,i,j) += eps;
            AT(g->w2,i,j) = (cost(m, ti, to)-c)/eps;
            AT(m->w2,i,j) = saved;            
        }
    }
    for(size_t i=0; i< m->b2->rows;++i){
        for(size_t j = 0; j < m->b2->cols; ++j){
            saved = AT(m->b2,i,j);
            AT(m->b2,i,j) += eps;
            AT(g->b2,i,j) = (cost(m, ti, to)-c)/eps;
            AT(m->b2,i,j) = saved;            
        }
    }
}

void learn(XOR* m, XOR* g, float rate){
    for(size_t i=0; i< m->w1->rows;++i){
        for(size_t j = 0; j < m->w1->cols; ++j){
            AT(m->w1,i,j) -= rate*AT(g->w1,i,j);
         }
    }
    for(size_t i=0; i< m->b1->rows;++i){
        for(size_t j = 0; j < m->b1->cols; ++j){
            AT(m->b1,i,j) -= rate*AT(g->b1,i,j);
         }
    }
    for(size_t i=0; i< m->w2->rows;++i){
        for(size_t j = 0; j < m->w2->cols; ++j){
            AT(m->w2,i,j) -= rate*AT(g->w2,i,j);
         }
    }
    for(size_t i=0; i< m->b2->rows;++i){
        for(size_t j = 0; j < m->b2->cols; ++j){
            AT(m->b2,i,j) -= rate*AT(g->b2,i,j);
        }
    }   
}

void xor_free(XOR* xor){
    mx_free(xor->a0);
    mx_free(xor->a1);
    mx_free(xor->a2);
    mx_free(xor->w1);
    mx_free(xor->w2);
    mx_free(xor->b1);
    mx_free(xor->b2);

}

int main(void){
    srand(time(NULL));
    
    XOR xor = xor_alloc();
    XOR gradient = xor_alloc();

    Matrix* xor_data = open_dataset("./datasets/XOR");
    Matrix* ti = COL_SLICE(xor_data,0,1);
    Matrix* to = COL_SLICE(xor_data,2,2);

    printf("cost = %f\n", cost(&xor,ti,to));
    float eps = 1e-1;
    float rate = 1;
    for(size_t i = 0; i<10000; ++i){
        finite_difference(&xor,&gradient,eps, ti, to);
        learn(&xor,&gradient,rate);
        printf("%zu cost = %f\n",i, cost(&xor,ti,to));
    }
    for(size_t i = 0; i < 2; ++i){
        for(size_t j = 0; j < 2; ++j){
            AT(xor.a0,0,0) = i;
            AT(xor.a0,0,1) = j;
            forward_xor(&xor);
            float y = SCALAR(xor.a2);
            printf("%zu ^ %zu = %f\n", i,j, round(y));
        }
    }
    mx_free(xor_data);
    mx_free(ti);
    mx_free(to);
    xor_free(&xor);
    xor_free(&gradient);
}
