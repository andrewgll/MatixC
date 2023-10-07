#include "unity.h"
#include "../mx.h" 

void setUp(void) {
    // This is run before EACH test.
}

void tearDown(void) {
    // This is run after EACH test. Used for cleanup.
}
// Problem set page 65 
void test_problem_set_page_65_A(void){

    float arr[] = {
    1, 2,
    4, 5,
    7, 8,
    };
    float arr2[] = {
    1, 1, 0,
    0, 1, 1,
    1, 0, 1
    };
    Matrix* m1 = MATRIX_FROM(arr, 3,2   );
    Matrix* m2 = MATRIX_FROM(arr2, 3,3);
    // Cannot be calculated 
    TEST_ASSERT_NOT_EQUAL(m1->cols, m2->rows);
    mx_free(m1);
    mx_free(m2);
}

void test_problem_set_page_65_B_C(){
    printf("Page 65 problem b\n");
    float arr[] = {
    1, 2, 3,
    4, 5, 6,
    7, 8, 9
    };
    float arr2[] = {
    1, 1, 0,
    0, 1, 1,
    1, 0, 1
    };
    Matrix* m1 = MATRIX_FROM(arr, 3,3);
    Matrix* m2 = MATRIX_FROM(arr2, 3,3);
    MX_ASSERT(m1->cols == m2->rows);
    Matrix* result = MATRIX(m1->rows, m2->cols);
    // not using dot operator intionally
    for(size_t i = 0; i < m1->rows; ++i){
        for(size_t j = 0; j < m2->cols; ++j){
            float sum = 0;
            for(size_t k = 0; k < m1->cols; ++k){
                sum += AT(m1,i,k) * AT(m2, k, j);
            }
            AT(result, i,j)=sum;
        }
    }
    printf("Result of multiplication\n");
    PRINTM(m1);
    printf("and\n");
    PRINTM(m2);
    printf("is:\n");
    PRINTM(result);

    printf("Page 65 problem c\n");
    // Let's tests matrix dot product non-commutativity
    Matrix* m12 = MATRIX_FROM(arr2, 3,3);
    Matrix* m22 = MATRIX_FROM(arr, 3,3);
    MX_ASSERT(m12->cols == m22->rows);
    Matrix* result2 = MATRIX(m12->cols, m22->rows);
    // not using dot operator intionally
    for(size_t i = 0; i < m12->rows; ++i){
        for(size_t j = 0; j < m22->cols; ++j){
            float sum = 0;
            for(size_t k = 0; k < m12->cols; ++k){
                sum += AT(m12,i,k) * AT(m22, k, j);
            }
            AT(result2, i,j)=sum;
        }
    }
    
    printf("Result of multiplication\n");
    PRINTM(m12);
    printf("and\n");
    PRINTM(m22);
    printf("is:\n");
    PRINTM(result2);

    mx_free(m1);
    mx_free(m2);
    mx_free(m12);
    mx_free(m22);
    mx_free(result);
    mx_free(result2);
}

void test_problem_set_page_65_D_E(void){

    printf("Page 65 problem d\n");
    float arr[] = {
        1,2,1,2,
        4,1,-1,-4
    };
    float arr2[] ={
        0,3,
        1,-1,
        2,1,
        5,2
    };
    Matrix* m1 = MATRIX_FROM(arr,2,4);
    Matrix* m2 = MATRIX_FROM(arr2,4,2);

    Matrix* result = MATRIX(m1->rows, m2->cols);
    // OR just Matrix* result = DOT_COPY(m1,m2);
    for(size_t i = 0; i < m1->rows; ++i){
        for(size_t j = 0; j < m2->cols; ++j){
            float sum = 0;
            for(size_t k = 0; k < m1->cols; ++k){
                sum += AT(m1,i,k) * AT(m2, k, j);
            }
            AT(result, i,j)=sum;
        }
    }
    printf("Result of multiplication\n");
    PRINTM(m1);
    printf("and\n");
    PRINTM(m2);
    printf("is:\n");
    PRINTM(result);
    Matrix* m12 = MATRIX_FROM(arr2,4,2);
    Matrix* m22 = MATRIX_FROM(arr,2,4);

    Matrix* result2 = MATRIX(m12->rows, m22->cols);
    // OR just Matrix* result2 = DOT_COPY(m12,m22);
    for(size_t i = 0; i < m12->rows; ++i){
        for(size_t j = 0; j < m22->cols; ++j){
            float sum = 0;
            for(size_t k = 0; k < m12->cols; ++k){
                sum += AT(m12,i,k) * AT(m22, k, j);
            }
            AT(result2, i,j)=sum;
        }
    }
    
    printf("Result of multiplication\n");
    PRINTM(m12);
    printf("and\n");
    PRINTM(m22);
    printf("is:\n");
    PRINTM(result2);
    mx_free(m1);
    mx_free(m2);
    mx_free(m12);
    mx_free(m22);
    mx_free(result);
    mx_free(result2);
}

void test_problem_set_page_65_2_5_A(void) {
    float arr_A[] = {
        1,1,-1,-1,
        2,5,-7,-5,
        2,-1,1,3,
        5,2,-4,2
    };
    float arr_b[] = {
        1,
        -2,
        4,
        6
    };
    Matrix* A = MATRIX_FROM(arr_A,4,4);
    Matrix* b = MATRIX_FROM(arr_b,4,1);
    // to solv: x = A^-1 * b
    Matrix* A_inv = MATRIX(4, 4);
    Matrix* x = MATRIX(4, 1);
    uint8_t result = mx_inverse(A, A_inv);
    if (result == 1) {
        DOT(x, A_inv, b);

        // Print the solution vector x
        for (size_t i = 0; i < 4; ++i) {
            printf("x[%zu] = %f\n", i, AT(x, i, 0));
        }
    } else {
        printf("Matrix is singular or near-singular. Cannot find inverse.\n");
    }

    mx_free(A);
    mx_free(b);
    mx_free(A_inv);
    mx_free(x);
}
void test_problem_set_page_65_2_5_B(){
    float arr_A[] = {
        1,1,0,0,1,
        1,1,0,-3,0,
        2,-1,0,1,-1,
        -1,2,0,-2,-1
    };
    float arr_b[] = {
        3,
        6,
        5,
        -1
    };
    Matrix* A = MATRIX_FROM(arr_A,5,4);
    Matrix* b = MATRIX_FROM(arr_b,4,1);
    // to solv: x = A^-1 * b
    Matrix* A_inv = MATRIX(4, 4);
    Matrix* x = MATRIX(4, 1);
    uint8_t result = mx_inverse(A, A_inv);
    if (result == 1) {
        DOT(x, A_inv, b);

        // Print the solution vector x
        for (size_t i = 0; i < 4; ++i) {
            printf("x[%zu] = %f\n", i, AT(x, i, 0));
        }
    } else {
        printf("Matrix is singular or near-singular. Cannot find inverse.\n");
    }

    mx_free(A);
    mx_free(b);
    mx_free(A_inv);
    mx_free(x);

}

int main(void){
    UNITY_BEGIN();

    test_problem_set_page_65_A();
    test_problem_set_page_65_B_C();
    test_problem_set_page_65_D_E();
    // impossible to find solution, becasue A is singular
    test_problem_set_page_65_2_5_A();
    // impossible to find solution: A rows!=cols != b sizes
    test_problem_set_page_65_2_5_B();
    // 2.6 same issue
    
    return UNITY_END();


}
