// tests/test_nnc.c
#include "nnc.h"
#include <assert.h>

void test_mat_init() {
    Matrix* m = MATRIX_WITH(2, 2, 1.0);
    assert(AT(m, 0, 0) == 1.0);
    assert(AT(m, 1, 1) == 1.0);
    free_mat(m);
}

void test_mat_random_init(){
    Matrix* m = mat_rand(3,3);
    assert(m->rows == 3);
    assert(m->cols == 3);
    free_mat(m);
}

void test_mat_view(){
    Matrix* m = MATRIX_WITH(2,2,1.0);
    Matrix* view = MATRIX_VIEW(m);
    assert(view->rows == m->rows);
    assert(view->cols == m->cols);
    assert(view->col_stride == m->col_stride);
    assert(view->size == m->size);
    assert(view->container->ref_count == m->container->ref_count);
    assert(view->container->ref_count == 2);
    assert(view->container->data[0] == m->container->data[0]);
    assert(view->container->data[0] == 1.0);
    free_mat(m);
    assert(view->container->ref_count == 1);
    assert(view->container->data[0] == 1.0);
    free_mat(view);
}

void test_mat_row_slice(){
    Matrix* m = MATRIX_WITH(3,3,1.0);
    Matrix* m_row_slice = ROW_SLICE(m,0,1);
    assert(m_row_slice->rows == 2);
    assert(m_row_slice->cols == 3);
    assert(m_row_slice->size == 6);
    assert(m_row_slice->container->data[0] == 1);
    assert(m_row_slice->container->data[m_row_slice->size-1] == 1);
    assert(m_row_slice->row_stride == 3);
    assert(m_row_slice->col_stride == 1);
    free_mat(m);
    free_mat(m_row_slice);
}

void test_mat_col_slice(){
    Matrix* m = mat_arrange(3,3,1);
    Matrix* slice = COL_SLICE(m,1,2);
    assert(slice->col_stride==1);
    assert(slice->row_stride==2);
    assert(slice->cols==2);
    assert(slice->rows==3);
    assert(slice->size==6);
    assert(slice->container->data[0]==2.0);
    assert(slice->container->data[1]==3.0);
    assert(slice->container->data[2]==5.0);
    assert(slice->container->data[3]==6.0);
    assert(slice->container->data[4]==8.0);
    assert(slice->container->data[5]==9.0);
    free_mat(m);
    free_mat(slice);
}


int main() {
    test_mat_init();
    test_mat_random_init();
    test_mat_row_slice();
    test_mat_col_slice();
    printf("All tests passed!\n");
    return 0;
}
