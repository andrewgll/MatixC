#ifndef NNC
#define NNC

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define AT(matrix, i, j) ((matrix)->container->data[(i) * (matrix)->row_stride + (j) * (matrix)->col_stride])
#define MATRIX(rows, cols) mat_init(rows, cols, 0)
#define MATRIX_VIEW(matrix) mat_view(matrix)
#define MATRIX_WITH(rows, cols, init_value) mat_init(rows, cols, init_value)
#define ROW_SLICE(matrix,i,j) mat_slice(matrix,i,j,0,(matrix)->cols-1)
#define COL_SLICE(matrix,i,j) mat_slice(matrix, 0, (matrix)->rows-1, i, j)

double sigmoidf(double value);

typedef struct{
    size_t ref_count;
    double *data;
} __matrix_container;


typedef struct{
    __matrix_container *container;  // Points to the original matrix
    size_t size;
    size_t rows;
    size_t cols;
    size_t row_stride;
    size_t col_stride;
} Matrix;

void free_mat(Matrix *matrix);
Matrix* mat_init(size_t rows, size_t cols, double init_value);
Matrix* mat_view(const Matrix* matrix);
Matrix* mat_transpose(Matrix* matrix);
Matrix* mat_arrange(size_t rows, size_t cols, double start_arrange);
Matrix* mat_rand(size_t rows, size_t cols);
Matrix* mat_scale(Matrix* matrix, double scalar);
Matrix* mat_add(const Matrix* matrix1, const Matrix* matrix2);
Matrix* mat_subtract(const Matrix* matrix1, const Matrix* matrix2);
Matrix* mat_dot(const Matrix* matrix1, const Matrix* matrix2);
Matrix* open_dataset(const char* name);
Matrix* mat_slice(const Matrix* src, size_t start_row, size_t end_row, size_t start_col, size_t end_col);
void print_mat(const Matrix* matrix);

#endif // NNC
