#ifndef mx
#define mx

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#ifdef USE_DOUBLE_PRECISION
    typedef double dtype;
#else
    typedef float dtype;
#endif
#define VALID_DIMENSIONS(rows, cols) ((rows) > 0 && (cols) > 0)
#define VALID_MATRIX(matrix) \
    (matrix && VALID_DIMENSIONS(matrix->rows, matrix->cols))

#define CHECK_MATRIX_VALIDITY(matrix) \
    do { \
        if (!VALID_MATRIX(matrix)) { \
            perror("Invalid matrix dimensions."); \
            return NULL; \
        } \
    } while(0)
#define AT(matrix, i, j) \
    *(CHECK_FLAG((matrix)->flags, 1) ? \
      &(dtype){(matrix)->default_value} : \
      &(matrix)->container->data[(i) * (matrix)->row_stride + (j) * (matrix)->col_stride])
#define MATRIX(rows, cols) mx_init(rows, cols, 0)
#define MATRIX_VIEW(matrix) mx_view(matrix,0,0,0)
#define MATRIX_ONES(rows,cols) mx_view(NULL,rows,cols,1)
#define MATRIX_COPY(matrix) mx_copy(matrix)
#define MATRIX_RAND(rows, cols) mx_rand(rows,cols)
#define MATRIX_WITH(rows, cols, init_value) mx_init(rows, cols, init_value)
#define ROW_SLICE(matrix,i,j) mx_slice(matrix,i,j,0,(matrix)->cols-1)
#define COL_SLICE(matrix,i,j) mx_slice(matrix, 0, (matrix)->rows-1, i, j)

#define SET_FLAG(f, index)   ((f) |= (1U << (index)))
#define CLEAR_FLAG(f, index) ((f) &= ~(1U << (index)))
#define CHECK_FLAG(f, index) (((f) & (1U << (index))) != 0)

typedef struct{
    uint16_t ref_count;
    dtype *data;
} __matrix_container;

typedef struct{
    uint8_t flags; // lazy_mat, ...
    uint16_t rows;
    uint16_t cols;
    uint16_t row_stride;
    uint16_t col_stride;
    dtype default_value;
    __matrix_container *container;  // Points to the original matrix
} Matrix;

dtype sigmoidf(dtype value);

void mx_free(Matrix *matrix);
Matrix* mx_copy(const Matrix* src);
__matrix_container* init_container(size_t size);
Matrix* mx_init(size_t rows, size_t cols, dtype init_value);
Matrix* mx_view(const Matrix* matrix, size_t cols, size_t rows, size_t default_value);
Matrix* mx_identity(size_t rows, size_t cols);
bool mx_equal(Matrix* matrix1, Matrix* matrix2);

Matrix* mx_transpose(Matrix* matrix);
Matrix* mx_arrange(size_t rows, size_t cols, dtype start_arrange);
Matrix* mx_rand(size_t rows, size_t cols);
Matrix* mx_scale(Matrix* matrix, dtype scalar);
Matrix* mx_add(const Matrix* matrix1, const Matrix* matrix2);
Matrix* mx_subtract(const Matrix* matrix1, const Matrix* matrix2);
Matrix* mx_dot(const Matrix* matrix1, const Matrix* matrix2);
Matrix* mx_slice(const Matrix* src, size_t start_row, size_t end_row, size_t start_col, size_t end_col);
Matrix* mx_inverse(const Matrix* matrix);

Matrix* open_dataset(const char* name);
void* mx_print(const Matrix* matrix);

#endif // mx
