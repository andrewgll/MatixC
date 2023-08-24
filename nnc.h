#ifndef NNC
#define NNC

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
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
#define AT(matrix, i, j) ((matrix)->container->data[(i) * (matrix)->row_stride + (j) * (matrix)->col_stride])
#define MATRIX(rows, cols) mat_init(rows, cols, 0)
#define MATRIX_VIEW(matrix) mat_view(matrix)
#define MATRIX_COPY(matrix) mat_copy(matrix)
#define MATRIX_WITH(rows, cols, init_value) mat_init(rows, cols, init_value)
#define ROW_SLICE(matrix,i,j) mat_slice(matrix,i,j,0,(matrix)->cols-1)
#define COL_SLICE(matrix,i,j) mat_slice(matrix, 0, (matrix)->rows-1, i, j)

typedef struct{
    size_t ref_count;
    dtype *data;
} __matrix_container;


typedef struct{
    __matrix_container *container;  // Points to the original matrix
    size_t size;
    size_t rows;
    size_t cols;
    size_t row_stride;
    size_t col_stride;
} Matrix;

dtype sigmoidf(dtype value);
void free_mat(Matrix *matrix);
Matrix* mat_copy(const Matrix* src);
__matrix_container* init_container(size_t size);
Matrix* mat_init(size_t rows, size_t cols, dtype init_value);
Matrix* mat_view(const Matrix* matrix);
Matrix* mat_transpose(Matrix* matrix);
Matrix* mat_arrange(size_t rows, size_t cols, dtype start_arrange);
Matrix* mat_rand(size_t rows, size_t cols);
Matrix* mat_scale(Matrix* matrix, dtype scalar);
Matrix* mat_add(const Matrix* matrix1, const Matrix* matrix2);
Matrix* mat_subtract(const Matrix* matrix1, const Matrix* matrix2);
Matrix* mat_dot(const Matrix* matrix1, const Matrix* matrix2);
Matrix* mat_slice(const Matrix* src, size_t start_row, size_t end_row, size_t start_col, size_t end_col);
Matrix* open_dataset(const char* name);
void* print_mat(const Matrix* matrix);

#endif // NNC
