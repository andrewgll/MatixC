#ifndef mx
#define mx

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>

#ifdef USE_DOUBLE_PRECISION
    typedef double dtype;
#else
    typedef float dtype;
#endif
#define ARRAY_ROWS(arr) (sizeof(arr) / sizeof((arr)[0]))
#define ARRAY_COLS(arr) (sizeof(arr[0]) / sizeof(dtype))
#define VALID_DIMENSIONS(rows, cols) ((rows) > 0 && (cols) > 0)
#define VALID_MATRIX(matrix) \
    ((matrix) && (matrix)->container && (matrix)->container->data && VALID_DIMENSIONS((matrix)->rows, (matrix)->cols))

#define CHECK_MATRIX_VALIDITY(matrix) \
    do { \
        if (!VALID_MATRIX(matrix)) { \
            errno = EINVAL; \
            perror("Invalid matrix or matrix dimensions."); \
            return NULL; \
        } \
    } while(0)

#define AT(matrix, i, j) \
    *(CHECK_FLAG((matrix)->flags, 0) ? \
      &(dtype){(matrix)->default_value} : \
      &(matrix)->container->data[(i) * (matrix)->row_stride + (j) * (matrix)->col_stride])
#define MATRIX(rows, cols) __mx_init(NULL,rows, cols, 0)
#define MATRIX_FROM(array,rows,cols) __mx_init(array, rows,cols, 0)
#define MATRIX_VIEW(matrix) safe_mx_view(matrix)
#define MATRIX_ONES(rows,cols)  \
    ((rows) <= 0 || (cols) <= 0) ? \
    (errno = EINVAL, perror("Invalid matrix dimensions."), (Matrix*)NULL) : \
    mx_view(NULL,rows,cols,1)

/**
 * @brief Creates a deep copy of the given matrix.
 * 
 * This macro is a shorthand for the `mx_copy` function and creates a new matrix that is a
 * deep copy of the given `matrix`.
 *
 * @param matrix The matrix to be copied.
 * @return A pointer to the copied matrix.
 */
#define MATRIX_COPY(matrix) mx_copy(matrix)

/**
 * @brief Generates a matrix with random values.
 * 
 * This macro is a shorthand for the `mx_rand` function. It creates a new matrix of the
 * specified size and initializes it with random values.
 *
 * @param rows The number of rows for the new matrix.
 * @param cols The number of columns for the new matrix.
 * @return A pointer to the generated matrix with random values.
 */
#define MATRIX_RAND(rows, cols) mx_rand(rows,cols)

/**
 * @brief Generates a matrix initialized with a specific value.
 * 
 * This macro is a shorthand for the `__mx_init` function. It creates a new matrix of the 
 * specified size and initializes all elements with `init_value`.
 *
 * @param rows The number of rows for the new matrix.
 * @param cols The number of columns for the new matrix.
 * @param init_value The value with which all elements of the matrix should be initialized.
 * @return A pointer to the generated matrix initialized with `init_value`.
 */
#define MATRIX_WITH(rows, cols, init_value) __mx_init(NULL, rows, cols, init_value)

#define TRANSPOSE(matrix) mx_transpose(matrix, 1U<<0)
#define TRANSPOSE_VIEW(matrix) mx_transpose(matrix, 1U<<1)
#define TRANSPOSE_COPY(matrix) mx_transpose(matrix, 1U<<2)

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

/**
 * @brief Frees the memory of a matrix, taking shared data containers into account.
 *
 * If the matrix shares its data container with other matrices, 
 * the function decrements its reference count. The data is freed 
 * only when no other matrices reference it.
 *
 * @param matrix Pointer to the Matrix to be freed. No-op if NULL.
 */
void mx_free(Matrix *matrix);

/**
 * @brief Creates a deep copy of the given matrix.
 *
 * Allocates a new matrix and copies the contents of the source matrix 
 * into it. Returns NULL if allocation fails.
 *
 * @param src Pointer to the source Matrix to be copied.
 * @return Pointer to the copied Matrix or NULL if allocation failed.
 */
Matrix* mx_copy(const Matrix* src);

/**
 * @brief Applies a function element-wise to the given matrix.
 * 
 * Iterates over each element in the matrix and updates its value 
 * using the provided function.
 *
 * @param matrix Pointer to the Matrix whose elements are to be updated.
 * @param func Pointer to the function that defines the transformation.
 */
void* mx_apply_function(Matrix* matrix, dtype (*func)(dtype));

/**
 * @brief Initializes a new matrix container with a specified size.
 *
 * Allocates memory for a new matrix container and its data array.
 * Sets the reference count of the container to 1.
 *
 * @param size The number of elements in the matrix container's data array.
 * @return A pointer to the newly allocated matrix container or NULL if the 
 *         allocation failed.
 */
__matrix_container* __init_container(dtype* array,size_t size);


/**
 * @brief Initializes a matrix with the specified dimensions and initial value.
 *
 * Allocates memory for a matrix with given rows and columns. Each element of the
 * matrix is initialized with the provided initial value.
 *
 * @param rows The number of rows for the matrix.
 * @param cols The number of columns for the matrix.
 * @param init_value The initial value for each matrix element.
 * @return A pointer to the newly allocated matrix or NULL if allocation failed.
 */
Matrix* __mx_init(dtype* array, size_t rows, size_t cols, dtype init_value);

/**
 * @brief Creates a view of an existing matrix or initializes a lazy matrix.
 * 
 * If provided with a valid matrix, the function creates a view of that matrix,
 * increasing its container's reference count. If no matrix is provided,
 * it creates a lazy matrix with the specified dimensions and default value.
 *
 * @param matrix The matrix to create a view of. If NULL, a lazy matrix is initialized.
 * @param rows The number of rows for the view (or the lazy matrix).
 * @param cols The number of columns for the view (or the lazy matrix).
 * @param default_value The default value for matrix data.
 * @return A pointer to the matrix view or NULL if allocation failed.
 */
Matrix* mx_view(const Matrix* matrix, size_t cols, size_t rows, dtype default_value);

static inline Matrix* safe_mx_view(const Matrix* matrix) {
    if (!matrix) {
        perror("Error: Null matrix provided.");
        return NULL;
    }
    return mx_view(matrix, matrix->rows, matrix->cols, 0);
}

/**
 * @brief Initializes and returns an identity matrix of the given dimensions.
 * 
 * An identity matrix has ones on the main diagonal and zeros elsewhere.
 * If the matrix is not square, only the main diagonal up to the minimum of 
 * rows and columns will be filled with ones.
 *
 * @param rows The number of rows for the identity matrix.
 * @param cols The number of columns for the identity matrix.
 * @return A pointer to the identity matrix or NULL if dimensions are invalid or memory allocation failed.
 */
Matrix* mx_identity(size_t rows, size_t cols);

/**
 * @brief Determines if two matrices are equal.
 *
 * Two matrices are considered equal if they have the same dimensions 
 * and all their corresponding elements are equal.
 *
 * @param matrix1 The first matrix.
 * @param matrix2 The second matrix.
 * @return 1 if matrices are equal, 0 otherwise.
 */
uint8_t mx_equal(Matrix* matrix1, Matrix* matrix2);

/**
 * @brief Returns the transpose of a given matrix.
 * 
 * Transposes the input matrix based on the provided flags:
 * - If the first bit in flags is set, transpose in-place.
 * - If the third bit is set, return a transposed copy.
 * - Otherwise, return a transposed view.
 *
 * @param matrix The input matrix to be transposed.
 * @param flags Bit flags determining the transpose behavior.
 * @return The transposed matrix or the original matrix if transposed in-place.
 */
Matrix* mx_transpose(Matrix* matrix, uint8_t flags);

/**
 * Create a matrix of the given dimensions and populate it with
 * sequentially increasing values starting from `start_arrange`.
 * 
 * @param rows The number of rows for the matrix.
 * @param cols The number of columns for the matrix.
 * @param start_arrange The starting value for arranging the matrix elements.
 * @return Pointer to the newly created matrix, or NULL on failure.
 */
Matrix* mx_arrange(size_t rows, size_t cols, dtype start_arrange);

/**
 * Generates a matrix with random values between 0 and 1.
 *
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return A pointer to the matrix with random values or NULL if allocation fails.
 */
Matrix* mx_rand(size_t rows, size_t cols);

/**
 * Scales all elements of the given matrix by a scalar value.
 *
 * @param matrix The input matrix to be scaled.
 * @param scalar The value by which each element of the matrix is multiplied.
 * @return A new matrix with scaled values or NULL if memory allocation fails.
 */
Matrix* mx_scale(Matrix* matrix, dtype scalar);

/**
 * Adds the elements of two matrices element-wise.
 *
 * @param matrix1 The first matrix operand.
 * @param matrix2 The second matrix operand.
 * @return A new matrix with the summed values or NULL if dimensions mismatch or memory allocation fails.
 */
Matrix* mx_add(const Matrix* matrix1, const Matrix* matrix2);

/**
 * Subtracts the elements of the second matrix from the first one, element-wise.
 *
 * @param matrix1 The matrix from which values will be subtracted.
 * @param matrix2 The matrix whose values will be subtracted.
 * @return A new matrix with the difference or NULL if dimensions mismatch or memory allocation fails.
 */
Matrix* mx_subtract(const Matrix* matrix1, const Matrix* matrix2);

/**
 * @brief Multiplies two matrices. If the matrices are not directly compatible
 *        for multiplication, the function tries to transpose the second matrix
 *        to make it compatible.
 * 
 * The function first checks if the two matrices are compatible for multiplication.
 * If not, it checks if transposing the second matrix will make them compatible.
 * If neither of these conditions are met, the function returns an error.
 *
 * @param matrix1 Pointer to the first Matrix.
 * @param matrix2 Pointer to the second Matrix.
 * @return Pointer to the resulting Matrix after multiplication. If the matrices
 *         are not compatible and cannot be made compatible by transposing, 
 *         the function returns NULL.
 */
Matrix* mx_dot(Matrix* matrix1, Matrix* matrix2);

/**
 * @brief Extracts a submatrix (or slice) from a given source matrix based on specified start and end indices.
 * 
 * This function creates a new matrix using the given indices and copies the appropriate values from 
 * the source matrix to this new matrix. If invalid indices are provided (for example, if start_row is 
 * greater than end_row or if the indices go out of bounds of the source matrix), the function will return NULL.
 *
 * @param src Pointer to the source matrix.
 * @param start_row Starting row index for the slice.
 * @param end_row Ending row index for the slice.
 * @param start_col Starting column index for the slice.
 * @param end_col Ending column index for the slice.
 * @return Pointer to the new Matrix containing the slice. Returns NULL if indices are invalid.
 */
Matrix* mx_slice(const Matrix* src, size_t start_row, size_t end_row, size_t start_col, size_t end_col);

/*Not implemented*/ 
Matrix* mx_inverse(const Matrix* matrix);

Matrix* open_dataset(const char* name);
void* mx_print(const Matrix* matrix);

#endif // mx
