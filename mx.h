#ifndef MX_H_
#define MX_H_

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <pthread.h>

#ifdef USE_DOUBLE_PRECISION
    typedef double precision_type;
#else
    typedef float precision_type;
#endif

#ifndef MX_ASSERT
#include <assert.h>
#define MX_ASSERT assert
#endif // MX_ASSERT

#ifndef MX_FREE
#define MX_FREE free
#endif

#ifndef THREAD_COUNT
#define THREAD_COUNT 8 
#endif

#ifndef MX_MALLOC
#define MX_MALLOC malloc
#endif // MX_MALLOC

#define ARRAY_ROWS(arr) (sizeof(arr) / sizeof((arr)[0]))
#define ARRAY_COLS(arr) (sizeof(arr[0]) / sizeof(float))
#define VALID_DIMENSIONS(rows, cols) ((rows) > 0 && (cols) > 0)
#define VALID_MATRIX(matrix) \
    ((matrix) && (matrix)->container && (matrix)->container->data && VALID_DIMENSIONS((matrix)->rows, (matrix)->cols))
#define CHECK_MATRIX_VALIDITY(matrix) matrix_is_valid(matrix)
#define AT(matrix, i, j) \
    (matrix)->container->data[(i) * (matrix)->row_stride + (j) * (matrix)->col_stride]
/**
 * @brief Allocates a matrix with rows and cols size. 
 * also allocates a memory for matrix container with size rows x cols
*/
#define MATRIX(rows, cols) __mx_init(NULL,rows, cols, 0)
#define MATRIX_FROM(array,rows,cols) __mx_init(array, rows,cols, 0)
#define MATRIX_FROM_ARRAY(array) MATRIX_FROM(array, ARRAY_ROWS(array), ARRAY_COLS(array))
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
 * @brief Initializes and returns an identity matrix of the given dimensions.
 * 
 * An identity matrix has ones on the main diagonal and zeros elsewhere.
 *
 * @param rows The number of rows for the identity matrix.
 * @return A pointer to the identity matrix or NULL if dimensions are invalid or memory allocation failed.
 */
#define MATRIX_IDENTITY(rows) mx_identity_new(rows)
#define MATRIX_DIAGONAL(rows,value) mx_diagonal_new(rows,value)

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
#define MATRIX_RAND(rows,cols) mx_rand_alloc(rows, cols)

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

/**
 * Allocates memory for a neural network (NN) with the given architecture.
 *
 * @param arch: An array representing the architecture of the NN. 
 *              Each element specifies the number of neurons in each layer.
 * @param arch_count: The number of layers in the NN.
 * @return: An initialized NN structure.
 */
#define NN(arch) __mx_nn_alloc(arch, ARRAY_ROWS(arch))

#define TRANSPOSE(matrix) mx_transpose(matrix, 1U<<0)
#define TRANSPOSE_VIEW(matrix) mx_transpose(matrix, 1U<<1)
#define TRANSPOSE_NEW(matrix) mx_transpose(matrix, 1U<<2)

#define ROW_SLICE(matrix,i,j) mx_slice(matrix,i,j,0,(matrix)->cols-1)
#define COL_SLICE(matrix,i,j) mx_slice(matrix, 0, (matrix)->rows-1, i, j)

#define UNIT_VECTOR_FROM(matrix) mx_unit_vector_from(matrix)
#define UNIT_VECTOR(size) mx_identity_new(size)

#define SCALAR(matrix) AT(matrix,0,0)
#define AVERAGE(matrix) mx_average(matrix)
#define SAFE_DOT(matrix1, matrix2) mx_dot_new(matrix1, matrix2, 0, 1U<<0)

/**
 * @brief Computes the dot product of two matrices.
 * 
 * This function is designed for performance optimization.
 * 
 * Important Note:
 * The caller MUST ensure that the matrices have compatible dimensions for multiplication 
 * The function does not perform any boundary checks, so using it without the correct preconditions 
 * can lead to undefined behavior or potential memory corruption.
 * 
 * If you need safe dot_product without optimizations use SAFE_DOT instead.
 * 
 * @param src       A pointer to the Matrix where the result is stored.
 * @param dst1      A pointer to the first Matrix operand.
 * @param dst2      A pointer to the second Matrix operand.
 * 
 */
#define DOT(src, dst1, dst2) mx_fast_dot(src, dst1, dst2)
#define SCALAR_DOT(matrix, scalar_value) mx_dot_new(matrix, NULL, scalar_value, 1U<<1)
#define ADD(matrix1, matrix2) APPLY_TO_BOTH(matrix1,matrix2, __add_elements)
#define ADD_NEW(matrix1, matrix2) APPLY_TO_BOTH_NEW(matrix1,matrix2, __add_elements)
#define SUBTRACT(matrix1,matrix2) APPLY_TO_BOTH_INPLACE(matrix1, matrix2, __subtract_elements)
#define SUBTRACT_NEW(matrix1,matrix2) APPLY_TO_BOTH_NEW(matrix1, matrix2, __subtract_elements)

#define APPLY_TO_BOTH(matrix1, matrix2, function) mx_apply_function_to_both(matrix1, matrix2, function)
#define APPLY_TO_BOTH_NEW(matrix1, matrix2, function) mx_apply_function_to_both_new(matrix1, matrix2, function)

#define SET_FLAG(f, index)   ((f) |= (1U << (index)))
#define CLEAR_FLAG(f, index) ((f) &= ~(1U << (index)))
#define CHECK_FLAG(f, index) (((f) & (1U << (index))) != 0)

#define PRINTM(matrix) mx_print(matrix, #matrix, 0)
#define PRINTM_PADDING(matrix, padding) mx_print(matrix, #matrix, padding)
#define PRINTNN(nn) mx_nn_print(nn, #nn)

typedef struct{
    uint16_t ref_count;
    size_t size;
    precision_type *data;
} __matrix_container;

typedef struct{
    uint8_t flags; // lazy_mat, ...
    uint16_t rows;
    uint16_t cols;
    uint16_t row_stride;
    uint16_t col_stride;
    precision_type default_value;
    __matrix_container *container;  // Points to the original matrix
} Matrix;

typedef struct {
    const Matrix *dst1;
    const Matrix *dst2;
    Matrix *src;
    size_t start_row;
    size_t end_row;
} ThreadData;

/**
 * Neural Network (NN) Structure
 * Represents a feed-forward neural network.
 */
typedef struct {

    size_t count;       /**< Total number of layers in the neural network minus one.
                             This represents the number of weight matrices, bias vectors, 
                             and activation matrices excluding the input layer. */

    Matrix** ws;        /**< Pointer to an array of weight matrices.
                             Each matrix in this array represents the weights between
                             two consecutive layers in the neural network. */

    Matrix** bs;        /**< Pointer to an array of bias vectors.
                             Each matrix in this array (typically 1 x N dimension) 
                             represents the biases for each neuron in a layer. */

    Matrix** as;        /**< Pointer to an array of activation matrices.
                             Each matrix in this array holds the activation values 
                             for each neuron in a layer, after applying the activation function. */

} NN;

float sigmoidf(float value);
float __add_elements(float a, float b);
float __subtract_elements(float a, float b); 
void swap(float *a, float *b);
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
void mx_apply_function(Matrix* matrix, float (*func)(float));

void mx_apply_sigmoid(Matrix* matrix);

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
__matrix_container* __init_container(float* array,size_t size);


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
Matrix* __mx_init(float* array, size_t rows, size_t cols, float init_value);

NN* __mx_nn_alloc(size_t* arch, size_t arch_count);

void mx_set_to_rand(Matrix* m, float min, float max);

void mx_nn_set_to_rand(NN* nn, float min, float max);

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
Matrix* mx_view(const Matrix* matrix, size_t cols, size_t rows, float default_value);

static inline Matrix* safe_mx_view(const Matrix* matrix) {
    if (!matrix) {
        perror("Error: Null matrix provided.");
        return NULL;
    }
    return mx_view(matrix, matrix->rows, matrix->cols, 0);
}

static inline int8_t matrix_is_valid(const Matrix* matrix) {
    if (!VALID_MATRIX(matrix)) { 
        errno = EINVAL; 
        perror("Invalid matrix or matrix dimensions."); 
        return -1; 
    } 
    return 1;
}

/**
 * Calculate the Frobenius norm (or 'length') of a matrix.
 * The Frobenius norm of a matrix A is the square root of the sum of the absolute squares of its elements.
 *
 * For a matrix A of dimensions MxN, the formula is:
 * ||A||_F = sqrt(sum(sum(A_ij^2)))
 *
 * @param matrix The matrix whose length (or norm) is to be computed.
 * @return The Frobenius norm of the matrix. Returns -1 if there's an error during computation.
 */
float mx_length(const Matrix* matrix);

/**
 * @brief Initializes and returns an identity matrix of the given dimensions.
 * 
 * An identity matrix has ones on the main diagonal and zeros elsewhere.
 * If the matrix is not square, only the main diagonal up to the minimum of 
 * rows and columns will be filled with ones.
 *
 * @param rows The number of rows for the identity matrix(rows=cols).
 * @return A pointer to the identity matrix or NULL if dimensions are invalid or memory allocation failed.
 */
Matrix* mx_identity_new(size_t rows);

Matrix* mx_diagonal_new(size_t rows, float value);

/**
 * @brief Compute the cosine of the angle between two vectors.
 * 
 * This function calculates the cosine of the angle between two matrices (vectors) 
 * using the formula: cos(theta) = dot(matrix1, matrix2) / (||matrix1|| * ||matrix2||)
 *
 * @param matrix1 First vector.
 * @param matrix2 Second vector.
 * @return The cosine of the angle between the two vectors.
 */
float mx_cosine_between_two_vectors(Matrix* matrix1, Matrix* matrix2);


/**
 * Generate a unit vector from the provided matrix (vector).
 * 
 * @param matrix Input matrix. Must be a row or column vector.
 * @return Unit vector matrix, NULL if the input is invalid or not a vector.
 */
Matrix* mx_unit_vector_from(const Matrix* matrix);

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
Matrix* mx_arrange_alloc(size_t rows, size_t cols, float start_arrange);

/**
 * Generates a matrix with random values between 0 and 1.
 *
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return A pointer to the matrix with random values or NULL if allocation fails.
 */
Matrix* mx_rand_alloc(size_t rows, size_t cols);

/**
 * --DEPRECATED--
 * Scales all elements of the given matrix by a scalar value.
 *
 * @param matrix The input matrix to be scaled.
 * @param scalar The value by which each element of the matrix is multiplied.
 * @return A new matrix with scaled values or NULL if memory allocation fails.
 */
Matrix* mx_scale(Matrix* matrix, float scalar);

/**
 * Adds the elements of two matrices element-wise.
 *
 * @param matrix1 The first matrix operand.
 * @param matrix2 The second matrix operand.
 * @return A new matrix with the summed values or NULL if dimensions mismatch or memory allocation fails.
 */
Matrix* mx_add(Matrix* matrix1, Matrix* matrix2, uint8_t flags);

uint8_t mx_apply_function_to_both(Matrix* matrix1,Matrix* matrix2, float (*func)(float, float));

Matrix* mx_apply_function_to_both_new(Matrix* matrix1,Matrix* matrix2, float (*func)(float, float));
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
Matrix* mx_dot_new(const Matrix* matrix1, const Matrix* matrix2, float scalar, uint8_t flags);

/**
 * @brief Computes the dot product of two matrices.
 * 
 * This function is designed for performance optimization by employing loop unrolling techniques.
 * 
 * Important Note:
 * The caller MUST ensure that the matrices have compatible dimensions for multiplication 
 * (i.e., dst1's number of columns must equal dst2's number of rows). 
 * Moreover, the src matrix must be of the appropriate size to store the result 
 * (i.e., dst1's number of rows by dst2's number of columns).
 * The function does not perform any boundary checks, so using it without the correct preconditions 
 * can lead to undefined behavior or potential memory corruption.
 * 
 * @param src       A pointer to the Matrix where the result is stored.
 * @param dst1      A pointer to the first Matrix operand.
 * @param dst2      A pointer to the second Matrix operand.
 * 
 */
void mx_fast_dot(const Matrix *src, const Matrix *dst1, const Matrix *dst2);
/**
 * Returns a perpendicular vector to the given 2D or 3D matrix-vector.
 *
 * @param matrix: A pointer to a 2D or 3D Matrix in vector form (either row or column).
 */
Matrix* mx_perpendicular_new(const Matrix* matrix);

/**
 * @brief Computes the dot product of a vector with itself.
 * 
 * This function calculates the dot product (or scalar product) of a vector with 
 * itself, effectively returning the squared magnitude of the vector. The input
 * should be a 1D matrix, representing either a row or column vector.
 *
 * @param vector   Pointer to a Matrix structure representing a 1D vector. 
 *                 The matrix should have either one row or one column.
 * 
 * @return The scalar result of the dot product. 
 *         Returns -1 (or any other predefined error value) if:
 *         - The matrix is invalid.
 *         - The provided matrix is not a 1D vector (i.e., neither a row 
 *           nor a column vector).
 * 
 * @note The function assumes that the matrix is well-formed and 
 *       checks for matrix validity. For invalid matrices, it sets
 *       the global 'errno' to EINVAL.
 */
float mx_self_dot_product(Matrix* vector);

Matrix* mx_cross_product_alloc(const Matrix* A, const Matrix* B);
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

uint8_t mx_inverse(Matrix *input, Matrix *output);
Matrix* open_dataset(const char* name);
void mx_nn_free(NN* nn);
uint8_t mx_print(const Matrix* matrix, const char* name, size_t padding);
void mx_nn_print(const NN* nn, const char* name);

#endif // MX_H_
