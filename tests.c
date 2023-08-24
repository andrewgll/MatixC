#include "unity.h"
#include "nnc.h" 

#ifdef USE_DOUBLE_PRECISION
    #define TEST_ASSERT_EQUAL_DTYPE(expected, actual) TEST_ASSERT_EQUAL_DOUBLE(expected, actual)
    #define TEST_ASSERT_DTYPE_WITHIN(delta, expected, actual) TEST_ASSERT_DOUBLE_WITHIN(delta, expected, actual)
#else
    #define TEST_ASSERT_EQUAL_DTYPE(expected, actual) TEST_ASSERT_EQUAL_FLOAT(expected, actual)
    #define TEST_ASSERT_DTYPE_WITHIN(delta, expected, actual) TEST_ASSERT_FLOAT_WITHIN(delta, expected, actual)
#endif

void setUp(void) {
    // This is run before EACH test.
}

void tearDown(void) {
    // This is run after EACH test. Used for cleanup.
}

void test_mat_arrange_memory_layout() {
    size_t rows = 3;
    size_t cols = 3;
    Matrix *m = mat_arrange(rows, cols, 0);

    TEST_ASSERT_NOT_NULL(m);
    TEST_ASSERT_NOT_NULL(m->container);
    TEST_ASSERT_NOT_NULL(m->container->data);

    free_mat(m);
}

void test_mat_arrange_correctness() {
    size_t rows = 3;
    size_t cols = 3;
    double start_val = 5.0;
    Matrix *m = mat_arrange(rows, cols, start_val);

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            TEST_ASSERT_EQUAL_DTYPE(start_val, AT(m, i, j));
            start_val++;
        }
    }

    free_mat(m);
}


void test_basic_free(void)
{
    Matrix *mat = malloc(sizeof(Matrix));
    mat->container = malloc(sizeof(__matrix_container));
    mat->container->data = malloc(10 * sizeof(dtype));
    mat->container->ref_count = 1;

    free_mat(mat);

    // If we've reached here, the basic free worked without crashing.
    // (We're not checking the memory content because it's undefined after free)
}

void test_ref_count_free(void)
{
    Matrix *mat = MATRIX(5,2);
    mat->container->ref_count = 2;
    __matrix_container* ct = mat->container;

    free_mat(mat);

    // The matrix's container and data shouldn't be freed since ref_count > 1
    TEST_ASSERT_NOT_NULL(ct);
    TEST_ASSERT_NOT_NULL(ct->data);
    TEST_ASSERT_EQUAL_INT(1, ct->ref_count);

    // Cleanup
    free(ct->data);
    free(ct);
}

void test_data_check(void)
{
    Matrix *mat = malloc(sizeof(Matrix));
    mat->container = malloc(sizeof(__matrix_container));
    mat->container->data = malloc(10 * sizeof(dtype));
    mat->container->ref_count = 1;

    free_mat(mat);

    // Can't check mat->container->data since it's freed.
    // If we've reached here without crashes, we're good.
}

void test_container_check(void)
{
    Matrix *mat = malloc(sizeof(Matrix));
    mat->container = malloc(sizeof(__matrix_container));
    mat->container->data = malloc(10 * sizeof(dtype));
    mat->container->ref_count = 1;

    free_mat(mat);

    // Can't check mat->container since it's freed.
    // If we've reached here without crashes, we're good.
}


void test_basic_matrix_initialization(void)
{
    Matrix *mat = MATRIX_WITH(5, 5, 1.0);

    TEST_ASSERT_NOT_NULL(mat);
    TEST_ASSERT_NOT_NULL(mat->container);
    TEST_ASSERT_NOT_NULL(mat->container->data);
    TEST_ASSERT_EQUAL_INT(5, mat->rows);
    TEST_ASSERT_EQUAL_INT(5, mat->cols);
    TEST_ASSERT_EQUAL_INT(25, mat->size);
    TEST_ASSERT_EQUAL_INT(1, mat->container->ref_count);
    TEST_ASSERT_EQUAL_INT(5, mat->row_stride);
    TEST_ASSERT_EQUAL_INT(1, mat->col_stride);

    for(size_t i = 0; i < mat->rows; i++){
        for(size_t j = 0; j < mat->cols; j++){
            TEST_ASSERT_EQUAL_DTYPE(1.0, AT(mat, i, j));
        }
    }

    free_mat(mat);
}

void test_matrix_initialization_with_value(void)
{
    dtype value = 7.0;
    Matrix *mat = MATRIX_WITH(3, 3, value);

    TEST_ASSERT_NOT_NULL(mat);
    TEST_ASSERT_NOT_NULL(mat->container);
    TEST_ASSERT_NOT_NULL(mat->container->data);
    TEST_ASSERT_EQUAL_INT(3, mat->rows);
    TEST_ASSERT_EQUAL_INT(3, mat->cols);
    TEST_ASSERT_EQUAL_INT(9, mat->size);
    TEST_ASSERT_EQUAL_INT(1, mat->container->ref_count);
    TEST_ASSERT_EQUAL_INT(3, mat->row_stride);
    TEST_ASSERT_EQUAL_INT(1, mat->col_stride);

    for(size_t i = 0; i < mat->rows; i++){
        for(size_t j = 0; j < mat->cols; j++){
            TEST_ASSERT_EQUAL_DTYPE(value, AT(mat, i, j));
        }
    }

    free_mat(mat);
}

void test_matrix_indexing(void)
{
    Matrix *mat = MATRIX(5, 5);
    dtype value = 5.0;

    // Setting values using the AT macro
    for(size_t i = 0; i < mat->rows; i++){
        for(size_t j = 0; j < mat->cols; j++){
            AT(mat, i, j) = value;
        }
    }

    // Checking values using the AT macro
    for(size_t i = 0; i < mat->rows; i++){
        for(size_t j = 0; j < mat->cols; j++){
            TEST_ASSERT_EQUAL_DTYPE(value, AT(mat, i, j));
        }
    }

    free_mat(mat);
}

void test_init_container_successful_allocation(void) {
    size_t size = 10;
    __matrix_container* container = init_container(size);
    
    TEST_ASSERT_NOT_NULL(container);
    TEST_ASSERT_NOT_NULL(container->data);
    TEST_ASSERT_EQUAL_INT(1, container->ref_count);
    
    free(container->data);
    free(container);
}

void test_mat_init_successful_allocation(void) {
    size_t rows = 3;
    size_t cols = 3;
    dtype init_value = 5.0;
    
    Matrix* mat = mat_init(rows, cols, init_value);
    
    TEST_ASSERT_NOT_NULL(mat);
    TEST_ASSERT_NOT_NULL(mat->container);
    TEST_ASSERT_NOT_NULL(mat->container->data);
    TEST_ASSERT_EQUAL_INT(rows, mat->rows);
    TEST_ASSERT_EQUAL_INT(cols, mat->cols);
    TEST_ASSERT_EQUAL_INT(rows * cols, mat->size);
    TEST_ASSERT_EQUAL_INT(1, mat->container->ref_count);
    TEST_ASSERT_EQUAL_INT(cols, mat->row_stride);
    TEST_ASSERT_EQUAL_INT(1, mat->col_stride);
    
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            TEST_ASSERT_EQUAL_DTYPE(init_value, AT(mat, i, j));
        }
    }
    
    free_mat(mat);  // Assuming you have this function to free the matrix.
}

void test_mat_init_zero_value(void) {
    size_t rows = 4;
    size_t cols = 4;
    
    Matrix* mat = mat_init(rows, cols, 0);
    
    TEST_ASSERT_NOT_NULL(mat);
    TEST_ASSERT_NOT_NULL(mat->container);
    TEST_ASSERT_NOT_NULL(mat->container->data);
    
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            TEST_ASSERT_EQUAL_DTYPE(0, AT(mat, i, j));
        }
    }
    
    free_mat(mat);
}



void test_mat_view_with_NULL_matrix(void) {
    Matrix* view = mat_view(NULL);
    TEST_ASSERT_NULL(view);
}

void test_successful_matrix_view_creation(void) {
    Matrix* mat = MATRIX(3, 3);
    Matrix* view = mat_view(mat);

    TEST_ASSERT_NOT_NULL(view);
    TEST_ASSERT_EQUAL_INT(mat->container->ref_count, 2);
    TEST_ASSERT_EQUAL_INT(view->cols, mat->cols);
    TEST_ASSERT_EQUAL_INT(view->rows, mat->rows);
    TEST_ASSERT_EQUAL_INT(view->size, mat->size);
    
    free_mat(mat);
    TEST_ASSERT_NOT_NULL(view->container);  // Ensure the view's container is still intact

    free_mat(view);
}

void test_data_in_matrix_view_and_original_matrix(void) {
    Matrix* mat = MATRIX_WITH(3, 3, 5);
    Matrix* view = mat_view(mat);

    AT(mat, 1, 1) = 7;
    TEST_ASSERT_EQUAL_DTYPE(7, AT(view, 1, 1)); // Ensure the view reflects changes in the original

    AT(view, 2, 2) = 9;
    TEST_ASSERT_EQUAL_DTYPE(9, AT(mat, 2, 2)); // Ensure the original reflects changes in the view

    free_mat(mat);
    free_mat(view);
}

void test_memory_checks_for_matrix_views(void) {
    Matrix* mat = MATRIX(4, 4);
    Matrix* view1 = mat_view(mat);
    Matrix* view2 = mat_view(mat);

    // ref_count should be incremented appropriately
    TEST_ASSERT_EQUAL_INT(mat->container->ref_count, 3);

    free_mat(mat); // This should not free the container or data since views are still existing

    TEST_ASSERT_NOT_NULL(view1->container);
    TEST_ASSERT_NOT_NULL(view2->container);

    free_mat(view1); // This should not free the container or data since another view is still existing
    TEST_ASSERT_NOT_NULL(view2->container);

    free_mat(view2); // Now, this should free the container and data since it's the last reference
}

void test_basic_transpose(void) {
    Matrix *mat = MATRIX(5, 2);
    Matrix *transposed = mat_transpose(mat);

    TEST_ASSERT_EQUAL_INT(mat->cols, transposed->rows);
    TEST_ASSERT_EQUAL_INT(mat->rows, transposed->cols);

    free_mat(transposed);
    free_mat(mat);
}

void test_memory_sharing(void) {
    Matrix *mat = MATRIX(5, 2);
    Matrix *transposed = mat_transpose(mat);

    TEST_ASSERT_TRUE(mat->container->data == transposed->container->data);

    free_mat(transposed);
    free_mat(mat);
}

void test_ref_count(void) {
    Matrix *mat = MATRIX(5, 2);
    Matrix *transposed = mat_transpose(mat);

    TEST_ASSERT_EQUAL_INT(2, mat->container->ref_count);

    free_mat(transposed);
    free_mat(mat);
}

void test_values(void) {
    Matrix *mat = MATRIX_WITH(2, 3, 1.0);
    AT(mat, 0, 1) = 2.0;
    Matrix *transposed = mat_transpose(mat);

    TEST_ASSERT_EQUAL_DTYPE(1.0, AT(transposed, 0, 0));
    TEST_ASSERT_EQUAL_DTYPE(2.0, AT(transposed, 1, 0));

    free_mat(transposed);
    free_mat(mat);
}

void test_memory_release(void) {
    Matrix *mat = MATRIX(2, 3);
    Matrix *transposed = mat_transpose(mat);
    
    // After this, the original matrix should not be freed entirely.
    free_mat(transposed);
    
    TEST_ASSERT_NOT_NULL(mat->container);
    TEST_ASSERT_NOT_NULL(mat->container->data);

    free_mat(mat); // this should now free everything
}

void test_transpose_of_transpose(void) {
    Matrix *mat = MATRIX(5, 2);
    Matrix *transposed = mat_transpose(mat);
    Matrix *transposed_twice = mat_transpose(transposed);

    TEST_ASSERT_EQUAL_INT(mat->rows, transposed_twice->rows);
    TEST_ASSERT_EQUAL_INT(mat->cols, transposed_twice->cols);

    free_mat(transposed_twice);
    free_mat(transposed);
    free_mat(mat);
}

void test_null_input(void) {
    Matrix *transposed = mat_transpose(NULL);

    TEST_ASSERT_NULL(transposed);
}

void test_large_matrix(void) {
    Matrix *mat = MATRIX(1000, 1000);
    Matrix *transposed = mat_transpose(mat);

    TEST_ASSERT_EQUAL_INT(1000, transposed->cols);
    TEST_ASSERT_EQUAL_INT(1000, transposed->rows);

    free_mat(transposed);
    free_mat(mat);
}
// Test initialization of a 3x3 matrix starting from 0
void test_mat_arrange_3x3_start_from_0(void) {
    Matrix *mat = mat_arrange(3, 3, 0);
    TEST_ASSERT_NOT_NULL(mat);
    for (size_t i = 0; i < mat->rows; i++) {
        for (size_t j = 0; j < mat->cols; j++) {
            TEST_ASSERT_EQUAL_DTYPE((i * mat->cols) + j, AT(mat, i, j));
        }
    }
    free_mat(mat);
}

// Test initialization of a 2x5 matrix starting from 10
void test_mat_arrange_2x5_start_from_10(void) {
    dtype start_value = 10;
    Matrix *mat = mat_arrange(2, 5, start_value);
    TEST_ASSERT_NOT_NULL(mat);
    for (size_t i = 0; i < mat->rows; i++) {
        for (size_t j = 0; j < mat->cols; j++) {
            TEST_ASSERT_EQUAL_DTYPE(start_value, AT(mat, i, j));
            start_value++;
        }
    }
    free_mat(mat);
}

// Test that memory is allocated for a 1x1 matrix
void test_mat_arrange_1x1(void) {
    Matrix *mat = mat_arrange(1, 1, 5);
    TEST_ASSERT_NOT_NULL(mat);
    TEST_ASSERT_EQUAL_DTYPE(5, AT(mat, 0, 0));
    free_mat(mat);
}

// Negative test: Expect the function to handle 0 rows or 0 columns (even though it might not be a common use case)
void test_mat_arrange_invalid_dimensions(void) {
    Matrix *mat_zero_rows = mat_arrange(0, 5, 0);
    TEST_ASSERT_NULL(mat_zero_rows);

    Matrix *mat_zero_cols = mat_arrange(5, 0, 0);
    TEST_ASSERT_NULL(mat_zero_cols);
}

// Test for a large matrix. This is to check if there are any memory constraints, etc.
void test_mat_arrange_large_matrix(void) {
    size_t large_size = 1000;
    Matrix *mat = mat_arrange(large_size, large_size, 0);
    TEST_ASSERT_NOT_NULL(mat);
    // Optionally, you can also iterate and verify values, but it might be time-consuming for very large matrices.
    free_mat(mat);
}


void test_mat_rand_basic_properties(void) {
    size_t rows = 5;
    size_t cols = 5;
    Matrix* matrix = mat_rand(rows, cols);
    
    TEST_ASSERT_NOT_NULL(matrix);
    TEST_ASSERT_EQUAL_UINT(rows, matrix->rows);
    TEST_ASSERT_EQUAL_UINT(cols, matrix->cols);

    free_mat(matrix);
}

void test_mat_rand_value_range(void) {
    size_t rows = 5;
    size_t cols = 5;
    Matrix* matrix = mat_rand(rows, cols);

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            TEST_ASSERT_TRUE(AT(matrix, i, j) >= 0 && AT(matrix, i, j) <= 1);
        }
    }

    free_mat(matrix);
}

void test_mat_rand_distinct_runs(void) {
    size_t rows = 5;
    size_t cols = 5;
    Matrix* matrix1 = mat_rand(rows, cols);
    Matrix* matrix2 = mat_rand(rows, cols);

    size_t matrices_are_equal = 1;
    for (size_t i = 0; i < rows && matrices_are_equal; i++) {
        for (size_t j = 0; j < cols; j++) {
            if (AT(matrix1, i, j) != AT(matrix2, i, j)) {
                matrices_are_equal = 0;
                break;
            }
        }
    }
    TEST_ASSERT_EQUAL_INT32(0,matrices_are_equal);

    free_mat(matrix1);
    free_mat(matrix2);
}



void test_mat_scale_basic_scaling(void) {
    size_t rows = 3;
    size_t cols = 3;
    dtype scalar = 2.0;

    Matrix* matrix = mat_arrange(rows, cols, 1);  // Assuming you have a mat_arrange function.
    Matrix* scaled_matrix = mat_scale(matrix, scalar);

    // Checking if all values are correctly scaled.
    for(size_t i = 0; i < rows; i++) {
        for(size_t j = 0; j < cols; j++) {
            TEST_ASSERT_EQUAL_FLOAT(AT(matrix, i, j) * scalar, AT(scaled_matrix, i, j));
        }
    }

    free_mat(matrix);
    free_mat(scaled_matrix);
}

void test_mat_scale_null_input(void) {
    Matrix* scaled_matrix = mat_scale(NULL, 2.0);
    TEST_ASSERT_NULL(scaled_matrix);
}

void test_mat_scale_ref_count_and_memory(void) {
    size_t rows = 3;
    size_t cols = 3;

    Matrix* matrix = mat_arrange(rows, cols, 1);
    Matrix* scaled_matrix = mat_scale(matrix, 2.0);

    // Assuming ref_count is a publicly accessible member of the container.
    TEST_ASSERT_EQUAL_INT(1, matrix->container->ref_count);
    TEST_ASSERT_NOT_EQUAL(matrix->container, scaled_matrix->container);  // Both matrices should point to the same data container.

    free_mat(matrix);
    // At this point, only the scaled matrix should have a reference to the data. Ref count should be 1.
    TEST_ASSERT_EQUAL_INT(1, scaled_matrix->container->ref_count);

    free_mat(scaled_matrix);
}

void test_add_matching_matrices(void) {
    Matrix* matrix1 = mat_arrange(2, 2, 1);  // Produces [[1,2], [3,4]]
    Matrix* matrix2 = mat_arrange(2, 2, 2);  // Produces [[2,3], [4,5]]

    Matrix* result = mat_add(matrix1, matrix2);

    TEST_ASSERT_EQUAL_DTYPE(3, AT(result, 0, 0));
    TEST_ASSERT_EQUAL_DTYPE(5, AT(result, 0, 1));
    TEST_ASSERT_EQUAL_DTYPE(7, AT(result, 1, 0));
    TEST_ASSERT_EQUAL_DTYPE(9, AT(result, 1, 1));

    free_mat(matrix1);
    free_mat(matrix2);
    free_mat(result);
}

void test_add_with_different_rows(void) {
    Matrix* matrix1 = MATRIX(3, 2);
    Matrix* matrix2 = MATRIX(2, 2);

    Matrix* result = mat_add(matrix1, matrix2);

    TEST_ASSERT_NULL(result);  // Should be NULL due to dimension mismatch

    free_mat(matrix1);
    free_mat(matrix2);
    free_mat(result);  // Safe to call, as it checks for NULL internally
}

void test_add_with_different_columns(void) {
    Matrix* matrix1 = MATRIX(2, 3);
    Matrix* matrix2 = MATRIX(2, 2);

    Matrix* result = mat_add(matrix1, matrix2);

    TEST_ASSERT_NULL(result);  // Should be NULL due to dimension mismatch

    free_mat(matrix1);
    free_mat(matrix2);
    free_mat(result);  // Safe to call, as it checks for NULL internally
}

void test_add_zeros(void) {
    Matrix* matrix1 = MATRIX(2, 2);
    Matrix* matrix2 = MATRIX(2, 2);

    for(size_t i = 0; i < matrix1->rows; i++)
        for(size_t j = 0; j < matrix1->cols; j++)
            AT(matrix1, i, j) = 0;

    for(size_t i = 0; i < matrix2->rows; i++)
        for(size_t j = 0; j < matrix2->cols; j++)
            AT(matrix2, i, j) = 0;

    Matrix* result = mat_add(matrix1, matrix2);

    for(size_t i = 0; i < result->rows; i++)
        for(size_t j = 0; j < result->cols; j++)
            TEST_ASSERT_EQUAL_DTYPE(0, AT(result, i, j));

    free_mat(matrix1);
    free_mat(matrix2);
    free_mat(result);
}

void test_add_matrix_to_itself(void) {
    Matrix* matrix1 = mat_arrange(2, 2, 1);  // Produces [[1,2], [3,4]]

    Matrix* result = mat_add(matrix1, matrix1);

    TEST_ASSERT_EQUAL_DTYPE(2, AT(result, 0, 0));
    TEST_ASSERT_EQUAL_DTYPE(4, AT(result, 0, 1));
    TEST_ASSERT_EQUAL_DTYPE(6, AT(result, 1, 0));
    TEST_ASSERT_EQUAL_DTYPE(8, AT(result, 1, 1));

    free_mat(matrix1);
    free_mat(result);
}

void test_add_null_matrices(void) {
    Matrix* matrix1 = NULL;
    Matrix* matrix2 = MATRIX(2, 2);

    Matrix* result = mat_add(matrix1, matrix2);
    TEST_ASSERT_NULL(result);

    result = mat_add(matrix2, matrix1);
    TEST_ASSERT_NULL(result);

    free_mat(matrix2);
    free_mat(result);  // Safe to call, as it checks for NULL internally
}

void test_subtract_matching_matrices(void) {
    Matrix* matrix1 = mat_arrange(2, 2, 1);  // Produces [[1,2], [3,4]]
    Matrix* matrix2 = mat_arrange(2, 2, 2);  // Produces [[2,3], [4,5]]

    Matrix* result = mat_subtract(matrix1, matrix2);

    TEST_ASSERT_EQUAL_DTYPE(-1, AT(result, 0, 0));
    TEST_ASSERT_EQUAL_DTYPE(-1, AT(result, 0, 1));
    TEST_ASSERT_EQUAL_DTYPE(-1, AT(result, 1, 0));
    TEST_ASSERT_EQUAL_DTYPE(-1, AT(result, 1, 1));

    free_mat(matrix1);
    free_mat(matrix2);
    free_mat(result);
}

void test_subtract_with_different_rows(void) {
    Matrix* matrix1 = MATRIX(3, 2);
    Matrix* matrix2 = MATRIX(2, 2);

    Matrix* result = mat_subtract(matrix1, matrix2);

    TEST_ASSERT_NULL(result);  // Should be NULL due to dimension mismatch

    free_mat(matrix1);
    free_mat(matrix2);
    free_mat(result);  // Safe to call, as it checks for NULL internally
}

void test_subtract_with_different_columns(void) {
    Matrix* matrix1 = MATRIX(2, 3);
    Matrix* matrix2 = MATRIX(2, 2);

    Matrix* result = mat_subtract(matrix1, matrix2);

    TEST_ASSERT_NULL(result);  // Should be NULL due to dimension mismatch

    free_mat(matrix1);
    free_mat(matrix2);
    free_mat(result);  // Safe to call, as it checks for NULL internally
}

void test_subtract_matrix_from_itself(void) {
    Matrix* matrix1 = mat_arrange(2, 2, 1);  // Produces [[1,2], [3,4]]

    Matrix* result = mat_subtract(matrix1, matrix1);

    for(size_t i = 0; i < result->rows; i++)
        for(size_t j = 0; j < result->cols; j++)
            TEST_ASSERT_EQUAL_DTYPE(0, AT(result, i, j));

    free_mat(matrix1);
    free_mat(result);
}

void test_subtract_null_matrices(void) {
    Matrix* matrix1 = NULL;
    Matrix* matrix2 = MATRIX(2, 2);

    Matrix* result = mat_subtract(matrix1, matrix2);
    TEST_ASSERT_NULL(result);

    result = mat_subtract(matrix2, matrix1);
    TEST_ASSERT_NULL(result);

    free_mat(matrix2);
    free_mat(result);  // Safe to call, as it checks for NULL internally
}

void test_dot_valid_matrices(void) {
    Matrix* matrix1 = mat_arrange(2, 3, 1);  // Produces [[1,2,3], [4,5,6]]
    Matrix* matrix2 = mat_arrange(3, 2, 1);  // Produces [[1,2], [3,4], [5,6]]

    Matrix* result = mat_dot(matrix1, matrix2);

    TEST_ASSERT_EQUAL_DTYPE(22, AT(result, 0, 0));
    TEST_ASSERT_EQUAL_DTYPE(28, AT(result, 0, 1));
    TEST_ASSERT_EQUAL_DTYPE(49, AT(result, 1, 0));
    TEST_ASSERT_EQUAL_DTYPE(64, AT(result, 1, 1));

    free_mat(matrix1);
    free_mat(matrix2);
    free_mat(result);
}

void test_dot_invalid_dimensions(void) {
    Matrix* matrix1 = MATRIX(2, 2);
    Matrix* matrix2 = MATRIX(3, 2);

    Matrix* result = mat_dot(matrix1, matrix2);

    TEST_ASSERT_NULL(result);  // Should be NULL due to dimension mismatch

    free_mat(matrix1);
    free_mat(matrix2);
    free_mat(result);  // Safe to call, as it checks for NULL internally
}

void test_dot_null_matrices(void) {
    Matrix* matrix1 = NULL;
    Matrix* matrix2 = MATRIX(2, 2);

    Matrix* result = mat_dot(matrix1, matrix2);
    TEST_ASSERT_NULL(result);

    result = mat_dot(matrix2, matrix1);
    TEST_ASSERT_NULL(result);

    free_mat(matrix2);
    free_mat(result);  // Safe to call, as it checks for NULL internally
}

void test_dot_matrix_and_its_transpose(void) {
    Matrix* matrix1 = mat_arrange(2, 3, 1);   // Produces [[1,2,3], [4,5,6]]
    Matrix* matrix2 = mat_transpose(matrix1); // Should produce [[1,4], [2,5], [3,6]]

    Matrix* result = mat_dot(matrix1, matrix2);

    TEST_ASSERT_EQUAL_DTYPE(14, AT(result, 0, 0));
    TEST_ASSERT_EQUAL_DTYPE(32, AT(result, 0, 1));
    TEST_ASSERT_EQUAL_DTYPE(32, AT(result, 1, 0));
    TEST_ASSERT_EQUAL_DTYPE(77, AT(result, 1, 1));

    free_mat(matrix1);
    free_mat(matrix2);
    free_mat(result);
}

void test_slice_valid_submatrix(void) {
    Matrix* matrix = mat_arrange(4, 4, 1); // Produces a 4x4 matrix with values from 1 to 16

    Matrix* slice = mat_slice(matrix, 1, 2, 1, 2); // 2x2 slice from row 1-2 and col 1-2

    TEST_ASSERT_EQUAL_DTYPE(6, AT(slice, 0, 0));
    TEST_ASSERT_EQUAL_DTYPE(7, AT(slice, 0, 1));
    TEST_ASSERT_EQUAL_DTYPE(10, AT(slice, 1, 0));
    TEST_ASSERT_EQUAL_DTYPE(11, AT(slice, 1, 1));

    free_mat(matrix);
    free_mat(slice);
}

void test_slice_invalid_dimensions(void) {
    Matrix* matrix = MATRIX(3, 3);
    Matrix* slice = mat_slice(matrix, 2, 1, 0, 1);  // start_row > end_row
    TEST_ASSERT_NULL(slice);

    slice = mat_slice(matrix, 0, 1, 2, 1); // start_col > end_col
    TEST_ASSERT_NULL(slice);

    slice = mat_slice(matrix, 0, 3, 0, 2); // end_row >= matrix->rows
    TEST_ASSERT_NULL(slice);

    free_mat(matrix);
}

void test_slice_null_matrix(void) {
    Matrix* matrix = NULL;
    Matrix* slice = mat_slice(matrix, 0, 1, 0, 1);
    TEST_ASSERT_NULL(slice);
}

void test_slice_entire_matrix(void) {
    Matrix* matrix = mat_arrange(4, 4, 1); // Produces a 4x4 matrix

    Matrix* slice = mat_slice(matrix, 0, 3, 0, 3); // Slice the entire matrix
    TEST_ASSERT_EQUAL_DTYPE(1, AT(slice, 0, 0));
    TEST_ASSERT_EQUAL_DTYPE(16, AT(slice, 3, 3));

    free_mat(matrix);
    free_mat(slice);
}

void test_slice_single_row_col(void) {
    Matrix* matrix = mat_arrange(4, 4, 1); // Produces a 4x4 matrix

    Matrix* row_slice = mat_slice(matrix, 2, 2, 0, 3);  // Third row
    Matrix* col_slice = mat_slice(matrix, 0, 3, 2, 2);  // Third column

    TEST_ASSERT_EQUAL_DTYPE(9, AT(row_slice, 0, 0));
    TEST_ASSERT_EQUAL_DTYPE(3, AT(col_slice, 0, 0));

    free_mat(matrix);
    free_mat(row_slice);
    free_mat(col_slice);
}
void test_AT_macro(void) {
    Matrix* matrix = MATRIX(3, 3); // 3x3 matrix with zeroes
    AT(matrix, 1, 1) = 5;

    TEST_ASSERT_EQUAL_DTYPE(5, AT(matrix, 1, 1));

    free_mat(matrix);
}

void test_MATRIX_macro(void) {
    Matrix* matrix = MATRIX(2, 2);

    TEST_ASSERT_NOT_NULL(matrix);
    TEST_ASSERT_EQUAL_DTYPE(0, AT(matrix, 0, 0));

    free_mat(matrix);
}

void test_MATRIX_VIEW_macro(void) {
    Matrix* original = MATRIX(3, 3);
    Matrix* view = MATRIX_VIEW(original);

    TEST_ASSERT_NOT_NULL(view);
    TEST_ASSERT_EQUAL_DTYPE(0, AT(view, 0, 0));

    free_mat(original);
    free_mat(view);
}

void test_MATRIX_COPY_macro(void) {
    Matrix* original = MATRIX(3, 3);
    Matrix* copy = MATRIX_COPY(original);

    TEST_ASSERT_NOT_NULL(copy);
    AT(original, 1, 1) = 5;

    TEST_ASSERT_NOT_EQUAL(AT(copy, 1, 1), AT(original, 1, 1));

    free_mat(original);
    free_mat(copy);
}

void test_MATRIX_WITH_macro(void) {
    Matrix* matrix = MATRIX_WITH(2, 2, 7);

    TEST_ASSERT_NOT_NULL(matrix);
    TEST_ASSERT_EQUAL_DTYPE(7, AT(matrix, 0, 0));

    free_mat(matrix);
}

void test_ROW_SLICE_macro(void) {
    Matrix* matrix = MATRIX(4, 4);
    Matrix* row_slice = ROW_SLICE(matrix, 1, 2);  // Slice rows 1 to 2 inclusive

    TEST_ASSERT_NOT_NULL(row_slice);
    TEST_ASSERT_EQUAL_UINT(2, row_slice->rows);
    TEST_ASSERT_EQUAL_UINT(4, row_slice->cols);

    free_mat(matrix);
    free_mat(row_slice);
}

void test_COL_SLICE_macro(void) {
    Matrix* matrix = MATRIX(4, 4);
    Matrix* col_slice = COL_SLICE(matrix, 1, 2);  // Slice columns 1 to 2 inclusive

    TEST_ASSERT_NOT_NULL(col_slice);
    TEST_ASSERT_EQUAL_UINT(4, col_slice->rows);
    TEST_ASSERT_EQUAL_UINT(2, col_slice->cols);

    free_mat(matrix);
    free_mat(col_slice);
}

int main(void) {
    UNITY_BEGIN();

    // initialization
    RUN_TEST(test_basic_matrix_initialization);
    RUN_TEST(test_matrix_initialization_with_value);
    RUN_TEST(test_mat_arrange_memory_layout);
    RUN_TEST(test_mat_arrange_correctness);

    // free
    RUN_TEST(test_basic_free);
    RUN_TEST(test_ref_count_free);
    RUN_TEST(test_data_check);

    // indexing
    RUN_TEST(test_matrix_indexing);

    // container
    RUN_TEST(test_container_check);
    RUN_TEST(test_init_container_successful_allocation);
    RUN_TEST(test_mat_init_successful_allocation);
    RUN_TEST(test_mat_init_zero_value);

    // view
    RUN_TEST(test_mat_view_with_NULL_matrix);
    RUN_TEST(test_successful_matrix_view_creation);
    RUN_TEST(test_data_in_matrix_view_and_original_matrix);
    RUN_TEST(test_memory_checks_for_matrix_views);

    // matrix transpose
    RUN_TEST(test_basic_transpose);
    RUN_TEST(test_memory_sharing);
    RUN_TEST(test_basic_transpose);
    RUN_TEST(test_memory_sharing);
    RUN_TEST(test_ref_count);
    RUN_TEST(test_values);
    RUN_TEST(test_memory_release);
    RUN_TEST(test_transpose_of_transpose);
    RUN_TEST(test_null_input);
    RUN_TEST(test_large_matrix);

    // arrange
    RUN_TEST(test_mat_arrange_3x3_start_from_0);
    RUN_TEST(test_mat_arrange_2x5_start_from_10);
    RUN_TEST(test_mat_arrange_1x1);
    // RUN_TEST(test_mat_arrange_invalid_dimensions);
    RUN_TEST(test_mat_arrange_large_matrix);

    // random
    RUN_TEST(test_mat_rand_basic_properties);
    RUN_TEST(test_mat_rand_value_range);
    RUN_TEST(test_mat_rand_distinct_runs);

    // scale
    RUN_TEST(test_mat_scale_basic_scaling);
    RUN_TEST(test_mat_scale_null_input);
    RUN_TEST(test_mat_scale_ref_count_and_memory);

    // addition
    RUN_TEST(test_add_matching_matrices);
    RUN_TEST(test_add_with_different_rows);
    RUN_TEST(test_add_with_different_columns);
    RUN_TEST(test_add_zeros);
    RUN_TEST(test_add_matrix_to_itself);
    RUN_TEST(test_add_null_matrices);

    // subtraction
    RUN_TEST(test_subtract_matching_matrices);
    RUN_TEST(test_subtract_with_different_rows);
    RUN_TEST(test_subtract_with_different_columns);
    RUN_TEST(test_subtract_matrix_from_itself);
    RUN_TEST(test_subtract_null_matrices);

    // dot product
    RUN_TEST(test_dot_valid_matrices);
    RUN_TEST(test_dot_invalid_dimensions);
    RUN_TEST(test_dot_null_matrices);
    RUN_TEST(test_dot_matrix_and_its_transpose);

    // slice
    RUN_TEST(test_slice_valid_submatrix);
    RUN_TEST(test_slice_invalid_dimensions);
    RUN_TEST(test_slice_null_matrix);
    RUN_TEST(test_slice_entire_matrix);
    RUN_TEST(test_slice_single_row_col);

    // macros
    RUN_TEST(test_AT_macro);
    RUN_TEST(test_MATRIX_macro);
    RUN_TEST(test_MATRIX_VIEW_macro);
    RUN_TEST(test_MATRIX_COPY_macro);
    RUN_TEST(test_MATRIX_WITH_macro);
    RUN_TEST(test_ROW_SLICE_macro);
    RUN_TEST(test_COL_SLICE_macro);
    return UNITY_END();
}
