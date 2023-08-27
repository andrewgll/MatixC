#include "unity.h"
#include "mx.h" 

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

void test_mx_arrange_memory_layout(void) {
    size_t rows = 3;
    size_t cols = 3;
    Matrix *m = mx_arrange(rows, cols, 0);

    TEST_ASSERT_NOT_NULL(m);
    TEST_ASSERT_NOT_NULL(m->container);
    TEST_ASSERT_NOT_NULL(m->container->data);

    mx_free(m);
}

void test_mx_arrange_correctness(void) {
    size_t rows = 3;
    size_t cols = 3;
    double start_val = 5.0;
    Matrix *m = mx_arrange(rows, cols, start_val);

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            TEST_ASSERT_EQUAL_DTYPE(start_val, AT(m, i, j));
            start_val++;
        }
    }

    mx_free(m);
}


void test_basic_free(void)
{
    Matrix *mat = malloc(sizeof(Matrix));
    mat->container = malloc(sizeof(__matrix_container));
    mat->container->data = malloc(10 * sizeof(dtype));
    mat->container->ref_count = 1;

    mx_free(mat);

    // If we've reached here, the basic free worked without crashing.
    // (We're not checking the memory content because it's undefined after free)
}

void test_ref_count_free(void)
{
    Matrix *mat = MATRIX(5,2);
    mat->container->ref_count = 2;
    __matrix_container* ct = mat->container;

    mx_free(mat);

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

    mx_free(mat);

    // Can't check mat->container->data since it's freed.
    // If we've reached here without crashes, we're good.
}

void test_container_check(void)
{
    Matrix *mat = malloc(sizeof(Matrix));
    mat->container = malloc(sizeof(__matrix_container));
    mat->container->data = malloc(10 * sizeof(dtype));
    mat->container->ref_count = 1;

    mx_free(mat);

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
    TEST_ASSERT_EQUAL_INT(25, mat->cols*mat->rows);
    TEST_ASSERT_EQUAL_INT(1, mat->container->ref_count);
    TEST_ASSERT_EQUAL_INT(5, mat->row_stride);
    TEST_ASSERT_EQUAL_INT(1, mat->col_stride);

    for(size_t i = 0; i < mat->rows; i++){
        for(size_t j = 0; j < mat->cols; j++){
            TEST_ASSERT_EQUAL_DTYPE(1.0, AT(mat, i, j));
        }
    }

    mx_free(mat);
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
    TEST_ASSERT_EQUAL_INT(9, mat->cols*mat->rows);
    TEST_ASSERT_EQUAL_INT(1, mat->container->ref_count);
    TEST_ASSERT_EQUAL_INT(3, mat->row_stride);
    TEST_ASSERT_EQUAL_INT(1, mat->col_stride);

    for(size_t i = 0; i < mat->rows; i++){
        for(size_t j = 0; j < mat->cols; j++){
            TEST_ASSERT_EQUAL_DTYPE(value, AT(mat, i, j));
        }
    }

    mx_free(mat);
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

    mx_free(mat);
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

void test_mx_init_successful_allocation(void) {
    size_t rows = 3;
    size_t cols = 3;
    dtype init_value = 5.0;
    
    Matrix* mat = mx_init(rows, cols, init_value);
    
    TEST_ASSERT_NOT_NULL(mat);
    TEST_ASSERT_NOT_NULL(mat->container);
    TEST_ASSERT_NOT_NULL(mat->container->data);
    TEST_ASSERT_EQUAL_INT(rows, mat->rows);
    TEST_ASSERT_EQUAL_INT(cols, mat->cols);
    TEST_ASSERT_EQUAL_INT(rows * cols, mat->cols*mat->rows);
    TEST_ASSERT_EQUAL_INT(1, mat->container->ref_count);
    TEST_ASSERT_EQUAL_INT(cols, mat->row_stride);
    TEST_ASSERT_EQUAL_INT(1, mat->col_stride);
    
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            TEST_ASSERT_EQUAL_DTYPE(init_value, AT(mat, i, j));
        }
    }
    
    mx_free(mat);  // Assuming you have this function to free the matrix.
}

void test_mx_init_zero_value(void) {
    size_t rows = 4;
    size_t cols = 4;
    
    Matrix* mat = mx_init(rows, cols, 0);
    
    TEST_ASSERT_NOT_NULL(mat);
    TEST_ASSERT_NOT_NULL(mat->container);
    TEST_ASSERT_NOT_NULL(mat->container->data);
    
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            TEST_ASSERT_EQUAL_DTYPE(0, AT(mat, i, j));
        }
    }
    
    mx_free(mat);
}



void test_MATRIX_VIEW_with_NULL_matrix(void) {
    Matrix* view = MATRIX_VIEW(NULL);

    // Check that the view is not NULL
    TEST_ASSERT_NOT_NULL(view);

    // Check the dimensions of the matrix
    TEST_ASSERT_EQUAL_UINT16(0, view->rows);
    TEST_ASSERT_EQUAL_UINT16(0, view->cols);

    // Check that the default value is 0
    TEST_ASSERT_EQUAL(0, view->default_value);

    mx_free(view);
}


void test_successful_matrix_view_creation(void) {
    Matrix* mat = MATRIX(3, 3);
    Matrix* view = MATRIX_VIEW(mat);

    TEST_ASSERT_NOT_NULL(view);
    TEST_ASSERT_EQUAL_INT(mat->container->ref_count, 2);
    TEST_ASSERT_EQUAL_INT(view->cols, mat->cols);
    TEST_ASSERT_EQUAL_INT(view->rows, mat->rows);
    TEST_ASSERT_EQUAL_INT(view->cols*mat->rows, mat->cols*mat->rows);
    
    mx_free(mat);
    TEST_ASSERT_NOT_NULL(view->container);  // Ensure the view's container is still intact

    mx_free(view);
}

void test_data_in_matrix_view_and_original_matrix(void) {
    Matrix* mat = MATRIX_WITH(3, 3, 5);
    Matrix* view = MATRIX_VIEW(mat);

    AT(mat, 1, 1) = 7;
    TEST_ASSERT_EQUAL_DTYPE(7, AT(view, 1, 1)); // Ensure the view reflects changes in the original

    AT(view, 2, 2) = 9;
    TEST_ASSERT_EQUAL_DTYPE(9, AT(mat, 2, 2)); // Ensure the original reflects changes in the view

    mx_free(mat);
    mx_free(view);
}

void test_memory_checks_for_matrix_views(void) {
    Matrix* mat = MATRIX(4, 4);
    Matrix* view1 = MATRIX_VIEW(mat);
    Matrix* view2 = MATRIX_VIEW(mat);

    // ref_count should be incremented appropriately
    TEST_ASSERT_EQUAL_INT(mat->container->ref_count, 3);

    mx_free(mat); // This should not free the container or data since views are still existing

    TEST_ASSERT_NOT_NULL(view1->container);
    TEST_ASSERT_NOT_NULL(view2->container);

    mx_free(view1); // This should not free the container or data since another view is still existing
    TEST_ASSERT_NOT_NULL(view2->container);

    mx_free(view2); // Now, this should free the container and data since it's the last reference
}

void test_basic_transpose(void) {
    Matrix *mat = MATRIX(5, 2);
    Matrix *transposed = TRANSPOSE_VIEW(mat);

    TEST_ASSERT_EQUAL_INT(mat->cols, transposed->rows);
    TEST_ASSERT_EQUAL_INT(mat->rows, transposed->cols);

    mx_free(transposed);
    mx_free(mat);
}

void test_memory_sharing(void) {
    Matrix *mat = MATRIX(5, 2);
    Matrix *transposed = TRANSPOSE_VIEW(mat);

    TEST_ASSERT_TRUE(mat->container->data == transposed->container->data);

    mx_free(transposed);
    mx_free(mat);
}

void test_ref_count(void) {
    Matrix *mat = MATRIX(5, 2);
    Matrix *transposed = TRANSPOSE_VIEW(mat);

    TEST_ASSERT_EQUAL_INT(2, mat->container->ref_count);

    mx_free(transposed);
    mx_free(mat);
}

void test_values(void) {
    Matrix *mat = MATRIX_WITH(2, 3, 1.0);
    AT(mat, 0, 1) = 2.0;
    Matrix *transposed = TRANSPOSE_VIEW(mat);

    TEST_ASSERT_EQUAL_DTYPE(1.0, AT(transposed, 0, 0));
    TEST_ASSERT_EQUAL_DTYPE(2.0, AT(transposed, 1, 0));

    mx_free(transposed);
    mx_free(mat);
}

void test_memory_release(void) {
    Matrix *mat = MATRIX(2, 3);
    Matrix *transposed = TRANSPOSE_VIEW(mat);
    
    // After this, the original matrix should not be freed entirely.
    mx_free(transposed);
    
    TEST_ASSERT_NOT_NULL(mat->container);
    TEST_ASSERT_NOT_NULL(mat->container->data);

    mx_free(mat); // this should now free everything
}

void test_transpose_of_transpose(void) {
    Matrix *mat = MATRIX(5, 2);
    Matrix *transposed = TRANSPOSE_VIEW(mat);
    Matrix *transposed_twice = TRANSPOSE_VIEW(transposed);

    TEST_ASSERT_EQUAL_INT(mat->rows, transposed_twice->rows);
    TEST_ASSERT_EQUAL_INT(mat->cols, transposed_twice->cols);

    mx_free(transposed_twice);
    mx_free(transposed);
    mx_free(mat);
}

void test_null_input(void) {
    Matrix *transposed = TRANSPOSE_VIEW(NULL);

    TEST_ASSERT_NULL(transposed);
}

void test_large_matrix(void) {
    Matrix *mat = MATRIX(1000, 1000);
    Matrix *transposed = TRANSPOSE_VIEW(mat);

    TEST_ASSERT_EQUAL_INT(1000, transposed->cols);
    TEST_ASSERT_EQUAL_INT(1000, transposed->rows);

    mx_free(transposed);
    mx_free(mat);
}
// Test initialization of a 3x3 matrix starting from 0
void test_mx_arrange_3x3_start_from_0(void) {
    Matrix *mat = mx_arrange(3, 3, 0);
    TEST_ASSERT_NOT_NULL(mat);
    for (size_t i = 0; i < mat->rows; i++) {
        for (size_t j = 0; j < mat->cols; j++) {
            TEST_ASSERT_EQUAL_DTYPE((i * mat->cols) + j, AT(mat, i, j));
        }
    }
    mx_free(mat);
}

// Test initialization of a 2x5 matrix starting from 10
void test_mx_arrange_2x5_start_from_10(void) {
    dtype start_value = 10;
    Matrix *mat = mx_arrange(2, 5, start_value);
    TEST_ASSERT_NOT_NULL(mat);
    for (size_t i = 0; i < mat->rows; i++) {
        for (size_t j = 0; j < mat->cols; j++) {
            TEST_ASSERT_EQUAL_DTYPE(start_value, AT(mat, i, j));
            start_value++;
        }
    }
    mx_free(mat);
}

// Test that memory is allocated for a 1x1 matrix
void test_mx_arrange_1x1(void) {
    Matrix *mat = mx_arrange(1, 1, 5);
    TEST_ASSERT_NOT_NULL(mat);
    TEST_ASSERT_EQUAL_DTYPE(5, AT(mat, 0, 0));
    mx_free(mat);
}

// Negative test: Expect the function to handle 0 rows or 0 columns (even though it might not be a common use case)
void test_mx_arrange_invalid_dimensions(void) {
    Matrix *mx_zero_rows = mx_arrange(0, 5, 0);
    TEST_ASSERT_NULL(mx_zero_rows);

    Matrix *mx_zero_cols = mx_arrange(5, 0, 0);
    TEST_ASSERT_NULL(mx_zero_cols);
}

// Test for a large matrix. This is to check if there are any memory constraints, etc.
void test_mx_arrange_large_matrix(void) {
    size_t large_size = 1000;
    Matrix *mat = mx_arrange(large_size, large_size, 0);
    TEST_ASSERT_NOT_NULL(mat);
    // Optionally, you can also iterate and verify values, but it might be time-consuming for very large matrices.
    mx_free(mat);
}


void test_mx_rand_basic_properties(void) {
    size_t rows = 5;
    size_t cols = 5;
    Matrix* matrix = mx_rand(rows, cols);
    
    TEST_ASSERT_NOT_NULL(matrix);
    TEST_ASSERT_EQUAL_UINT(rows, matrix->rows);
    TEST_ASSERT_EQUAL_UINT(cols, matrix->cols);

    mx_free(matrix);
}

void test_mx_rand_value_range(void) {
    size_t rows = 5;
    size_t cols = 5;
    Matrix* matrix = mx_rand(rows, cols);

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            TEST_ASSERT_TRUE(AT(matrix, i, j) >= 0 && AT(matrix, i, j) <= 1);
        }
    }

    mx_free(matrix);
}

void test_mx_rand_distinct_runs(void) {
    size_t rows = 5;
    size_t cols = 5;
    Matrix* matrix1 = mx_rand(rows, cols);
    Matrix* matrix2 = mx_rand(rows, cols);

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

    mx_free(matrix1);
    mx_free(matrix2);
}



void test_mx_scale_basic_scaling(void) {
    size_t rows = 3;
    size_t cols = 3;
    dtype scalar = 2.0;

    Matrix* matrix = mx_arrange(rows, cols, 1);  // Assuming you have a mx_arrange function.
    Matrix* scaled_matrix = mx_scale(matrix, scalar);

    // Checking if all values are correctly scaled.
    for(size_t i = 0; i < rows; i++) {
        for(size_t j = 0; j < cols; j++) {
            TEST_ASSERT_EQUAL_FLOAT(AT(matrix, i, j) * scalar, AT(scaled_matrix, i, j));
        }
    }

    mx_free(matrix);
    mx_free(scaled_matrix);
}

void test_mx_scale_null_input(void) {
    Matrix* scaled_matrix = mx_scale(NULL, 2.0);
    TEST_ASSERT_NULL(scaled_matrix);
}

void test_mx_scale_ref_count_and_memory(void) {
    size_t rows = 3;
    size_t cols = 3;

    Matrix* matrix = mx_arrange(rows, cols, 1);
    Matrix* scaled_matrix = mx_scale(matrix, 2.0);

    // Assuming ref_count is a publicly accessible member of the container.
    TEST_ASSERT_EQUAL_INT(1, matrix->container->ref_count);
    TEST_ASSERT_NOT_EQUAL(matrix->container, scaled_matrix->container);  // Both matrices should point to the same data container.

    mx_free(matrix);
    // At this point, only the scaled matrix should have a reference to the data. Ref count should be 1.
    TEST_ASSERT_EQUAL_INT(1, scaled_matrix->container->ref_count);

    mx_free(scaled_matrix);
}

void test_add_matching_matrices(void) {
    Matrix* matrix1 = mx_arrange(2, 2, 1);  // Produces [[1,2], [3,4]]
    Matrix* matrix2 = mx_arrange(2, 2, 2);  // Produces [[2,3], [4,5]]

    Matrix* result = mx_add(matrix1, matrix2);

    TEST_ASSERT_EQUAL_DTYPE(3, AT(result, 0, 0));
    TEST_ASSERT_EQUAL_DTYPE(5, AT(result, 0, 1));
    TEST_ASSERT_EQUAL_DTYPE(7, AT(result, 1, 0));
    TEST_ASSERT_EQUAL_DTYPE(9, AT(result, 1, 1));

    mx_free(matrix1);
    mx_free(matrix2);
    mx_free(result);
}

void test_add_with_different_rows(void) {
    Matrix* matrix1 = MATRIX(3, 2);
    Matrix* matrix2 = MATRIX(2, 2);

    Matrix* result = mx_add(matrix1, matrix2);

    TEST_ASSERT_NULL(result);  // Should be NULL due to dimension mismatch

    mx_free(matrix1);
    mx_free(matrix2);
    mx_free(result);  // Safe to call, as it checks for NULL internally
}

void test_add_with_different_columns(void) {
    Matrix* matrix1 = MATRIX(2, 3);
    Matrix* matrix2 = MATRIX(2, 2);

    Matrix* result = mx_add(matrix1, matrix2);

    TEST_ASSERT_NULL(result);  // Should be NULL due to dimension mismatch

    mx_free(matrix1);
    mx_free(matrix2);
    mx_free(result);  // Safe to call, as it checks for NULL internally
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

    Matrix* result = mx_add(matrix1, matrix2);

    for(size_t i = 0; i < result->rows; i++)
        for(size_t j = 0; j < result->cols; j++)
            TEST_ASSERT_EQUAL_DTYPE(0, AT(result, i, j));

    mx_free(matrix1);
    mx_free(matrix2);
    mx_free(result);
}

void test_add_matrix_to_itself(void) {
    Matrix* matrix1 = mx_arrange(2, 2, 1);  // Produces [[1,2], [3,4]]

    Matrix* result = mx_add(matrix1, matrix1);

    TEST_ASSERT_EQUAL_DTYPE(2, AT(result, 0, 0));
    TEST_ASSERT_EQUAL_DTYPE(4, AT(result, 0, 1));
    TEST_ASSERT_EQUAL_DTYPE(6, AT(result, 1, 0));
    TEST_ASSERT_EQUAL_DTYPE(8, AT(result, 1, 1));

    mx_free(matrix1);
    mx_free(result);
}

void test_add_null_matrices(void) {
    Matrix* matrix1 = NULL;
    Matrix* matrix2 = MATRIX(2, 2);

    Matrix* result = mx_add(matrix1, matrix2);
    TEST_ASSERT_NULL(result);

    result = mx_add(matrix2, matrix1);
    TEST_ASSERT_NULL(result);

    mx_free(matrix2);
    mx_free(result);  // Safe to call, as it checks for NULL internally
}

void test_subtract_matching_matrices(void) {
    Matrix* matrix1 = mx_arrange(2, 2, 1);  // Produces [[1,2], [3,4]]
    Matrix* matrix2 = mx_arrange(2, 2, 2);  // Produces [[2,3], [4,5]]

    Matrix* result = mx_subtract(matrix1, matrix2);

    TEST_ASSERT_EQUAL_DTYPE(-1, AT(result, 0, 0));
    TEST_ASSERT_EQUAL_DTYPE(-1, AT(result, 0, 1));
    TEST_ASSERT_EQUAL_DTYPE(-1, AT(result, 1, 0));
    TEST_ASSERT_EQUAL_DTYPE(-1, AT(result, 1, 1));

    mx_free(matrix1);
    mx_free(matrix2);
    mx_free(result);
}

void test_subtract_with_different_rows(void) {
    Matrix* matrix1 = MATRIX(3, 2);
    Matrix* matrix2 = MATRIX(2, 2);

    Matrix* result = mx_subtract(matrix1, matrix2);

    TEST_ASSERT_NULL(result);  // Should be NULL due to dimension mismatch

    mx_free(matrix1);
    mx_free(matrix2);
    mx_free(result);  // Safe to call, as it checks for NULL internally
}

void test_subtract_with_different_columns(void) {
    Matrix* matrix1 = MATRIX(2, 3);
    Matrix* matrix2 = MATRIX(2, 2);

    Matrix* result = mx_subtract(matrix1, matrix2);

    TEST_ASSERT_NULL(result);  // Should be NULL due to dimension mismatch

    mx_free(matrix1);
    mx_free(matrix2);
    mx_free(result);  // Safe to call, as it checks for NULL internally
}

void test_subtract_matrix_from_itself(void) {
    Matrix* matrix1 = mx_arrange(2, 2, 1);  // Produces [[1,2], [3,4]]

    Matrix* result = mx_subtract(matrix1, matrix1);

    for(size_t i = 0; i < result->rows; i++)
        for(size_t j = 0; j < result->cols; j++)
            TEST_ASSERT_EQUAL_DTYPE(0, AT(result, i, j));

    mx_free(matrix1);
    mx_free(result);
}

void test_subtract_null_matrices(void) {
    Matrix* matrix1 = NULL;
    Matrix* matrix2 = MATRIX(2, 2);

    Matrix* result = mx_subtract(matrix1, matrix2);
    TEST_ASSERT_NULL(result);

    result = mx_subtract(matrix2, matrix1);
    TEST_ASSERT_NULL(result);

    mx_free(matrix2);
    mx_free(result);  // Safe to call, as it checks for NULL internally
}

void test_dot_valid_matrices(void) {
    Matrix* matrix1 = mx_arrange(2, 3, 1);  // Produces [[1,2,3], [4,5,6]]
    Matrix* matrix2 = mx_arrange(3, 2, 1);  // Produces [[1,2], [3,4], [5,6]]

    Matrix* result = mx_dot(matrix1, matrix2);

    TEST_ASSERT_EQUAL_DTYPE(22, AT(result, 0, 0));
    TEST_ASSERT_EQUAL_DTYPE(28, AT(result, 0, 1));
    TEST_ASSERT_EQUAL_DTYPE(49, AT(result, 1, 0));
    TEST_ASSERT_EQUAL_DTYPE(64, AT(result, 1, 1));

    mx_free(matrix1);
    mx_free(matrix2);
    mx_free(result);
}

// void test_dot_invalid_dimensions(void) {
//     Matrix* matrix1 = MATRIX(2, 2);
//     Matrix* matrix2 = MATRIX(3, 2);

//     Matrix* result = mx_dot(matrix1, matrix2);

//     TEST_ASSERT_NULL(result);  // Should be NULL due to dimension mismatch

//     mx_free(matrix1);
//     mx_free(matrix2);
//     mx_free(result);  // Safe to call, as it checks for NULL internally
// }

void test_dot_null_matrices(void) {
    Matrix* matrix1 = NULL;
    Matrix* matrix2 = MATRIX(2, 2);

    Matrix* result = mx_dot(matrix1, matrix2);
    TEST_ASSERT_NULL(result);

    result = mx_dot(matrix2, matrix1);
    TEST_ASSERT_NULL(result);

    mx_free(matrix2);
    mx_free(result);  // Safe to call, as it checks for NULL internally
}

void test_dot_matrix_and_its_transpose(void) {
    Matrix* matrix1 = mx_arrange(2, 3, 1);   // Produces [[1,2,3], [4,5,6]]
    Matrix* matrix2 = TRANSPOSE_VIEW(matrix1); // Should produce [[1,4], [2,5], [3,6]]

    Matrix* result = mx_dot(matrix1, matrix2);

    TEST_ASSERT_EQUAL_DTYPE(14, AT(result, 0, 0));
    TEST_ASSERT_EQUAL_DTYPE(32, AT(result, 0, 1));
    TEST_ASSERT_EQUAL_DTYPE(32, AT(result, 1, 0));
    TEST_ASSERT_EQUAL_DTYPE(77, AT(result, 1, 1));

    mx_free(matrix1);
    mx_free(matrix2);
    mx_free(result);
}

void test_slice_valid_submatrix(void) {
    Matrix* matrix = mx_arrange(4, 4, 1); // Produces a 4x4 matrix with values from 1 to 16

    Matrix* slice = mx_slice(matrix, 1, 2, 1, 2); // 2x2 slice from row 1-2 and col 1-2

    TEST_ASSERT_EQUAL_DTYPE(6, AT(slice, 0, 0));
    TEST_ASSERT_EQUAL_DTYPE(7, AT(slice, 0, 1));
    TEST_ASSERT_EQUAL_DTYPE(10, AT(slice, 1, 0));
    TEST_ASSERT_EQUAL_DTYPE(11, AT(slice, 1, 1));

    mx_free(matrix);
    mx_free(slice);
}

void test_slice_invalid_dimensions(void) {
    Matrix* matrix = MATRIX(3, 3);
    Matrix* slice = mx_slice(matrix, 2, 1, 0, 1);  // start_row > end_row
    TEST_ASSERT_NULL(slice);

    slice = mx_slice(matrix, 0, 1, 2, 1); // start_col > end_col
    TEST_ASSERT_NULL(slice);

    slice = mx_slice(matrix, 0, 3, 0, 2); // end_row >= matrix->rows
    TEST_ASSERT_NULL(slice);

    mx_free(matrix);
}

void test_slice_null_matrix(void) {
    Matrix* matrix = NULL;
    Matrix* slice = mx_slice(matrix, 0, 1, 0, 1);
    TEST_ASSERT_NULL(slice);
}

void test_slice_entire_matrix(void) {
    Matrix* matrix = mx_arrange(4, 4, 1); // Produces a 4x4 matrix

    Matrix* slice = mx_slice(matrix, 0, 3, 0, 3); // Slice the entire matrix
    TEST_ASSERT_EQUAL_DTYPE(1, AT(slice, 0, 0));
    TEST_ASSERT_EQUAL_DTYPE(16, AT(slice, 3, 3));

    mx_free(matrix);
    mx_free(slice);
}

void test_slice_single_row_col(void) {
    Matrix* matrix = mx_arrange(4, 4, 1); // Produces a 4x4 matrix

    Matrix* row_slice = mx_slice(matrix, 2, 2, 0, 3);  // Third row
    Matrix* col_slice = mx_slice(matrix, 0, 3, 2, 2);  // Third column

    TEST_ASSERT_EQUAL_DTYPE(9, AT(row_slice, 0, 0));
    TEST_ASSERT_EQUAL_DTYPE(3, AT(col_slice, 0, 0));

    mx_free(matrix);
    mx_free(row_slice);
    mx_free(col_slice);
}
void test_AT_macro(void) {
    Matrix* matrix = MATRIX(3, 3); // 3x3 matrix with zeroes
    AT(matrix, 1, 1) = 5;

    TEST_ASSERT_EQUAL_DTYPE(5, AT(matrix, 1, 1));

    mx_free(matrix);
}

void test_MATRIX_macro(void) {
    Matrix* matrix = MATRIX(2, 2);

    TEST_ASSERT_NOT_NULL(matrix);
    TEST_ASSERT_EQUAL_DTYPE(0, AT(matrix, 0, 0));

    mx_free(matrix);
}

void test_MATRIX_VIEW_macro(void) {
    Matrix* original = MATRIX(3, 3);
    Matrix* view = MATRIX_VIEW(original);

    TEST_ASSERT_NOT_NULL(view);
    TEST_ASSERT_EQUAL_DTYPE(0, AT(view, 0, 0));

    mx_free(original);
    mx_free(view);
}

void test_MATRIX_COPY_macro(void) {
    Matrix* original = MATRIX(3, 3);
    Matrix* copy = MATRIX_COPY(original);

    TEST_ASSERT_NOT_NULL(copy);
    AT(original, 1, 1) = 5;

    TEST_ASSERT_NOT_EQUAL(AT(copy, 1, 1), AT(original, 1, 1));

    mx_free(original);
    mx_free(copy);
}

void test_MATRIX_WITH_macro(void) {
    Matrix* matrix = MATRIX_WITH(2, 2, 7);

    TEST_ASSERT_NOT_NULL(matrix);
    TEST_ASSERT_EQUAL_DTYPE(7, AT(matrix, 0, 0));

    mx_free(matrix);
}

void test_ROW_SLICE_macro(void) {
    Matrix* matrix = MATRIX(4, 4);
    Matrix* row_slice = ROW_SLICE(matrix, 1, 2);  // Slice rows 1 to 2 inclusive

    TEST_ASSERT_NOT_NULL(row_slice);
    TEST_ASSERT_EQUAL_UINT(2, row_slice->rows);
    TEST_ASSERT_EQUAL_UINT(4, row_slice->cols);

    mx_free(matrix);
    mx_free(row_slice);
}

void test_COL_SLICE_macro(void) {
    Matrix* matrix = MATRIX(4, 4);
    Matrix* col_slice = COL_SLICE(matrix, 1, 2);  // Slice columns 1 to 2 inclusive

    TEST_ASSERT_NOT_NULL(col_slice);
    TEST_ASSERT_EQUAL_UINT(4, col_slice->rows);
    TEST_ASSERT_EQUAL_UINT(2, col_slice->cols);

    mx_free(matrix);
    mx_free(col_slice);
}

void test_mx_identity_square_matrix(void) {
    size_t rows = 3, cols = 3;
    Matrix* result = mx_identity(rows, cols);

    // Check main diagonal elements
    TEST_ASSERT_EQUAL_DTYPE(1, AT(result, 0, 0));
    TEST_ASSERT_EQUAL_DTYPE(1, AT(result, 1, 1));
    TEST_ASSERT_EQUAL_DTYPE(1, AT(result, 2, 2));

    // Check off-diagonal elements
    TEST_ASSERT_EQUAL_DTYPE(0, AT(result, 0, 1));
    TEST_ASSERT_EQUAL_DTYPE(0, AT(result, 0, 2));
    TEST_ASSERT_EQUAL_DTYPE(0, AT(result, 1, 0));
    TEST_ASSERT_EQUAL_DTYPE(0, AT(result, 1, 2));
    TEST_ASSERT_EQUAL_DTYPE(0, AT(result, 2, 0));
    TEST_ASSERT_EQUAL_DTYPE(0, AT(result, 2, 1));

    mx_free(result);
}

void test_mx_identity_non_square_matrix(void) {
    size_t rows = 3, cols = 2;
    Matrix* result = mx_identity(rows, cols);

    // Check the elements
    TEST_ASSERT_EQUAL_DTYPE(1, AT(result, 0, 0));
    TEST_ASSERT_EQUAL_DTYPE(1, AT(result, 1, 1));

    TEST_ASSERT_EQUAL_DTYPE(0, AT(result, 0, 1));
    TEST_ASSERT_EQUAL_DTYPE(0, AT(result, 1, 0));
    TEST_ASSERT_EQUAL_DTYPE(0, AT(result, 2, 0));
    TEST_ASSERT_EQUAL_DTYPE(0, AT(result, 2, 1));

    mx_free(result);
}

void test_mx_identity_invalid_dimensions(void) {
    Matrix* result = mx_identity(0, 2);
    TEST_ASSERT_NULL(result);

    result = mx_identity(3, 0);
    TEST_ASSERT_NULL(result);

    result = mx_identity(0, 0);
    TEST_ASSERT_NULL(result);
}

void test_basic_equality(void) {
    Matrix* A = MATRIX_WITH(3,3,1); // Initialize matrix A
    Matrix* B = MATRIX_WITH(3,3,1); // Initialize matrix B with the same values as A
    TEST_ASSERT_TRUE(mx_equal(A, B));
    mx_free(A);
    mx_free(B);
}

void test_single_element_difference(void) {
    Matrix* A = MATRIX_WITH(2,1,1); // Initialize matrix A
    Matrix* B = MATRIX_WITH(2,1,1); // Initialize matrix B almost identical to A, but with one differing element
    AT(B, 1,0) = 2;
    TEST_ASSERT_FALSE(mx_equal(A, B));
    mx_free(A);
    mx_free(B);
}

void test_different_sizes(void) {
    Matrix* A = MATRIX_WITH(3,2,1); // Initialize a 2x2 matrix A
    Matrix* B = MATRIX_WITH(3,3,1); // Initialize a 3x3 matrix B
    TEST_ASSERT_FALSE(mx_equal(A, B));
    mx_free(A);
    mx_free(B);
}

void test_same_sizes_different_values(void) {
    Matrix* A = MATRIX_WITH(2,2,1); // Initialize a 2x2 matrix A
    Matrix* B = MATRIX_WITH(2,2,1); // Initialize another 2x2 matrix B with different values
    TEST_ASSERT_FALSE(mx_equal(A, B));
    mx_free(A);
    mx_free(B);
}

void test_invalid_matrices(void) {
    Matrix* A = MATRIX_WITH(2,2,1); // Initialize a valid matrix A
    Matrix* B = NULL;
    TEST_ASSERT_FALSE(mx_equal(A, B));
    mx_free(A);
}


void test_mx_equal_different_dimensions(void) {
    Matrix* mat1 = MATRIX(3, 3);
    Matrix* mat2 = MATRIX(3, 2);
    TEST_ASSERT_FALSE(mx_equal(mat1, mat2));
    mx_free(mat1);
    mx_free(mat2);
}

void test_mx_equal_identical_matrices(void) {
    Matrix* mat1 = MATRIX(3, 3);
    Matrix* mat2 = MATRIX(3, 3);
    // For simplicity, assuming your matrices get initialized to zeros or some default values
    TEST_ASSERT_TRUE(mx_equal(mat1, mat2));
    mx_free(mat1);
    mx_free(mat2);
}

void test_mx_equal_different_values(void) {
    Matrix* mat1 = MATRIX(2, 2);
    Matrix* mat2 = MATRIX(2, 2);
    AT(mat1, 0, 0) = 5;
    AT(mat2, 0, 0) = 6; 
    TEST_ASSERT_FALSE(mx_equal(mat1, mat2));
    mx_free(mat1);
    mx_free(mat2);
}

void test_mx_equal_matrix_with_itself(void) {
    Matrix* mat = MATRIX(3, 3);
    TEST_ASSERT_TRUE(mx_equal(mat, mat));
    mx_free(mat);
}

void test_mx_equal_invalid_matrices(void) {
    Matrix* mat1 = NULL;
    Matrix* mat2 = MATRIX(3, 3);
    TEST_ASSERT_FALSE(mx_equal(mat1, mat2));
    TEST_ASSERT_FALSE(mx_equal(mat2, mat1));
    mx_free(mat2);
}

dtype add_five(dtype x) {
    return x + 5;
}

dtype multiply_by_two(dtype x) {
    return x * 2;
}

dtype identity_function(dtype x) {
    return x;
}

void test_mx_apply_function_add_five(void) {
    Matrix* mat = MATRIX(2, 2);
    AT(mat, 0, 0) = 1;
    AT(mat, 0, 1) = 2;
    AT(mat, 1, 0) = 3;
    AT(mat, 1, 1) = 4;
    mx_apply_function(mat, add_five);

    TEST_ASSERT_EQUAL_DTYPE(6, AT(mat, 0, 0));
    TEST_ASSERT_EQUAL_DTYPE(7, AT(mat, 0, 1));
    TEST_ASSERT_EQUAL_DTYPE(8, AT(mat, 1, 0));
    TEST_ASSERT_EQUAL_DTYPE(9, AT(mat, 1, 1));

    mx_free(mat);
}

void test_mx_apply_function_multiply_by_two(void) {
    Matrix* mat = MATRIX(2, 2);
    AT(mat, 0, 0) = 1;
    AT(mat, 0, 1) = 2;
    AT(mat, 1, 0) = 3;
    AT(mat, 1, 1) = 4;
    mx_apply_function(mat, multiply_by_two);

    TEST_ASSERT_EQUAL_DTYPE(2, AT(mat, 0, 0));
    TEST_ASSERT_EQUAL_DTYPE(4, AT(mat, 0, 1));
    TEST_ASSERT_EQUAL_DTYPE(6, AT(mat, 1, 0));
    TEST_ASSERT_EQUAL_DTYPE(8, AT(mat, 1, 1));

    mx_free(mat);
}

void test_mx_apply_function_identity(void) {
    Matrix* mat = MATRIX(2, 2);
    AT(mat, 0, 0) = 1;
    AT(mat, 0, 1) = 2;
    AT(mat, 1, 0) = 3;
    AT(mat, 1, 1) = 4;
    mx_apply_function(mat, identity_function);

    TEST_ASSERT_EQUAL_DTYPE(1, AT(mat, 0, 0));
    TEST_ASSERT_EQUAL_DTYPE(2, AT(mat, 0, 1));
    TEST_ASSERT_EQUAL_DTYPE(3, AT(mat, 1, 0));
    TEST_ASSERT_EQUAL_DTYPE(4, AT(mat, 1, 1));

    mx_free(mat);
}

void test_mx_apply_function_empty_matrix(void) {
    Matrix* mat = MATRIX(0, 0);
    mx_apply_function(mat, add_five); // Should not crash or have undefined behavior

    mx_free(mat);
}


void test_MATRIX_ONES_lazy_initialization(void) {
    Matrix* ones = MATRIX_ONES(3, 3);
    TEST_ASSERT_NULL(ones->container);
    mx_free(ones);
}

void test_MATRIX_ONES_data_values(void) {
    Matrix* ones = MATRIX_ONES(3, 3);
    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
            TEST_ASSERT_EQUAL_DTYPE(1, AT(ones, i, j));
        }
    }
    mx_free(ones);
}

void test_MATRIX_ONES_metadata(void) {
    Matrix* ones = MATRIX_ONES(3, 4);
    TEST_ASSERT_EQUAL_UINT(3, ones->rows);
    TEST_ASSERT_EQUAL_UINT(4, ones->cols);
    TEST_ASSERT_EQUAL_UINT(1, ones->col_stride);
    TEST_ASSERT_EQUAL_UINT(4, ones->row_stride);
    TEST_ASSERT_EQUAL_UINT(1, ones->default_value);
    TEST_ASSERT_TRUE(CHECK_FLAG(ones->flags, 0)); // Assert lazy flag is set
    mx_free(ones);
}

void test_mx_view_ref_count_increase(void) {
    Matrix* original = MATRIX(3, 3);
    uint16_t initial_ref_count = original->container->ref_count;
    
    Matrix* view = mx_view(original, 3, 3, 1);
    TEST_ASSERT_EQUAL_UINT(initial_ref_count + 1, original->container->ref_count);

    mx_free(original);
    mx_free(view);
}


int main(void) {
    UNITY_BEGIN();

    // initialization
    RUN_TEST(test_basic_matrix_initialization);
    RUN_TEST(test_matrix_initialization_with_value);
    RUN_TEST(test_mx_arrange_memory_layout);
    RUN_TEST(test_mx_arrange_correctness);

    // free
    RUN_TEST(test_basic_free);
    RUN_TEST(test_ref_count_free);
    RUN_TEST(test_data_check);

    // indexing
    RUN_TEST(test_matrix_indexing);

    // container
    RUN_TEST(test_container_check);
    RUN_TEST(test_init_container_successful_allocation);
    RUN_TEST(test_mx_init_successful_allocation);
    RUN_TEST(test_mx_init_zero_value);

    // view
    RUN_TEST(test_MATRIX_VIEW_with_NULL_matrix);
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
    RUN_TEST(test_mx_arrange_3x3_start_from_0);
    RUN_TEST(test_mx_arrange_2x5_start_from_10);
    RUN_TEST(test_mx_arrange_1x1);
    // RUN_TEST(test_mx_arrange_invalid_dimensions);
    RUN_TEST(test_mx_arrange_large_matrix);

    // random
    RUN_TEST(test_mx_rand_basic_properties);
    RUN_TEST(test_mx_rand_value_range);
    RUN_TEST(test_mx_rand_distinct_runs);

    // scale
    RUN_TEST(test_mx_scale_basic_scaling);
    RUN_TEST(test_mx_scale_null_input);
    RUN_TEST(test_mx_scale_ref_count_and_memory);

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
    // RUN_TEST(test_dot_invalid_dimensions);
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

    // identity
    RUN_TEST(test_mx_identity_square_matrix);
    RUN_TEST(test_mx_identity_non_square_matrix);
    RUN_TEST(test_mx_identity_invalid_dimensions);
    
    // equal
    RUN_TEST(test_mx_equal_different_dimensions);
    RUN_TEST(test_mx_equal_identical_matrices);
    RUN_TEST(test_mx_equal_different_values);
    RUN_TEST(test_mx_equal_matrix_with_itself);
    RUN_TEST(test_mx_equal_invalid_matrices);

    // apply function
    RUN_TEST(test_mx_apply_function_add_five);
    RUN_TEST(test_mx_apply_function_multiply_by_two);
    RUN_TEST(test_mx_apply_function_identity);
    RUN_TEST(test_mx_apply_function_empty_matrix);

    // "lazy" matrix
    RUN_TEST(test_MATRIX_ONES_lazy_initialization);
    RUN_TEST(test_MATRIX_ONES_data_values);
    RUN_TEST(test_MATRIX_ONES_metadata);
    RUN_TEST(test_mx_view_ref_count_increase);
    return UNITY_END();
}
