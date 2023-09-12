#include "unity.h"
#include "mx.h" 
#define MX_IMPLEMENTATION

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
            TEST_ASSERT_EQUAL_FLOAT(start_val, AT(m, i, j));
            start_val++;
        }
    }

    mx_free(m);
}


void test_basic_free(void)
{
    Matrix *mat = malloc(sizeof(Matrix));
    mat->container = malloc(sizeof(__matrix_container));
    mat->container->data = malloc(10 * sizeof(float));
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
    mat->container->data = malloc(10 * sizeof(float));
    mat->container->ref_count = 1;

    mx_free(mat);

    // Can't check mat->container->data since it's freed.
    // If we've reached here without crashes, we're good.
}

void test_container_check(void)
{
    Matrix *mat = malloc(sizeof(Matrix));
    mat->container = malloc(sizeof(__matrix_container));
    mat->container->data = malloc(10 * sizeof(float));
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
            TEST_ASSERT_EQUAL_FLOAT(1.0, AT(mat, i, j));
        }
    }

    mx_free(mat);
}

void test_matrix_initialization_with_value(void)
{
    float value = 7.0;
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
            TEST_ASSERT_EQUAL_FLOAT(value, AT(mat, i, j));
        }
    }

    mx_free(mat);
}

void test_matrix_indexing(void)
{
    Matrix *mat = MATRIX(5, 5);
    float value = 5.0;

    // Setting values using the AT macro
    for(size_t i = 0; i < mat->rows; i++){
        for(size_t j = 0; j < mat->cols; j++){
            AT(mat, i, j) = value;
        }
    }

    // Checking values using the AT macro
    for(size_t i = 0; i < mat->rows; i++){
        for(size_t j = 0; j < mat->cols; j++){
            TEST_ASSERT_EQUAL_FLOAT(value, AT(mat, i, j));
        }
    }

    mx_free(mat);
}

void test_init_container_successful_allocation(void) {
    size_t size = 10;
    __matrix_container* container = __init_container(NULL,size);
    
    TEST_ASSERT_NOT_NULL(container);
    TEST_ASSERT_NOT_NULL(container->data);
    TEST_ASSERT_EQUAL_INT(1, container->ref_count);
    
    free(container->data);
    free(container);
}

void test_mx_init_successful_allocation(void) {
    size_t rows = 3;
    size_t cols = 3;
    float init_value = 5.0;
    
    Matrix* mat = __mx_init(NULL,rows, cols, init_value);
    
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
            TEST_ASSERT_EQUAL_FLOAT(init_value, AT(mat, i, j));
        }
    }
    
    mx_free(mat);  // Assuming you have this function to free the matrix.
}

void test_mx_init_zero_value(void) {
    size_t rows = 4;
    size_t cols = 4;
    
    Matrix* mat = __mx_init(NULL,rows, cols, 0);
    
    TEST_ASSERT_NOT_NULL(mat);
    TEST_ASSERT_NOT_NULL(mat->container);
    TEST_ASSERT_NOT_NULL(mat->container->data);
    
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            TEST_ASSERT_EQUAL_FLOAT(0, AT(mat, i, j));
        }
    }
    
    mx_free(mat);
}



void test_MATRIX_VIEW_with_NULL_matrix(void) {
    Matrix* view = MATRIX_VIEW(NULL);

    // Check that the view is not NULL
    TEST_ASSERT_NULL(view);

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
    TEST_ASSERT_EQUAL_FLOAT(7, AT(view, 1, 1)); // Ensure the view reflects changes in the original

    AT(view, 2, 2) = 9;
    TEST_ASSERT_EQUAL_FLOAT(9, AT(mat, 2, 2)); // Ensure the original reflects changes in the view

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

    TEST_ASSERT_EQUAL_FLOAT(1.0, AT(transposed, 0, 0));
    TEST_ASSERT_EQUAL_FLOAT(2.0, AT(transposed, 1, 0));

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
            TEST_ASSERT_EQUAL_FLOAT((i * mat->cols) + j, AT(mat, i, j));
        }
    }
    mx_free(mat);
}

// Test initialization of a 2x5 matrix starting from 10
void test_mx_arrange_2x5_start_from_10(void) {
    float start_value = 10;
    Matrix *mat = mx_arrange(2, 5, start_value);
    TEST_ASSERT_NOT_NULL(mat);
    for (size_t i = 0; i < mat->rows; i++) {
        for (size_t j = 0; j < mat->cols; j++) {
            TEST_ASSERT_EQUAL_FLOAT(start_value, AT(mat, i, j));
            start_value++;
        }
    }
    mx_free(mat);
}

// Test that memory is allocated for a 1x1 matrix
void test_mx_arrange_1x1(void) {
    Matrix *mat = mx_arrange(1, 1, 5);
    TEST_ASSERT_NOT_NULL(mat);
    TEST_ASSERT_EQUAL_FLOAT(5, AT(mat, 0, 0));
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
    float scalar = 2.0;

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

    Matrix* result = ADD(matrix1, matrix2);

    TEST_ASSERT_EQUAL_FLOAT(3, AT(result, 0, 0));
    TEST_ASSERT_EQUAL_FLOAT(5, AT(result, 0, 1));
    TEST_ASSERT_EQUAL_FLOAT(7, AT(result, 1, 0));
    TEST_ASSERT_EQUAL_FLOAT(9, AT(result, 1, 1));

    mx_free(matrix1);
    mx_free(matrix2);
}

void test_add_with_different_rows(void) {
    Matrix* matrix1 = MATRIX(3, 2);
    Matrix* matrix2 = MATRIX(2, 2);

    Matrix* result = ADD(matrix1, matrix2);

    TEST_ASSERT_NULL(result);  // Should be NULL due to dimension mismatch

    mx_free(matrix1);
    mx_free(matrix2);
    mx_free(result);  // Safe to call, as it checks for NULL internally
}

void test_add_with_different_columns(void) {
    Matrix* matrix1 = MATRIX(2, 3);
    Matrix* matrix2 = MATRIX(2, 2);

    Matrix* result = ADD(matrix1, matrix2);

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

    Matrix* result = ADD(matrix1, matrix2);

    for(size_t i = 0; i < result->rows; i++)
        for(size_t j = 0; j < result->cols; j++)
            TEST_ASSERT_EQUAL_FLOAT(0, AT(result, i, j));

    mx_free(matrix1);
    mx_free(matrix2);
}

void test_add_matrix_to_itself(void) {
    Matrix* matrix1 = mx_arrange(2, 2, 1);  // Produces [[1,2], [3,4]]

    Matrix* result = ADD(matrix1, matrix1);

    TEST_ASSERT_EQUAL_FLOAT(2, AT(result, 0, 0));
    TEST_ASSERT_EQUAL_FLOAT(4, AT(result, 0, 1));
    TEST_ASSERT_EQUAL_FLOAT(6, AT(result, 1, 0));
    TEST_ASSERT_EQUAL_FLOAT(8, AT(result, 1, 1));

    mx_free(matrix1);
}

void test_add_null_matrices(void) {
    Matrix* matrix1 = NULL;
    Matrix* matrix2 = MATRIX(2, 2);

    Matrix* result = ADD(matrix1, matrix2);
    TEST_ASSERT_NULL(result);

    result = ADD(matrix2, matrix1);
    TEST_ASSERT_NULL(result);

    mx_free(matrix2);
    mx_free(result);  // Safe to call, as it checks for NULL internally
}

void test_subtract_matching_matrices(void) {
    Matrix* matrix1 = mx_arrange(2, 2, 1);  // Produces [[1,2], [3,4]]
    Matrix* matrix2 = mx_arrange(2, 2, 2);  // Produces [[2,3], [4,5]]

    Matrix* result = mx_subtract(matrix1, matrix2);

    TEST_ASSERT_EQUAL_FLOAT(-1, AT(result, 0, 0));
    TEST_ASSERT_EQUAL_FLOAT(-1, AT(result, 0, 1));
    TEST_ASSERT_EQUAL_FLOAT(-1, AT(result, 1, 0));
    TEST_ASSERT_EQUAL_FLOAT(-1, AT(result, 1, 1));

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
            TEST_ASSERT_EQUAL_FLOAT(0, AT(result, i, j));

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

    Matrix* result = DOT(matrix1, matrix2);

    TEST_ASSERT_EQUAL_FLOAT(22, AT(result, 0, 0));
    TEST_ASSERT_EQUAL_FLOAT(28, AT(result, 0, 1));
    TEST_ASSERT_EQUAL_FLOAT(49, AT(result, 1, 0));
    TEST_ASSERT_EQUAL_FLOAT(64, AT(result, 1, 1));

    mx_free(matrix1);
    mx_free(matrix2);
    mx_free(result);
}

// void test_dot_invalid_dimensions(void) {
//     Matrix* matrix1 = MATRIX(2, 2);
//     Matrix* matrix2 = MATRIX(3, 2);

//     Matrix* result = DOT(matrix1, matrix2);

//     TEST_ASSERT_NULL(result);  // Should be NULL due to dimension mismatch

//     mx_free(matrix1);
//     mx_free(matrix2);
//     mx_free(result);  // Safe to call, as it checks for NULL internally
// }

void test_dot_null_matrices(void) {
    Matrix* matrix1 = NULL;
    Matrix* matrix2 = MATRIX(2, 2);

    Matrix* result = DOT(matrix1, matrix2);
    TEST_ASSERT_NULL(result);

    result = DOT(matrix2, matrix1);
    TEST_ASSERT_NULL(result);

    mx_free(matrix2);
    mx_free(result);  // Safe to call, as it checks for NULL internally
}

void test_dot_matrix_and_its_transpose(void) {
    Matrix* matrix1 = mx_arrange(2, 3, 1);   // Produces [[1,2,3], [4,5,6]]
    Matrix* matrix2 = TRANSPOSE_VIEW(matrix1); // Should produce [[1,4], [2,5], [3,6]]

    Matrix* result = DOT(matrix1, matrix2);

    TEST_ASSERT_EQUAL_FLOAT(14, AT(result, 0, 0));
    TEST_ASSERT_EQUAL_FLOAT(32, AT(result, 0, 1));
    TEST_ASSERT_EQUAL_FLOAT(32, AT(result, 1, 0));
    TEST_ASSERT_EQUAL_FLOAT(77, AT(result, 1, 1));

    mx_free(matrix1);
    mx_free(matrix2);
    mx_free(result);
}

void test_slice_valid_submatrix(void) {
    Matrix* matrix = mx_arrange(4, 4, 1); // Produces a 4x4 matrix with values from 1 to 16

    Matrix* slice = mx_slice(matrix, 1, 2, 1, 2); // 2x2 slice from row 1-2 and col 1-2

    TEST_ASSERT_EQUAL_FLOAT(6, AT(slice, 0, 0));
    TEST_ASSERT_EQUAL_FLOAT(7, AT(slice, 0, 1));
    TEST_ASSERT_EQUAL_FLOAT(10, AT(slice, 1, 0));
    TEST_ASSERT_EQUAL_FLOAT(11, AT(slice, 1, 1));

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
    TEST_ASSERT_EQUAL_FLOAT(1, AT(slice, 0, 0));
    TEST_ASSERT_EQUAL_FLOAT(16, AT(slice, 3, 3));

    mx_free(matrix);
    mx_free(slice);
}

void test_slice_single_row_col(void) {
    Matrix* matrix = mx_arrange(4, 4, 1); // Produces a 4x4 matrix

    Matrix* row_slice = mx_slice(matrix, 2, 2, 0, 3);  // Third row
    Matrix* col_slice = mx_slice(matrix, 0, 3, 2, 2);  // Third column

    TEST_ASSERT_EQUAL_FLOAT(9, AT(row_slice, 0, 0));
    TEST_ASSERT_EQUAL_FLOAT(3, AT(col_slice, 0, 0));

    mx_free(matrix);
    mx_free(row_slice);
    mx_free(col_slice);
}
void test_AT_macro(void) {
    Matrix* matrix = MATRIX(3, 3); // 3x3 matrix with zeroes
    AT(matrix, 1, 1) = 5;

    TEST_ASSERT_EQUAL_FLOAT(5, AT(matrix, 1, 1));

    mx_free(matrix);
}

void test_MATRIX_macro(void) {
    Matrix* matrix = MATRIX(2, 2);

    TEST_ASSERT_NOT_NULL(matrix);
    TEST_ASSERT_EQUAL_FLOAT(0, AT(matrix, 0, 0));

    mx_free(matrix);
}

void test_MATRIX_VIEW_macro(void) {
    Matrix* original = MATRIX(3, 3);
    Matrix* view = MATRIX_VIEW(original);

    TEST_ASSERT_NOT_NULL(view);
    TEST_ASSERT_EQUAL_FLOAT(0, AT(view, 0, 0));

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
    TEST_ASSERT_EQUAL_FLOAT(7, AT(matrix, 0, 0));

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
    size_t rows = 3;
    Matrix* result = mx_identity(rows);

    // Check main diagonal elements
    TEST_ASSERT_EQUAL_FLOAT(1, AT(result, 0, 0));
    TEST_ASSERT_EQUAL_FLOAT(1, AT(result, 1, 1));
    TEST_ASSERT_EQUAL_FLOAT(1, AT(result, 2, 2));

    // Check off-diagonal elements
    TEST_ASSERT_EQUAL_FLOAT(0, AT(result, 0, 1));
    TEST_ASSERT_EQUAL_FLOAT(0, AT(result, 0, 2));
    TEST_ASSERT_EQUAL_FLOAT(0, AT(result, 1, 0));
    TEST_ASSERT_EQUAL_FLOAT(0, AT(result, 1, 2));
    TEST_ASSERT_EQUAL_FLOAT(0, AT(result, 2, 0));
    TEST_ASSERT_EQUAL_FLOAT(0, AT(result, 2, 1));

    mx_free(result);
}

void test_mx_identity_non_square_matrix(void) {
    size_t rows = 3;
    Matrix* result = mx_identity(rows);

    // Check the elements
    TEST_ASSERT_EQUAL_FLOAT(1, AT(result, 0, 0));
    TEST_ASSERT_EQUAL_FLOAT(1, AT(result, 1, 1));

    TEST_ASSERT_EQUAL_FLOAT(0, AT(result, 0, 1));
    TEST_ASSERT_EQUAL_FLOAT(0, AT(result, 1, 0));
    TEST_ASSERT_EQUAL_FLOAT(0, AT(result, 2, 0));
    TEST_ASSERT_EQUAL_FLOAT(0, AT(result, 2, 1));

    mx_free(result);
}

void test_mx_identity_invalid_dimensions(void) {
    Matrix* result = mx_identity(0);
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

float add_five(float x) {
    return x + 5;
}

float multiply_by_two(float x) {
    return x * 2;
}

float identity_function(float x) {
    return x;
}

void test_mx_apply_function_add_five(void) {
    Matrix* mat = MATRIX(2, 2);
    AT(mat, 0, 0) = 1;
    AT(mat, 0, 1) = 2;
    AT(mat, 1, 0) = 3;
    AT(mat, 1, 1) = 4;
    mx_apply_function(mat, add_five);

    TEST_ASSERT_EQUAL_FLOAT(6, AT(mat, 0, 0));
    TEST_ASSERT_EQUAL_FLOAT(7, AT(mat, 0, 1));
    TEST_ASSERT_EQUAL_FLOAT(8, AT(mat, 1, 0));
    TEST_ASSERT_EQUAL_FLOAT(9, AT(mat, 1, 1));

    mx_free(mat);
}

void test_mx_apply_function_multiply_by_two(void) {
    Matrix* mat = MATRIX(2, 2);
    AT(mat, 0, 0) = 1;
    AT(mat, 0, 1) = 2;
    AT(mat, 1, 0) = 3;
    AT(mat, 1, 1) = 4;
    mx_apply_function(mat, multiply_by_two);

    TEST_ASSERT_EQUAL_FLOAT(2, AT(mat, 0, 0));
    TEST_ASSERT_EQUAL_FLOAT(4, AT(mat, 0, 1));
    TEST_ASSERT_EQUAL_FLOAT(6, AT(mat, 1, 0));
    TEST_ASSERT_EQUAL_FLOAT(8, AT(mat, 1, 1));

    mx_free(mat);
}

void test_mx_apply_function_identity(void) {
    Matrix* mat = MATRIX(2, 2);
    AT(mat, 0, 0) = 1;
    AT(mat, 0, 1) = 2;
    AT(mat, 1, 0) = 3;
    AT(mat, 1, 1) = 4;
    mx_apply_function(mat, identity_function);

    TEST_ASSERT_EQUAL_FLOAT(1, AT(mat, 0, 0));
    TEST_ASSERT_EQUAL_FLOAT(2, AT(mat, 0, 1));
    TEST_ASSERT_EQUAL_FLOAT(3, AT(mat, 1, 0));
    TEST_ASSERT_EQUAL_FLOAT(4, AT(mat, 1, 1));

    mx_free(mat);
}

void test_mx_apply_function_empty_matrix(void) {
    TEST_IGNORE();
    // Matrix* mat = MATRIX(0, 0);
    // mx_apply_function(mat, add_five); // Should not crash or have undefined behavior

    // mx_free(mat);
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
            TEST_ASSERT_EQUAL_FLOAT(1, AT(ones, i, j));
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

void test_mx_init_with_null_array(void) {
    Matrix *mat = __MATRIX_FROM(NULL, 2, 2);
    TEST_ASSERT_NOT_NULL(mat);
    TEST_ASSERT_EQUAL(2, mat->rows);
    TEST_ASSERT_EQUAL(2, mat->cols);
    mx_free(mat);
    // ADD any other checks relevant to the initial state
}

void test_mx_init_with_static_array(void) {
    float sampleArray[2][2] = {{1, 2}, {3, 4}};
    Matrix *mat = __MATRIX_FROM((float *)sampleArray, 2, 2);
    TEST_ASSERT_NOT_NULL(mat);
    TEST_ASSERT_EQUAL(2, mat->rows);
    TEST_ASSERT_EQUAL(2, mat->cols);
    TEST_ASSERT_EQUAL_FLOAT(1, AT(mat, 0, 0));
    TEST_ASSERT_EQUAL_FLOAT(2, AT(mat, 0, 1));
    TEST_ASSERT_EQUAL_FLOAT(3, AT(mat, 1, 0));
    TEST_ASSERT_EQUAL_FLOAT(4, AT(mat, 1, 1));
    mx_free(mat);
}

void test_mx_init_with_zero_dimensions(void) {
    Matrix *mat = __MATRIX_FROM(NULL, 0, 0);
    TEST_ASSERT_NULL(mat);
    mx_free(mat);
}

void test_init_container_with_null_array(void) {
    Matrix *mat = __MATRIX_FROM(NULL, 1,4);
    TEST_ASSERT_NOT_NULL(mat);
    TEST_ASSERT_EQUAL(1, mat->container->ref_count);
    mx_free(mat);
}

void test_init_matrix_with_static_array(void) {
    float sampleArray[4] = {1, 2, 3, 4};
    Matrix *m = __MATRIX_FROM(sampleArray, 4,1);
    TEST_ASSERT_NOT_NULL(m);
    TEST_ASSERT_EQUAL(1, m->container->ref_count);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(sampleArray, m->container->data, 4);
    mx_free(m);
}

void test_init_container_with_zero_size(void) {
    Matrix *container = __MATRIX_FROM(NULL, 0,0);
    TEST_ASSERT_NULL(container);
    mx_free(container);
}

void test_init_matrix_with_dynamic_array(void) {
    float* sampleArray = malloc(4 * sizeof(float));
    if (sampleArray) {
        sampleArray[0] = 1;
        sampleArray[1] = 2;
        sampleArray[2] = 3;
        sampleArray[3] = 4;

        Matrix *m = __MATRIX_FROM(sampleArray, 4, 1);
        TEST_ASSERT_NOT_NULL(m);
        TEST_ASSERT_EQUAL(1, m->container->ref_count);
        TEST_ASSERT_EQUAL_FLOAT_ARRAY(sampleArray, m->container->data, 4);

        // Since the __MATRIX_FROM function copies over the data and 
        // always allocates its own memory, you can free the dynamic array here.
        free(sampleArray);

        mx_free(m);
    }
}

void test_self_dot_product_with_valid_row_vector(void) {
    float data[3] = {1.0, 2.0, 3.0};
    Matrix *row_vector = __MATRIX_FROM(data, 1, 3);
    
    float result = mx_self_dot_product(row_vector);
    
    TEST_ASSERT_FLOAT_WITHIN(0.0001, 14.0, result);  // Expected value is 1^2 + 2^2 + 3^2 = 14
    
    mx_free(row_vector);
}

void test_self_dot_product_with_valid_column_vector(void) {
    float data[3] = {1.0, 2.0, 3.0};
    Matrix *col_vector = __MATRIX_FROM(data, 3, 1);
    
    float result = mx_self_dot_product(col_vector);
    
    TEST_ASSERT_FLOAT_WITHIN(0.0001, 14.0, result);
    
    mx_free(col_vector);
}

void test_self_dot_product_with_invalid_2D_matrix(void) {
    float data[4] = {1.0, 2.0, 3.0, 4.0};
    Matrix *matrix = __MATRIX_FROM(data, 2, 2);
    
    float result = mx_self_dot_product(matrix);
    
    TEST_ASSERT_EQUAL_FLOAT(-1.0, result);  // Assuming -1.0 is returned for invalid matrices
    
    mx_free(matrix);
}

void test_self_dot_product_with_null_vector(void) {
    float result = mx_self_dot_product(NULL);
    
    TEST_ASSERT_EQUAL_FLOAT(-1.0, result);
}


void test_vector_length_for_valid_vector(void) {
    // Test 1: Vector with 3 elements: [3, 4, 0]
    Matrix* vector1 = MATRIX_WITH(3, 1, 0);
    AT(vector1, 0, 0) = 3;
    AT(vector1, 1, 0) = 4;
    TEST_ASSERT_EQUAL_FLOAT(5.0, mx_length(vector1));
    mx_free(vector1);
}

void test_vector_length_for_zero_vector(void) {
    // Test 2: Vector with 2 elements: [0, 0]
    Matrix* vector2 = MATRIX_WITH(2, 1, 0);
    TEST_ASSERT_EQUAL_FLOAT(0.0, mx_length(vector2));
    mx_free(vector2);
}

void test_vector_length_for_non_vector_matrix(void) {
    Matrix* matrix = MATRIX_WITH(2, 2, 5);
    TEST_ASSERT_EQUAL_FLOAT(10, mx_length(matrix)); 
    mx_free(matrix);
}
void test_vector_length_for_4dim_vector_matrix(void) {
    // Test 4: Vector with 4x1 dimensions 
    Matrix* matrix = MATRIX_WITH(4,1, 1);
    TEST_ASSERT_EQUAL_FLOAT(2, mx_length(matrix));  // Assuming -1 is the error value
    mx_free(matrix);
}

void test_unit_vector_length(void){
    Matrix* unit_vector = MATRIX_IDENTITY(1);
    float length = mx_length(unit_vector);
    TEST_ASSERT_EQUAL_FLOAT(1, length);
    mx_free(unit_vector);
}

void test_cosine_between_two_vectors(void) {
    Matrix* m = MATRIX(3,1);
    AT(m,0,0) = 12;
    AT(m,1,0) = 23;
    AT(m,2,0) = 511;

    Matrix* m1 = MATRIX(3,1);
    AT(m1,0,0) = 9;
    AT(m1,1,0) = -1;
    AT(m1,2,0) = -123;

    float cosine = mx_cosine_between_two_vectors(m, m1);

    TEST_ASSERT_FLOAT_WITHIN(0.000001, -0.994671, cosine);
    mx_free(m);
    mx_free(m1);
}

void test_memory_allocation_and_deallocation(void) {
    Matrix* m = MATRIX(5,5);
    TEST_ASSERT_NOT_NULL(m);

    mx_free(m);
    // Perhaps ADD a custom function to validate that memory was actually freed
}

void test_invalid_matrix_dimensions(void) {
    Matrix* m = MATRIX(0,5); // Invalid rows
    TEST_ASSERT_NULL(m);

    Matrix* n = MATRIX(5,0); // Invalid columns
    TEST_ASSERT_NULL(n);
}

void test_cosine_of_orthogonal_vectors(void) {
    Matrix* m = MATRIX(3,1);
    AT(m,0,0) = 1;
    AT(m,1,0) = 0;
    AT(m,2,0) = 0;

    Matrix* m1 = MATRIX(3,1);
    AT(m1,0,0) = 0;
    AT(m1,1,0) = 1;
    AT(m1,2,0) = 0;

    float cosine = mx_cosine_between_two_vectors(m, m1);
    TEST_ASSERT_FLOAT_WITHIN(0.000001, 0.0, cosine);  // Orthogonal vectors

    mx_free(m);
    mx_free(m1);
}

void test_shwarz_inequality(void) {
    // Schwarz inequality |v * w| <= ||v|| * ||w||
    Matrix* rand1 = MATRIX(1, 3);
    Matrix* rand2 = MATRIX(1, 3);

    AT(rand1, 0, 0) = 1;
    AT(rand1, 0, 1) = 2;
    AT(rand1, 0, 2) = 4;

    AT(rand2, 0, 0) = 4;
    AT(rand2, 0, 1) = 2;
    AT(rand2, 0, 2) = 13;
    
    Matrix* dot = DOT(rand1, rand2);
    float length1 = mx_length(rand1);
    float length2 = mx_length(rand2);
    float result = length1 * length2;

    // Check that dot product is less than or equal to the product of lengths
    TEST_ASSERT_LESS_OR_EQUAL(result,AT(dot, 0, 0));

    // Clean up
    // Free your matrices if necessary
    mx_free(rand1);
    mx_free(rand2);
    mx_free(dot);
}

void problem121(void){
    Matrix* u = MATRIX(1,2);
    AT(u,0,0) = -6;
    AT(u,0,1) = 8;
    
    Matrix* v = MATRIX(1,2);
    AT(v,0,0) = 3;
    AT(v,0,1) = 4;
    
    Matrix* w = MATRIX(1,2);
    AT(w,0,0) = 8;
    AT(w,0,1) = 6;

    Matrix* uv_dot = DOT(u,v);
    TEST_ASSERT_EQUAL(AT(uv_dot,0,0), 14);
    Matrix* uw_dot = DOT(u,w);
    TEST_ASSERT_EQUAL(AT(uw_dot,0,0), 0);
    Matrix* uw_add = MATRIX_COPY(u);
    uw_add = ADD(uw_add,w);
    TEST_ASSERT_EQUAL(AT(uw_add,0,0), 2);
    TEST_ASSERT_EQUAL(AT(uw_add,0,1), 14);
    Matrix* u_dot_uw_add = DOT(u,uw_add);
    TEST_ASSERT_EQUAL(AT(u_dot_uw_add,0,0), 100);
    
    mx_free(u);
    mx_free(v);
    mx_free(w);
    mx_free(uw_dot);
    mx_free(uv_dot);
    mx_free(uw_add);
    mx_free(u_dot_uw_add); 
}

void problem122(void){
    Matrix* u = MATRIX(1,2);
    AT(u,0,0) = -6;
    AT(u,0,1) = 8;
    
    Matrix* v = MATRIX(1,2);
    AT(v,0,0) = 3;
    AT(v,0,1) = 4;
    
    Matrix* w = MATRIX(1,2);
    AT(w,0,0) = 8;
    AT(w,0,1) = 6;

    float length_u = mx_length(u);
    float length_v = mx_length(v);
    float length_w = mx_length(w);

    TEST_ASSERT_EQUAL(length_u, 10);
    TEST_ASSERT_EQUAL(length_v, 5);
    TEST_ASSERT_EQUAL(length_w, 10);

    Matrix* u_dot_v = DOT(u,v);
    Matrix* u_dot_w = DOT(u,w);
    
    //Shwarz inequality

    TEST_ASSERT_LESS_OR_EQUAL(length_u*length_v, AT(u_dot_v,0,0));
    TEST_ASSERT_LESS_OR_EQUAL(length_u*length_w, AT(u_dot_w,0,0));
    
    mx_free(u);
    mx_free(v);
    mx_free(w);
    mx_free(u_dot_v);
    mx_free(u_dot_w);
}

void problem123(void){
    
    Matrix* v = MATRIX(1,2);
    AT(v,0,0) = 3;
    AT(v,0,1) = 4;
    
    Matrix* w = MATRIX(1,2);
    AT(w,0,0) = 8;
    AT(w,0,1) = 6;

    Matrix* v_unit_vector = UNIT_VECTOR_FROM(v);
    Matrix* w_unit_vector = UNIT_VECTOR_FROM(w);

    TEST_ASSERT_EQUAL(3/5, AT(v_unit_vector,0,0));
    TEST_ASSERT_EQUAL(4/5, AT(v_unit_vector,0,1));
    TEST_ASSERT_EQUAL(8/10, AT(w_unit_vector,0,0));
    TEST_ASSERT_EQUAL(6/10, AT(w_unit_vector,0,1));

    float cosine = mx_cosine_between_two_vectors(v,w);

    TEST_ASSERT_EQUAL(48/50, cosine);

    mx_free(v);
    mx_free(w);
    mx_free(v_unit_vector);
    mx_free(w_unit_vector);
    
}

void problem124(void){
    Matrix* u = MATRIX(1,2);
    AT(u,0,0) = -6;
    AT(u,0,1) = 8;
    
    Matrix* v = MATRIX(1,2);
    AT(v,0,0) = 3;
    AT(v,0,1) = 4;
    
    Matrix* w = MATRIX(1,2);
    AT(w,0,0) = 8;
    AT(w,0,1) = 6;

    Matrix* minus_u = SCALAR_DOT(u, -1);
    Matrix* dot = DOT(u,minus_u);
    TEST_ASSERT_EQUAL(-100, AT(dot,0,0));
    Matrix* v_plus_w = MATRIX_COPY(v);
    v_plus_w = ADD(v_plus_w,w);
    Matrix* v_minus_w = mx_subtract(v,w);
    Matrix* v_pm_dot_w = DOT(v_plus_w,v_minus_w);
    TEST_ASSERT_EQUAL(-75, AT(v_pm_dot_w,0,0));

    Matrix* w_scaled = SCALAR_DOT(w,2);
    Matrix* v_plus_scaled_w = MATRIX_COPY(v);
    v_plus_scaled_w = ADD(v_plus_scaled_w,w_scaled);
    Matrix* v_minus_scaled_w = mx_subtract(v,w_scaled);
    Matrix* v_ps_ms_dot_w = DOT(v_plus_scaled_w,v_minus_scaled_w);
    TEST_ASSERT_EQUAL(-375, AT(v_ps_ms_dot_w,0,0));

    mx_free(u);
    mx_free(v);
    mx_free(w);
    mx_free(minus_u);
    mx_free(dot);
    mx_free(v_plus_w);
    mx_free(v_minus_w);
    mx_free(v_pm_dot_w);
    mx_free(w_scaled);
    mx_free(v_plus_scaled_w);
    mx_free(v_minus_scaled_w);
    mx_free(v_ps_ms_dot_w);
}

void problem125(void){
    Matrix* u = MATRIX(1,2);
    AT(u,0,0) = 3;
    AT(u,0,1) = 1;
    
    Matrix* w = MATRIX(1,3);
    AT(w,0,0) = 2;
    AT(w,0,1) = 1;
    AT(w,0,2) = 2;

    Matrix* u_unit_vector = UNIT_VECTOR_FROM(u);
    Matrix* w_unit_vector = UNIT_VECTOR_FROM(w);

    TEST_ASSERT_EQUAL_FLOAT(3/sqrt(10), AT(u_unit_vector,0,0));
    TEST_ASSERT_EQUAL_FLOAT(1/sqrt(10), AT(u_unit_vector,0,1));

    TEST_ASSERT_EQUAL_FLOAT((float)(2.0/3), AT(w_unit_vector,0,0));
    TEST_ASSERT_EQUAL_FLOAT((float)(1.0/3), AT(w_unit_vector,0,1));
    TEST_ASSERT_EQUAL_FLOAT((float)(2.0/3), AT(w_unit_vector,0,2));

    Matrix* u_perpendicular = mx_perpendicular(u);
    Matrix* w_perpendicular = mx_perpendicular(w);
    // should be 0
    Matrix* u_perpendicular_dot = DOT(u,u_perpendicular);
    Matrix* w_perpendicular_dot = DOT(w,w_perpendicular);

    TEST_ASSERT_EQUAL(0,AT(u_perpendicular_dot,0,0));
    TEST_ASSERT_EQUAL(0,w_perpendicular_dot);

    mx_free(u_perpendicular);
    mx_free(w_perpendicular);
    mx_free(w_perpendicular_dot);
    mx_free(u_perpendicular_dot);
    mx_free(u);
    mx_free(w);
    mx_free(u_unit_vector);
    mx_free(w_unit_vector);
}

void problem127(void){
    Matrix* v = MATRIX(1,2);
    AT(v,0,0) = 1;
    AT(v,0,1) = sqrt(3);
    
    Matrix* w = MATRIX(1,3);
    AT(w,0,0) = 1;
    AT(w,0,1) = 0;

    Matrix* v_w_dot = DOT(v,w);

    float v_length = mx_length(v);
    float w_length = mx_length(w);

    float cosine = AT(v_w_dot,0,0)/(v_length*w_length);
    
    TEST_ASSERT_EQUAL(0.5, cosine);

    mx_free(v);
    mx_free(w);
    mx_free(v_w_dot);
}

void problem126(void){
    Matrix* u = MATRIX(1,2);
    AT(u,0,0) = 3;
    AT(u,0,1) = 1;

    Matrix* perpendicular_1 = MATRIX_COPY(u);
    // 1st
    swap(&AT(perpendicular_1,0,0), &AT(perpendicular_1,0,1));
    AT(perpendicular_1,0,0) *= -1;

    // 2nd 
    Matrix* perpendicular_2 = MATRIX_COPY(u);
    swap(&AT(perpendicular_2,0,0), &AT(perpendicular_2,0,1));
    AT(perpendicular_2,0,1) *= -1;
    
    Matrix* u_dot_perpendicular_1 = DOT(u, perpendicular_1);
    TEST_ASSERT_EQUAL_FLOAT(0, AT(u_dot_perpendicular_1,0,0));
    Matrix* u_dot_perpendicular_2 = DOT(u, perpendicular_2);
    TEST_ASSERT_EQUAL_FLOAT(0, AT(u_dot_perpendicular_2,0,0));

    mx_free(u);
    mx_free(perpendicular_1);
    mx_free(perpendicular_2);
    mx_free(u_dot_perpendicular_1);
    mx_free(u_dot_perpendicular_2);
}

void rules1219(void){
    // v*w=w*v
    Matrix* v = MATRIX_RAND(1,2);
    Matrix* w = MATRIX_RAND(1,2);

    Matrix* v_dot_w = DOT(v,w);
    Matrix* w_dot_v = DOT(w,v);    
    TEST_ASSERT_EQUAL_FLOAT(AT(v_dot_w,0,0), AT(w_dot_v,0,0));
    mx_free(w_dot_v);

    // u(w+v) = uw+uv
    Matrix* u = MATRIX_RAND(1,2);
    Matrix* v_plus_w = MATRIX_COPY(v);
    v_plus_w = ADD(v_plus_w,w);
    Matrix* u_dot_vpw = DOT(u,v_plus_w);
    Matrix* v_dot_v_p_w = DOT(v, v_plus_w);
    Matrix* u_dot_v = DOT(u,v);
    Matrix* u_dot_w = DOT(u,w);
    Matrix* udv_plus_udw = MATRIX_COPY(u_dot_v);
    udv_plus_udw = ADD(udv_plus_udw, u_dot_w);
    TEST_ASSERT_EQUAL(AT(u_dot_vpw,0,0), AT(udv_plus_udw,0,0));
    mx_free(v_plus_w);
    mx_free(u_dot_vpw);
    mx_free(v_dot_v_p_w);
    mx_free(u_dot_v);
    mx_free(u_dot_w);
    mx_free(udv_plus_udw);
    mx_free(u);

    // (cv)*w=c(v*w)
    float c = 12.0;
    Matrix* cv = SCALAR_DOT(v,c);
    Matrix* cv_dot_w = DOT(cv, w);
    Matrix* c_dot_cdw = SCALAR_DOT(v_dot_w, c);
    TEST_ASSERT_EQUAL_FLOAT(AT(cv_dot_w,0,0), AT(c_dot_cdw,0,0));
    mx_free(v_dot_w);
    mx_free(cv);
    mx_free(cv_dot_w);
    mx_free(c_dot_cdw);

    mx_free(v);
    mx_free(w);
    
}

// ||v+w||^2=vv+2vw+ww
void rules12192(void){
    Matrix* v = MATRIX_RAND(1,2);
    Matrix* w = MATRIX_RAND(1,2);

    Matrix* u = MATRIX_COPY(v);
    u = ADD(u,w);

    float length = mx_length(u);
    float length_squared = length*length;
    
    Matrix* v_squared = DOT(v,v);
    Matrix* w_squared = DOT(w,w);

    Matrix* v_dot_w = DOT(v,w);

    Matrix* v_dot_w_scaled = SCALAR_DOT(v_dot_w,2);

    float v_squared_scalar = AT(v_squared,0,0);
    float w_squared_scalar = AT(w_squared,0,0);
    float vw_scalar = AT(v_dot_w_scaled,0,0);

    float result = v_squared_scalar+w_squared_scalar+vw_scalar;

    TEST_ASSERT_EQUAL_FLOAT(length_squared, result);

    mx_free(v);
    mx_free(w);
    mx_free(u);
    mx_free(v_squared);
    mx_free(w_squared);
    mx_free(v_dot_w);
    mx_free(v_dot_w_scaled);
}

// ||u-w||^2 = ||u||^2 - 2||u||*||w||*cos(θ) + ||w||^2
void rules1220(void){
    Matrix* u = MATRIX_RAND(1,2);
    Matrix* w = MATRIX_RAND(1,2);

    Matrix* u_subtract_w= mx_subtract(u,w);

    float u_s_w_length = mx_length(u_subtract_w);

    float usw_length_squared = u_s_w_length*u_s_w_length;

    float u_length = mx_length(u);

    float w_length = mx_length(w);

    float cos_u_w = mx_cosine_between_two_vectors(u,w);

    float result = u_length*u_length - 2 * u_length * w_length * cos_u_w + w_length*w_length;

    TEST_ASSERT_EQUAL_FLOAT(usw_length_squared, result);

    mx_free(u);
    mx_free(w);
    mx_free(u_subtract_w); 
}

void test_nn_allocation_and_freeing(void) {
    size_t arch[] = {3, 2, 2}; 
    NN* test_nn = NN(arch);
    
    // Check if allocation was successful
    TEST_ASSERT_NOT_NULL_MESSAGE(test_nn, "NN Allocation Failed");
    
    // Check count
    TEST_ASSERT_EQUAL_INT_MESSAGE(2, test_nn->count, "NN Count Mismatch");
    
    // Check if inner arrays are allocated
    TEST_ASSERT_NOT_NULL_MESSAGE(test_nn->ws, "Weight Matrices Allocation Failed");
    TEST_ASSERT_NOT_NULL_MESSAGE(test_nn->bs, "Bias Vectors Allocation Failed");
    TEST_ASSERT_NOT_NULL_MESSAGE(test_nn->as, "Activation Vectors Allocation Failed");
    
    // Free and test if it works without issues
    mx_nn_free(test_nn);
}

void problem130(void){
    float array_data[] = {1, 1, 1}; 
    Matrix* s = MATRIX_FROM_ARRAY(array_data);
    float array_data2[] = {0, 1, 1}; 
    Matrix* s2 = MATRIX_FROM_ARRAY(array_data2);
    float array_data3[] = {0, 0, 1}; 
    Matrix* s3 = MATRIX_FROM_ARRAY(array_data3);



    float array_data4[] = {2, 3, 4}; 
    Matrix* x = MATRIX_FROM_ARRAY(array_data4);

    mx_free(s);
    mx_free(s2);
    mx_free(s3);
    mx_free(x);
}

void test_nn_allocation_with_valid_arch(void) {
    size_t arch[] = {3, 2, 2}; 
    NN* test_nn = NN(arch);

    TEST_ASSERT_NOT_NULL_MESSAGE(test_nn, "NN Allocation Failed for Valid Architecture");
    mx_nn_free(test_nn);
}

void test_freeing_valid_nn(void) {
    size_t arch[] = {3, 2, 2}; 
    NN* test_nn = NN(arch);

    // Assuming this function doesn't crash, the test will pass.
    mx_nn_free(test_nn);
}

void test_freeing_null_nn(void) {
    NN* test_nn = NULL;
    
    // Shouldn't crash.
    mx_nn_free(test_nn);
}


float forward_xor(NN *xor){
    for(size_t i = 0; i < xor->count; ++i){
        mx_free(xor->as[i+1]);
        xor->as[i+1] = DOT(xor->as[i],xor->ws[i]);
        ADD(xor->as[i+1],xor->bs[i]);
        mx_apply_function(xor->as[i+1], sigmoidf);
    }
    return SCALAR(xor->as[xor->count]);
}

float cost(NN* m, Matrix* ti, Matrix* to){
    assert(ti->rows==to->rows);
    assert(to->cols == m->as[m->count]->cols); 
    size_t n = ti->rows;
    float c = 0;
    for(size_t i = 0; i< n; ++i){
        Matrix* x = ROW_SLICE(ti,i,i);
        Matrix* y = ROW_SLICE(to,i,i);
        mx_free(m->as[0]);
        m->as[0] = x;
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

void finite_difference(NN* m, NN* g,float eps, Matrix* ti, Matrix* to){
    float saved;
    float c = cost(m, ti,to);

    for(size_t d = 0; d < m->count; ++d){
        for(size_t i=0; i< m->ws[d]->rows;++i){
            for(size_t j = 0; j < m->ws[d]->cols; ++j){
                saved = AT(m->ws[d],i,j);
                AT(m->ws[d],i,j) += eps;
                AT(g->ws[d],i,j) = (cost(m, ti, to)-c)/eps;
                AT(m->ws[d],i,j) = saved;            
            }
        }
        for(size_t i=0; i< m->bs[d]->rows;++i){
            for(size_t j = 0; j < m->bs[d]->cols; ++j){
                saved = AT(m->bs[d],i,j);
                AT(m->bs[d],i,j) += eps;
                AT(g->bs[d],i,j) = (cost(m, ti, to)-c)/eps;
                AT(m->bs[d],i,j) = saved;            
            }
        }
    }
}

void learn(NN* m, NN* g, float rate){
    for(size_t d = 0; d < m->count; ++d){
        for(size_t i=0; i< m->ws[d]->rows;++i){
            for(size_t j = 0; j < m->ws[d]->cols; ++j){
                AT(m->ws[d],i,j) -= rate*AT(g->ws[d],i,j);
            }
        }
        for(size_t i=0; i< m->bs[d]->rows;++i){
            for(size_t j = 0; j < m->bs[d]->cols; ++j){
                AT(m->bs[d],i,j) -= rate*AT(g->bs[d],i,j);
            }
        } 
    }
}

// Gradient descent with finite difference. This function should not have memory leaks
void test_gradient_descent(void){
    size_t arch[] = {2,2,1};
    NN* xor = NN(arch);
    NN* gradient = NN(arch);
    mx_nn_set_to_rand(xor,0,1);
    mx_nn_set_to_rand(gradient,0,1);
    Matrix* xor_data = open_dataset("./datasets/XOR");
    Matrix* ti = COL_SLICE(xor_data,0,1);
    Matrix* to = COL_SLICE(xor_data,2,2);

    float eps = 1e-1;
    float rate = 1;
    for(size_t i = 0; i<1500; ++i){
        finite_difference(xor,gradient,eps, ti, to);
        learn(xor,gradient,rate);
    }
    for(size_t i = 0; i < 2; ++i){
        for(size_t j = 0; j < 2; ++j){
            AT(xor->as[0],0,0) = i;
            AT(xor->as[0],0,1) = j;
            forward_xor(xor);
            float y = SCALAR(xor->as[xor->count]);
            TEST_ASSERT_EQUAL_FLOAT(i^j, round(y));
        }
    }
    mx_nn_free(xor);
    mx_nn_free(gradient);
    mx_free(xor_data);
    mx_free(ti);
    mx_free(to);
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

    // init from array
    RUN_TEST(test_mx_init_with_null_array);
    RUN_TEST(test_mx_init_with_static_array);
    RUN_TEST(test_mx_init_with_zero_dimensions);
    RUN_TEST(test_init_container_with_null_array);
    RUN_TEST(test_init_matrix_with_static_array);
    RUN_TEST(test_init_container_with_zero_size);
    RUN_TEST(test_init_matrix_with_dynamic_array);

    // self-dot
    RUN_TEST(test_self_dot_product_with_valid_row_vector);
    RUN_TEST(test_self_dot_product_with_valid_column_vector);
    RUN_TEST(test_self_dot_product_with_invalid_2D_matrix);
    RUN_TEST(test_self_dot_product_with_null_vector);

    // vector length
    RUN_TEST(test_vector_length_for_valid_vector);
    RUN_TEST(test_vector_length_for_zero_vector);
    RUN_TEST(test_vector_length_for_non_vector_matrix);
    RUN_TEST(test_unit_vector_length);

    // cosine between vectors
    RUN_TEST(test_cosine_between_two_vectors);
    RUN_TEST(test_memory_allocation_and_deallocation);
    RUN_TEST(test_invalid_matrix_dimensions);

    // Gilbert Strang Introduction to Linear Algebra 4th edition
    // Problem set 1.2 
    RUN_TEST(test_shwarz_inequality);
    RUN_TEST(problem121);
    RUN_TEST(problem122);
    RUN_TEST(problem123);
    RUN_TEST(problem124);
    RUN_TEST(problem125);
    RUN_TEST(problem126);
    RUN_TEST(problem127);
    RUN_TEST(rules1219);
    RUN_TEST(rules12192);
    RUN_TEST(rules1220);

    // Problem set 1.3 
    RUN_TEST(problem130);

    // NN layers
    RUN_TEST(test_nn_allocation_and_freeing);
    RUN_TEST(test_nn_allocation_with_valid_arch);
    RUN_TEST(test_freeing_valid_nn);
    RUN_TEST(test_freeing_null_nn);

    // Gradient descent
    RUN_TEST(test_gradient_descent);

    return UNITY_END();
}
