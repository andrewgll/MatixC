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

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_mat_arrange_memory_layout);
    RUN_TEST(test_mat_arrange_correctness);

    return UNITY_END();
}
