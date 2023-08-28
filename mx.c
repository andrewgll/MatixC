
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <errno.h>

#include "mx.h"

dtype sigmoidf(dtype value){
    return 1.0/(1+expf(-value));
}

void mx_free(Matrix *matrix) {
    if (matrix)  {
        if(matrix->container){
            matrix->container->ref_count--;
            if (matrix->container->ref_count == 0) {
                if (matrix->container->data) {
                    free(matrix->container->data);
                    matrix->container->data = NULL;
                }
                free(matrix->container);
                matrix->container = NULL;
            }
        }
        free(matrix);  
    }
}

void* mx_apply_function(Matrix* matrix, dtype (*func)(dtype)) {
    CHECK_MATRIX_VALIDITY(matrix);

    for(size_t i = 0; i < matrix->rows; i++) {
        for(size_t j = 0; j < matrix->cols; j++) {
            AT(matrix,i,j)= func(AT(matrix,i,j));
        }
    }
    return NULL;
}

__matrix_container* __init_container(dtype* array, size_t size) {
    if(size == 0){
        return NULL;
    }
    
    __matrix_container* container = malloc(sizeof(__matrix_container));
    if (!container) {
        return NULL;
    }

    container->ref_count = 1;

    // Always allocate memory on the heap
    container->data = calloc(size, sizeof(dtype));

    if (!container->data) {
        free(container);
        return NULL;
    }

    // If an external array is provided, copy the data over
    if (array) {
        memcpy(container->data, array, size * sizeof(dtype));
    }

    return container;
}

Matrix* mx_copy(const Matrix* src){
    CHECK_MATRIX_VALIDITY(src);

    Matrix* copy = MATRIX(src->rows, src->cols);
    if(!copy){
        printf("Failed to create matrix.");
        return NULL;
    }
    memcpy(copy->container->data, src->container->data, sizeof(dtype) * src->rows*src->cols);
    return copy;
    
}
Matrix* __mx_init(dtype* array, size_t rows, size_t cols, dtype init_value) {

    if(!VALID_DIMENSIONS(rows, cols)){
        printf("Invalid matrix dimensions.");
        return NULL;
    }
    
    Matrix* mat = (Matrix*)malloc(sizeof(Matrix));
    if (!mat) {
        return NULL;
    }

    // Pass array to init_container
    if(array){
        mat->container = __init_container(array, cols * rows);
    }
    else{
        mat->container = __init_container(NULL, cols * rows);
    }
    if (!mat->container) {
        free(mat);
        return NULL;
    }

    mat->rows = rows;
    mat->cols = cols;
    mat->row_stride = cols; 
    mat->col_stride = 1;

    // TODO How about lazy matrix view by default?
    mat->default_value = init_value;
    mat->flags = 0;

    // Initialize only if the value is non-zero and if an external array was not provided
    if(init_value != 0 && !array){
        for(size_t i = 0; i < mat->rows; i++){
            for(size_t j = 0; j < mat->cols; j++){
                AT(mat, i, j) = init_value;
            }
        }        
    }

    return mat;
}

Matrix* mx_view(const Matrix* matrix, size_t rows, size_t cols, dtype default_value){
    Matrix* view = (Matrix*)malloc(sizeof(Matrix));
    if (!view) {
        printf("Failed to allocate memory for the matrix structure.");
        return NULL;
    }
    if(matrix){
        matrix->container->ref_count++;
        view->col_stride = matrix->col_stride;
        view->row_stride = matrix->row_stride;
        view->cols = rows;    // corrected to use passed rows and cols
        view->rows = cols;    // corrected to use passed rows and cols
        view->container = matrix->container;
        view->default_value = default_value;
        view->flags = 0;
    }
    else{
        view->col_stride = 1;
        view->row_stride = cols;
        view->cols = cols;
        view->rows = rows;
        view->container = NULL;
        view->default_value = default_value;
        view->flags = 0;
        SET_FLAG(view->flags, 0); // lazy matrix
    }
    return view;   
}

Matrix* mx_identity(size_t rows, size_t cols){
    if(!VALID_DIMENSIONS(rows, cols)){
        return NULL;
    }
    Matrix* m = MATRIX(rows, cols);

    size_t min_dimension = (rows < cols) ? rows : cols;
    for(size_t i = 0; i < min_dimension; i++){
        AT(m,i,i) = 1;
    }
    return m;
}


uint8_t mx_equal(Matrix* matrix1, Matrix* matrix2){
    if(!VALID_MATRIX(matrix1) || !VALID_MATRIX(matrix2)) {
        perror("Invalid matrix dimensions.");
        return 0;
    }
    if(matrix1->rows != matrix2->rows || matrix1->cols != matrix2->cols){
        return 0;
    }
    for(size_t i = 0; i < matrix1->rows; i++){
        for(size_t j =0; j<matrix1->cols; j++){
            if(AT(matrix1,i,j) != AT(matrix2,i,j)){
                return 0;
            }
        }
    }
    return 1;
}
Matrix* mx_transpose(Matrix* matrix, uint8_t flags){
    Matrix* mx_transposed;
    CHECK_MATRIX_VALIDITY(matrix);

    if(CHECK_FLAG(flags,0) == 1){
        if(matrix->rows != matrix->cols){
            perror("In-place transpose only supported for square matrices.");
            return NULL;
        }
        mx_transposed = matrix;
    }
    else if(CHECK_FLAG(flags,2) == 1){
        mx_transposed = MATRIX_COPY(matrix);
        if(!mx_transposed){
            return NULL;
        }
    }
    else {
        mx_transposed = MATRIX_VIEW(matrix);
        if(!mx_transposed){
            return NULL;
        }
    }

    mx_transposed->rows = matrix->cols;
    mx_transposed->cols = matrix->rows;
    mx_transposed->row_stride = matrix->col_stride;          
    mx_transposed->col_stride = matrix->row_stride; 

    return mx_transposed;
}

Matrix* mx_arrange(size_t rows, size_t cols, dtype start_arrange) {
    Matrix* matrix = MATRIX(rows, cols);
    if (!matrix) {
        printf("Failed to allocate memory for the matrix.\n");
        return NULL;
    }

    for(size_t i=0; i< matrix->rows; i++){
        for(size_t j=0; j< matrix->cols; j++, start_arrange++){
            AT(matrix, i, j) = start_arrange;
        }
    }

    return matrix;
}

Matrix* mx_inverse(const Matrix* matrix){
    // pass
    Matrix* m = MATRIX_VIEW(matrix);
    return m;
}

Matrix* mx_scale(Matrix* matrix, dtype scalar) {
    Matrix* result = MATRIX_COPY(matrix);
    if (!result) {
        printf("Failed to allocate memory for the scaled matrix.\n");
        return NULL;
    }

    for(size_t i = 0; i < matrix->rows; i++) {
        for(size_t j = 0; j < matrix->cols; j++) {
            AT(result, i, j) = AT(result, i, j) * scalar;
        }
    }

    return result;
}

Matrix* mx_rand(size_t rows, size_t cols) {
    Matrix* matrix = MATRIX(rows, cols);
    if (!matrix) {
        printf("Failed to allocate memory for the matrix.\n");
        return NULL;
    }

    for(size_t i = 0; i < matrix->rows; i++) {
        for(size_t j = 0; j < matrix->cols; j++) {
            dtype value = (dtype)((double)rand() / RAND_MAX);
            AT(matrix, i, j) = value;
        }
    }

    return matrix;
}

Matrix* mx_add(const Matrix* matrix1, const Matrix* matrix2){ 
    CHECK_MATRIX_VALIDITY(matrix1);
    CHECK_MATRIX_VALIDITY(matrix2);

    if (matrix1->rows != matrix2->rows || matrix1->cols != matrix2->cols) {
        printf("ERROR when 'mx_add': Sizes of two matrices should be equal.\n");
        return NULL;
    }

    Matrix* result = MATRIX(matrix1->rows, matrix1->cols);
    if (!result) {
        printf("Failed to allocate memory for the resultant matrix.\n");
        return NULL;
    }

    for (size_t i = 0; i < matrix1->rows; i++) {
        for (size_t j = 0; j < matrix1->cols; j++) {
            AT(result, i, j) = AT(matrix1, i, j) + AT(matrix2, i, j);
        }
    }

    return result;
}

Matrix* mx_subtract(const Matrix* matrix1, const Matrix* matrix2){
    CHECK_MATRIX_VALIDITY(matrix1);
    CHECK_MATRIX_VALIDITY(matrix2);

    if (matrix1->rows != matrix2->rows || matrix1->cols != matrix2->cols) {
        printf("ERROR when 'mx_subtract': Sizes of two matrices should be equal.\n");
        return NULL;
    }

    Matrix* result = MATRIX(matrix1->rows, matrix1->cols);
    if (!result) {
        printf("Failed to allocate memory for the resultant matrix.\n");
        return NULL;
    }

    for (size_t i = 0; i < matrix1->rows; i++) {
        for (size_t j = 0; j < matrix1->cols; j++) {
            AT(result, i, j) = AT(matrix1, i, j) - AT(matrix2, i, j);
        }
    }

    return result;
}

Matrix* mx_dot(Matrix* matrix1, Matrix* matrix2){
    CHECK_MATRIX_VALIDITY(matrix1);
    CHECK_MATRIX_VALIDITY(matrix2);
    
    const Matrix* actual_matrix2 = matrix2;

    if (matrix1->cols != matrix2->rows) {
        if (matrix1->cols == matrix2->cols) {
            actual_matrix2 = TRANSPOSE_VIEW(matrix2); 
            if (!actual_matrix2) {
                printf("ERROR when 'mx_dot': Unable to create transposed view of matrix2.");
                return NULL;
            }
        } else {
            printf("ERROR when 'mx_dot': Matrices are not compatible for dot product.");
            return NULL;
        }
    }

    Matrix* result = MATRIX(matrix1->rows, actual_matrix2->cols);
    if (!result) {
        if (actual_matrix2 != matrix2) {
            mx_free((Matrix*) actual_matrix2);
        }
        printf("ERROR when 'mx_dot': Unable to allocate memory for result matrix.");
        return NULL;
    }

    for(size_t i = 0; i < matrix1->rows; i++){
        for(size_t j = 0; j < actual_matrix2->cols; j++){            
            dtype sum = 0;
            for(size_t k = 0; k < matrix1->cols; k++){
                sum += AT(matrix1, i, k) * AT(actual_matrix2, k, j);
            }
            AT(result, i, j) = sum;
        }
    }

    if (actual_matrix2 != matrix2) {
        mx_free((Matrix*) actual_matrix2);
    }

    return result;
}

Matrix* mx_slice(const Matrix* src, size_t start_row, size_t end_row, size_t start_col, size_t end_col) {

    CHECK_MATRIX_VALIDITY(src);
    
    // Check for valid indices
    if (start_row > end_row || start_col > end_col || 
        end_row >= src->rows || end_col >= src->cols) {
        printf("ERROR when 'mx_slice': Invalid slice indices.\n");
        return NULL;  // Return NULL if the requested slice is invalid
    }

    size_t rows = end_row - start_row + 1;
    size_t cols = end_col - start_col + 1;

    Matrix* slice = MATRIX(rows, cols); // Assuming MATRIX(rows, cols) allocates and initializes a new matrix
    if (!slice) {
        printf("ERROR when 'mx_slice': Unable to allocate memory for slice matrix.\n");
        return NULL;
    }

    // Directly copy rows from the source matrix to the slice matrix
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            AT(slice, i, j) = AT(src, start_row + i, start_col + j);
        }
    }

    return slice;
}

Matrix* open_dataset(const char* name){
    FILE* fp = fopen(name,"r");
    if(fp == NULL){
        printf("ERROR: Incorrect filename");
        return NULL;
    }
    char line[256];

    size_t data_set_rows = 0;
    size_t data_set_cols = 0;
    
    if(!fgets(line, sizeof(line), fp)){
        printf("ERROR: Something went wrong during file reading.");
        fclose(fp); // Close the file before returning
        return NULL;
    }
    data_set_rows++;
    
    char* token = strtok(line, ",");

    while (token) {
        token = strtok(NULL, ",");
        data_set_cols++;
    }
    while(fgets(line, sizeof(line), fp)) {data_set_rows++;};

    fclose(fp);

    fp = fopen(name,"r");
    if(fp == NULL){
        printf("ERROR: Couldn't reopen the file.");
        return NULL;
    }
    
    Matrix* result = MATRIX(data_set_rows, data_set_cols);

    size_t i = 0;
    while (fgets(line, sizeof(line), fp)) {
        size_t j = 0;
        char* token = strtok(line, ",");
        while (token) {
            dtype value = atof(token);
            if (i < data_set_rows && j < data_set_cols) {
                AT(result, i, j) = value;
            } else {
                printf("ERROR: Trying to access out-of-bounds index. i: %zu, j: %zu\n", i, j);
            }
            j++;
            token = strtok(NULL, ",");
        }
        i++;
    }
    
    fclose(fp);

    return result;
}

void* mx_print(const Matrix* matrix) {
    CHECK_MATRIX_VALIDITY(matrix);
    printf("array([\n");
    for (size_t i = 0; i < matrix->rows; i++) {
        printf("[");
        for (size_t j = 0; j < matrix->cols; j++) {
            dtype value = AT(matrix,i,j);
            printf("%f", value);
            if (j < matrix->cols - 1) {
                printf(", ");
            }
        }
        printf("]");
        if (i < matrix->rows - 1) {
            printf(",");
        }

        printf("\n");
    }
    printf("])\n");
    return 0;
}
