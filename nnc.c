
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "nnc.h"


dtype sigmoidf(dtype value){
    return 1/(1+exp(-value));
}

void free_mat(Matrix *matrix) {
    if (matrix) {
        matrix->container->ref_count--;
        if (matrix->container->ref_count == 0) {
            if (matrix->container->data) {
                free(matrix->container->data);
                matrix->container->data = NULL;
            }
            free(matrix->container);
            matrix->container = NULL;
        }
        free(matrix);
        matrix = NULL;
    }
}

__matrix_container* init_container(size_t size) {
    __matrix_container* container = malloc(sizeof(__matrix_container));
    
    container->ref_count = 1;
    container->data = malloc(sizeof(dtype) * size);
    
    if (!container->data) {
        perror("Failed to allocate memory for the matrix data.");
        free(container->data);
        free(container); 
        return NULL;
    }
    return container;
}

Matrix* mat_init(size_t rows, size_t cols, dtype init_value) {

    Matrix* mat = (Matrix*)malloc(sizeof(Matrix));
    if (!mat) {
        perror("Failed to allocate memory for the matrix structure.");
        return NULL;
    }
    mat->container = init_container(cols*rows);
    
    mat->size = rows*cols;
    mat->rows = rows;
    mat->cols = cols;
    mat->container->ref_count = 1;  
    mat->row_stride = cols; 
    mat->col_stride = 1;  

    if(init_value != 0){
        for(size_t i = 0; i < mat->rows; i++){
            for(size_t j = 0; j < mat->cols; j++){
                AT(mat, i,j) = init_value;
            }
        }        
    }

    return mat;
}

Matrix* mat_view(const Matrix* matrix){
    if(!matrix){
        perror("ERROR: NULL matrix");
        return NULL;
    }
    Matrix* view = (Matrix*)malloc(sizeof(Matrix));
    if (!view) {
        perror("Failed to allocate memory for the matrix structure.");
        return NULL;
    }
    matrix->container->ref_count++;
    view->col_stride = matrix->col_stride;
    view->row_stride = matrix->row_stride;
    view->cols = matrix->cols;
    view->rows = matrix->rows;
    view->container = matrix->container;
    view->size = matrix->size;
    return view;
    
}

Matrix* mat_transpose(Matrix* matrix){

    Matrix* mat_transposed = MATRIX_VIEW(matrix);
    mat_transposed->rows = matrix->cols;
    mat_transposed->cols = matrix->rows;
    mat_transposed->row_stride = matrix->col_stride;          
    mat_transposed->col_stride = matrix->row_stride; 

    return mat_transposed;
}

Matrix* mat_arrange(size_t rows, size_t cols, dtype start_arrange){
    Matrix* matrix = MATRIX(rows, cols);
    for(size_t i=0; i< matrix->rows;i++){
        for(size_t j=0; j< matrix->cols;j++, start_arrange++){
            AT(matrix,i,j) = start_arrange;
        }
    }
    return matrix;
}

Matrix* mat_rand(size_t rows, size_t cols){
    Matrix* matrix = MATRIX(rows, cols);
    for(size_t i=0; i< matrix->rows;i++){
        for(size_t j=0; j< matrix->cols;j++){
            dtype value = (dtype)rand()/RAND_MAX*1;
            AT(matrix,i,j) = value;
        }
    }
    return matrix;
}

Matrix* mat_scale(Matrix* matrix, dtype scalar){
    Matrix* result = MATRIX_VIEW(matrix);
    for(size_t i =0; i< matrix->rows;i++){
        for(size_t j = 0; j<matrix->cols; j++){
            AT(result,i,j)=AT(result,i,j)*scalar;
        }
    }
    return result;
}

Matrix* mat_add(const Matrix* matrix1, const Matrix* matrix2){
    if(matrix1->rows != matrix2->rows || matrix1->cols != matrix2->cols){
        perror("ERROR: invalid matrices sizes");
        return NULL;
    }
    Matrix* result = MATRIX(matrix1->rows, matrix1->cols);
    for(size_t i =0;i<matrix1->size;i++){
        for(size_t j =0; j<matrix1->size;j++){
            AT(result,i,j) = AT(matrix1,i,j)+AT(matrix2,i,j);
        }
    }
    return result;
}

Matrix* mat_subtract(const Matrix* matrix1, const Matrix* matrix2){
    if(matrix1->rows != matrix2->rows || matrix1->cols != matrix2->cols){
        perror("ERROR: invalid matrices sizes");
        return NULL;
    }
    Matrix* result = MATRIX(matrix1->rows, matrix1->cols);
    for(size_t i =0;i<matrix1->size;i++){
        for(size_t j =0; j<matrix1->size;j++){
            AT(result,i,j) = AT(matrix1,i,j)-AT(matrix2,i,j);
        }
    }
    return result;
}

Matrix* mat_dot(const Matrix* matrix1, const Matrix* matrix2){
    if(matrix1->cols != matrix2->rows){
        perror("Incorrect matrix shape");
        return NULL;
    }
    Matrix* result = MATRIX(matrix1->rows, matrix2->cols);
    for(size_t i=0; i<matrix1->rows; i++){
        for(size_t j=0; j<matrix2->cols; j++){            
            dtype sum = 0;
            for (size_t k = 0; k < matrix1->cols; k++) {
                sum += AT(matrix1, i, k) * AT(matrix2, k, j);
            }
            AT(result, i, j) = sum;
        }
    }
    
    return result;
}

Matrix* open_dataset(const char* name){
    FILE* fp = fopen(name,"r");
    if(fp == NULL){
        return NULL;
    }
    char line[256];

    size_t data_set_rows = 0;
    size_t data_set_cols = 0;
    
    if(!fgets(line, sizeof(line), fp)){
        perror("ERROR: Something went wrong during file openning.");
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
    Matrix* result = MATRIX(data_set_rows, data_set_cols);

    size_t i = 0;
    while (fgets(line, sizeof(line), fp)) {
        size_t j = 0;
        char* token = strtok(line, ",");
        while (token) {
            dtype value = atof(token);
            AT(result, i, j) = value;
            j++;
            token = strtok(NULL, ",");
        }
        i++;
    }
    
    fclose(fp);

    return result;
}
Matrix* mat_slice(const Matrix* src, size_t start_row, size_t end_row, size_t start_col, size_t end_col) {
    if (start_row > end_row || start_col > end_col || 
        end_row >= src->rows || end_col >= src->cols) {
        return NULL;  // Return NULL if the requested slice is invalid
    }

    size_t rows = end_row - start_row+1;
    size_t cols = end_col - start_col+1;
    
    Matrix* slice = MATRIX(rows,cols); // Assuming MATRIX(rows,cols) allocates and initializes a new matrix
    
    // Directly copy rows from the source matrix to the slice matrix
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            AT(slice, i, j) = AT(src, start_row + i, start_col + j);
        }
    }

    return slice;
}

void print_mat(const Matrix* matrix) {
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
}
